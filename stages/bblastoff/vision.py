from __future__ import division

import tensorflow as tf
import time
import cv2
import PIL
import re
import numpy as np
import random
from pprint import PrettyPrinter
from copy import copy
from argparse import ArgumentParser
from .coco_detection_evaluator import CocoDetectionEvaluator
from .face_label_map import category_map
from tensorflow.python.data.experimental import parallel_interleave
from tensorflow.python.data.experimental import map_and_batch
from .inference import *
import logging
import os
import numpy as np

IMAGE_SIZE = (224,224)
COCO_NUM_VAL_IMAGES = 2008
os.environ['GLOG_minloglevel'] = '1'

def parse_and_preprocess(serialized_example):
  # Dense features in Example proto.
  feature_map = {
      'image/object/class/text': tf.compat.v1.FixedLenFeature([], 
        dtype=tf.string, default_value=''),
      'image/source_id': tf.compat.v1.FixedLenFeature([], dtype=tf.string, default_value='')
  }
  sparse_int64 = tf.compat.v1.VarLenFeature(dtype=tf.int64)
  # Sparse features in Example proto.
  feature_map.update(
      {k: sparse_int64 for k in ['image/object/bbox/xmin',
                                   'image/object/bbox/ymin',
                                   'image/object/bbox/xmax',
                                   'image/object/bbox/ymax']})

  features = tf.compat.v1.parse_single_example(serialized_example, feature_map, name='features')

  xmin = tf.expand_dims(features['image/object/bbox/xmin'].values, 0)
  ymin = tf.expand_dims(features['image/object/bbox/ymin'].values, 0)
  xmax = tf.expand_dims(features['image/object/bbox/xmax'].values, 0)
  ymax = tf.expand_dims(features['image/object/bbox/ymax'].values, 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.cast(tf.concat([ymin, xmin, ymax, xmax], 0), dtype=tf.int64)

  # Force the variable number of bounding boxes into the shape
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  label = features['image/object/class/text']

  image_id = features['image/source_id']

  return bbox[0], label, image_id, features


def preprocessing(images):
  images = cv2.resize(images, IMAGE_SIZE)
  images = images.transpose((2,0,1))
  images = np.expand_dims(images,0)
  return images

def exec_evaluator(ground_truth_dicts, detect_dicts, image_id_gt_dict,
total_iter, batch_size):
  import sys
  
  if "numpy" in sys.modules:
    del sys.modules["numpy"]
  if "np" in sys.modules:
    del sys.modules["np"]
  import numpy as np
  
  def setUp():
    np.round = round
  
  from coco_detection_evaluator import CocoDetectionEvaluator

  setUp()
  
  evaluator = CocoDetectionEvaluator()
  for step in range(total_iter):
    # add single ground truth info detected
    # add single image info detected
    if step in list(detect_dicts.keys()):
      evaluator.add_single_ground_truth_image_info(image_id_gt_dict[step], ground_truth_dicts[step])
      evaluator.add_single_detected_image_info(image_id_gt_dict[step], detect_dicts[step])

  if (step + 1) * batch_size >= COCO_NUM_VAL_IMAGES:
    metrics = evaluator.evaluate()
  
  if metrics:
    pp = PrettyPrinter(indent=4)
    self.logfile.info("Metrics:\n" + pp.pprint(metrics))

  self.logfile.info("Detection Dicts:\n" + pp.pprint(detect_dicts))

class ParamArgs():
    def __init__(self, param_dict):
        update_param_dict(param_dict, self)

def update_param_dict(param_dict, obj):
  for key, value in param_dict.items():
    obj.__setattr__(key, value)

def obtain_args(param_dict):
    return ParamArgs(param_dict)

class model_infer(object):

  need_reshape = False

  def __init__(self, param_dict, inference_object, factor=1e-1, output_log="output_log.log"):
    # parse the arguments
    self.args = obtain_args(param_dict)
    self.inference_object = inference_object
    self.factor = factor

    self.logfile = logging.basicConfig(filename=output_log, filemode='w', level=logging.INFO)

    self.config_dict = dict()
    self.config_dict['ARCFACE_PREBATCHNORM_LAYER_INDEX']=-3
    self.config_dict['ARCFACE_POOLING_LAYER_INDEX']=-4

    self.config = tf.ConfigProto()
    self.config.intra_op_parallelism_threads = self.args.num_intra_threads
    self.config.inter_op_parallelism_threads = self.args.num_inter_threads
    self.config.use_per_session_threads = 1

    self.setUp()

    if self.args.batch_size == -1:
      self.args.batch_size = 1

  def setUp(self):
    np.round = round
  
  def run_inference(self, params):
    self.logfile.info("Inference for accuracy check.")
    total_iter = COCO_NUM_VAL_IMAGES
    fm = category_map
    fm = dict(zip(list(fm.values()),list(fm.keys())))
    self.logfile.info('total iteration is {0}'.format(str(total_iter)))
    result = []
    global model, graph
    with tf.Session().as_default() as sess:
      if self.args.data_location:
        self.build_data_sess()
      else:
        raise Exception("no data location provided")
      evaluator = CocoDetectionEvaluator()
      total_samples = 0
      self.coord = tf.train.Coordinator()
      tfrecord_paths = [self.args.data_location]
      ds = tf.data.TFRecordDataset.list_files(tfrecord_paths)
      ds = ds.apply(
        parallel_interleave(
          tf.data.TFRecordDataset, cycle_length=1, block_length=1,
          buffer_output_elements=10000, prefetch_input_elements=10000))
      ds = ds.prefetch(buffer_size=10000)
      ds = ds.apply(
          map_and_batch(
            map_func=parse_and_preprocess,
            batch_size=self.args.batch_size,
            num_parallel_batches=1,
            num_parallel_calls=None))
      ds = ds.prefetch(buffer_size=10000)
      ds_iterator = tf.data.make_one_shot_iterator(ds)
      state = None
      warmup_iter = 0
      
      self.ground_truth_dicts = {}
      self.detect_dicts = {}
      self.total_iter = total_iter
      self.image_id_gt_dict = {}

      obj = self
      if self.args.data_location:
        for idx in range(total_iter):
          bbox, label, image_id, features = ds_iterator.get_next()
          result.append((bbox, label, image_id, features))
          
      for idx in range(total_iter):
        run_ice_breaker_session(result, obj, 
        params, fm, sess, total_iter, idx)

def run_ice_breaker_session(result, obj, params, 
fm, sess, total_iter, idx):
  # ground truth of bounding boxes from pascal voc
  ground_truth = {}
  inference_object = self.inference_object
  model_object = self.inference_object.model_object
  label_gt = [fm[l] if type(l) == 'str' else fm[l.decode('utf-8')] for l in label]
  image_id_gt = [i if type(i) == 'str' else i.decode('utf-8') for i in image_id]
  ground_truth['classes'] = np.array(label_gt*len(ground_truth['boxes']))
  # saving all ground truth dictionaries
  images = np.asarray(PIL.Image.open(os.path.join(obj.args.imagesets_dir, image_id_gt[0])).convert('RGB'))
  
  images = preprocessing(images)

  result = inference_object.process(**params)
  # detected conventional bounding box same as ground truth bounding boxes
  boxes, confs = model_object.preprocess_bounding_box_ssd(images, result, confidence_level=0.4)

  # object detection
  detect = copy(ground_truth)

  # detection for bounding boxes from pascal voc
  label_det = label_gt

  if len(boxes) > 0:
    detect['boxes'] = np.asarray(boxes)
    detect['classes'] = np.asarray(label_det*len(detect['boxes']))

    # 1, 1000, 1, 1
    detect['scores'] = np.asarray(confs)
    obj.detect_dicts[step] = detect
    ground_truth['boxes'] = detect['boxes'] - (obj.regularization_parameter)**2 * detect['boxes'] * self.factor
    obj.ground_truth_dicts[step] = ground_truth
    obj.image_id_gt_dict[step] = image_id_gt[0]
