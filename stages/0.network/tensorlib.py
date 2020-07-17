import tensorflow as tf
import sys
import dataset
from tf_text_graph_common import readTextMessage
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_faster_rcnn import createFasterRCNNGraph

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.compat.v1.GraphDef()

  import os
  file_ext = os.path.splitext(model_file)[1]

  with open(model_file, "rb") as f:
    if file_ext == '.pbtxt':
      text_format.Merge(f.read(), graph_def)
    else:
      graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def, name='')
  return graph

class TensorFlow:

    def __init__(self, param_args={}):
      self.param_args = param_args

    def predict(self, params):
      bbox, label, image_id, features = params
      step = idx
      features, bbox, label, image_id = \
      tuple(features.items()), bbox, label, image_id
      if features is None:
        return
      bbox, label, image_id = sess.run([bbox, label, image_id])
    
    def preprocess_bounding_box_ssd(self, image_data, result, confidence_level=0.5):
      identified = False
      boxes = []
      confs = []
      with tf.compat.v1.Session(graph=self.graph, config=self.config) as sess:
        images = image_data.eval()
        images = sess.run([image_data])
        height, width, _ = images[0].shape
        for box in result[0][0]: # Output shape is 1x1x100x7
            conf = box[2]
            if conf >= confidence_level:
                xmin = int(box[3] * width)
                ymin = int(box[4] * height)
                xmax = int(box[5] * width)
                ymax = int(box[6] * height)
                boxes.append([xmin, ymin, xmax, ymax])
                confs.append(conf)
      return boxes, confs