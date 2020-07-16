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

def main(param_args):
    data_graph = tf.Graph()
    batch_size = param_args['batch_size']
    input_height = param_args['input_height']
    input_width = param_args['input_width']
    num_inter_threads = param_args['num_inter_threads']
    num_intra_threads = param_args['num_intra_threads']
    data_location = param_args['data_location']
    with data_graph.as_default(): ###
      dataset = dataset.ImagenetData(data_location)
      preprocessor = image_preprocessing.ImagePreprocessor(
          input_height, input_width, batch_size,
          1, # device count
          tf.float32, # data_type for input fed to the graph
          train=False, # doing inference
          resize_method='crop')
      images, labels = preprocessor.minibatch(dataset, subset='validation')

    graph = load_graph(model_file)

    input_tensor = graph.get_tensor_by_name(input_layer + ":0")
    output_tensor = graph.get_tensor_by_name(output_layer + ":0")

    rewrite_options = rewriter_config_pb2.RewriterConfig(
            layout_optimizer=rewriter_config_pb2.RewriterConfig.ON)
    config = tf.compat.v1.ConfigProto()
    if (args.gpu < 0):
        config.inter_op_parallelism_threads = num_inter_threads
        config.intra_op_parallelism_threads = num_intra_threads
    config.graph_options.rewrite_options.remapping = (
            rewriter_config_pb2.RewriterConfig.OFF)

    data_config = tf.compat.v1.ConfigProto()
    data_config.inter_op_parallelism_threads = num_inter_threads
    data_config.intra_op_parallelism_threads = num_intra_threads
    
    data_sess = tf.compat.v1.Session(graph=data_graph, config=data_config)

    with tf.compat.v1.Session(graph=graph, config=config) as sess:
        sys.stdout.flush()
        print("[Running warmup steps...]")
        image_data = data_sess.run(images)
        start_time = time.time()
        sess.run(output_tensor, {input_tensor: image_data})
        elapsed_time = time.time() - start_time
        avg = 0
        print("[Running benchmark steps...]")
        total_time   = 0
        total_images = 0
        start_time = time.time()
        results = sess.run(output_tensor, {input_tensor: image_data})
        elapsed_time = time.time() - start_time
        avg += elapsed_time

classIds = []
confidences = []
boxes = []

class TensorFlow:

    def __init__(self, param_args={}):
        self.param_args = param_args

    def infer(self):
        pass

    def predict(self):
        pass

    