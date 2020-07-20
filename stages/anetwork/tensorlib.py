import tensorflow as tf
import sys
import dataset
from tf_text_graph_common import readTextMessage
from tf_text_graph_ssd import createSSDGraph
from tf_text_graph_faster_rcnn import createFasterRCNNGraph

class TensorFlow:

    def __init__(self, param_args={}):
        self.param_args = param_args

    def load_model(self, model_file):
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
        self.graph = graph
        self.sess = tf.InteractiveSession(graph = self.graph)
        
        return self.graph

    def predict(self, sess, input_dict, input_tensor):
        output_tensor = self.graph.get_tensor_by_name(str(input_tensor) + ":0")
        output = self.sess.run(output_tensor, feed_dict = input_dict)

        return output
    