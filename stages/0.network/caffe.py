import os
import sys
import logging as log
import caffe
import numpy as np

class Caffe:

    def __init__(self, input_graph, input_weights, in_blob_name, 
    out_blob_name, need_reshape=False):
        self.model = str(self.input_graph)
        self.weights = str(self.input_weights)
        self.in_blob_name = in_blob_name
        self.out_blob_name = out_blob_name
        self.need_reshape = need_reshape

    def load_model(self):
        caffe.set_mode_cpu()

        self.network = caffe.Net(self.model, self.weights, caffe.TEST)

    def get_output(self, input_blob):
        if self.need_reshape:
            self.network.blobs[self.in_blob_name].reshape(*input_blob.shape)
    
        return self.network.forward_all(**{self.in_blob_name: input_blob})[self.out_blob_name]

    