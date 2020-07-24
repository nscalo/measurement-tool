import os
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import numpy as np

class Caffe2:

    def __init__(self, predict_net, init_net=None, is_run_init=True, is_create_net=True):
        self.net_file = predict_net
        self.init_file = init_net
        self.is_run_init = is_run_init
        self.is_create_net = is_create_net
    
    def load_model(self):
        self.net = core.Net("net")
        if self.net_file is not None:
            self.net.Proto().ParseFromString(open(self.net_file, "rb").read())

        if self.init_file is None:
            fn, ext = os.path.splitext(self.net_file)
            self.init_file = fn + "_init" + ext

        self.init_net = caffe2_pb2.NetDef()
        self.init_net.ParseFromString(open(self.init_file, "rb").read())

        if self.is_run_init:
            workspace.RunNetOnce(self.init_net)
            if self.net.Proto().name == "":
                self.net.Proto().name = "net"
        if self.is_create_net:
            workspace.CreateNet(self.net)

        return (self.net, self.init_net)

    def predict(self, input_blob):
        input_blob = input_blob['data']
        p = workspace.Predictor(self.init_net, self.net)
        input_blob = input_blob.transpose((2,0,1))
        input_blob = np.expand_dims(input_blob, 0)
        input_blob = input_blob.astype(np.float32)
        return p.run([input_blob])
