import os
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

class Caffe2:

    def __init__(self, predict_net, init_net=None, is_run_init=True, is_create_net=True):
        self.net_file = predict_net
        self.init_file = init_net
        self.is_run_init = is_run_init
        self.is_create_net = is_create_net
    
    def load_model(self):
        net = core.Net("net")
        if net_file is not None:
            net.Proto().ParseFromString(open(net_file, "rb").read())

        if init_file is None:
            fn, ext = os.path.splitext(net_file)
            init_file = fn + "_init" + ext

        init_net = caffe2_pb2.NetDef()
        init_net.ParseFromString(open(init_file, "rb").read())

        if is_run_init:
            workspace.RunNetOnce(init_net)
            create_blobs_if_not_existed(net.external_inputs)
            if net.Proto().name == "":
                net.Proto().name = "net"
        if is_create_net:
            workspace.CreateNet(net)

        return (net, init_net)

    def predict(self, img):

        p = workspace.Predictor(init_net, predict_net)
        p.run({'data': img})

    