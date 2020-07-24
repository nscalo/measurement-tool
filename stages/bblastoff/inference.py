import tensorflow as tf
from time import time
from stages.anetwork.onnx import ONNX
from stages.anetwork.pytorch import PyTorch
from stages.anetwork.tensorlib import TensorFlow

class Inference:

    def __init__(self, framework=''):
        self.framework = framework

    def create_instance(self, *args, **kwargs):
        if self.framework == "caffe":
            from stages.anetwork.caffe import Caffe
            self.model_object = Caffe(*args, **kwargs)
        elif self.framework == "caffe2":
            from stages.anetwork.caffe2 import Caffe2
            self.model_object = Caffe2(*args, **kwargs)
        elif self.framework == "onnx":
            self.model_object = ONNX(*args, **kwargs)
        elif self.framework == "openvino":
            self.model_object = OpenVINO(*args, **kwargs)
        elif self.framework == "pytorch":
            self.model_object = PyTorch(*args, **kwargs)
        elif self.framework == "tensorflow":
            self.model_object = TensorFlow(*args, **kwargs)

    def load_model(self, **kwargs):
        if self.framework == "openvino":
            self.model_object.load_core(**kwargs)
            return self.model_object.load_model(**kwargs)
        else:
            return self.model_object.load_model(**kwargs)

    def process(self, *args, **kwargs):
        start_time = time()
        if self.framework == "caffe":
            self.results = self.model_object.get_output(*args, **kwargs)
        elif self.framework == "caffe2":
            self.results = self.model_object.predict(*args, **kwargs)
        elif self.framework == "onnx":
            self.results = self.model_object.get_session(*args, **kwargs)
        elif self.framework == "openvino":
            self.model_object.sync_inference(*args, **kwargs)
            self.results = self.model_object.extract_output()
        elif self.framework == "pytorch":
            self.model_object.infer(*args, **kwargs)
            self.results = self.model_object.predict(*args, **kwargs)
        elif self.framework == "tensorflow":
            self.results = self.model_object.predict(*args, **kwargs)
        end_time = time()

        return end_time - start_time

class SSDInference(Inference):

    def __init__(self, **kwargs):
        super(SSDInference, self).__init__(**kwargs)

    def preprocess_bounding_box_ssd(self, image_data, result, confidence_level=0.5):
        identified = False
        boxes = []
        confs = []
        height, width, _ = image_data[0].shape
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

    def preprocess_bounding_box_ssd_heatmap(self, image_data, result):
        identified = False
        boxes = []
        confs = []
        height, width, _ = image_data[0].shape
        # for box in result[0][0]: # Output shape is 1x1x100x7
        #     xmin = int(box[3] * width)
        #     ymin = int(box[4] * height)
        #     xmax = int(box[5] * width)
        #     ymax = int(box[6] * height)
        #     boxes.append([xmin, ymin, xmax, ymax])
        #     confs.append(conf)
        
        return boxes, confs

    def preprocess_bounding_box_ssd_tensor(self, image_data, result, confidence_level=0.5):
        result = result[self.model_object.output_blob[0]].numpy().detach().astype(np.float64)
        identified = False
        boxes = []
        confs = []
        height, width, _ = image_data[0].shape
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

    def preprocess_bounding_box_ssd_tensorflow(self, image_data, result, confidence_level=0.5):
        identified = False
        boxes = []
        confs = []
        with tf.compat.v1.Session(graph=self.model_object.graph, config=self.model_object.config) as sess:
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