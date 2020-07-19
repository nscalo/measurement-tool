import cv2

class SSDVideoInference:

    def __init__(self, video_file, inference_object):
        self.video_file = video_file
        self.inference_object = inference_object

    def obtain_bounding_boxes(self, image_data, result, confidence_level=0.5):
        boxes, conf = [], []
        if self.framework == "caffe" or self.framework == "caffe2" or self.framework == "onnx" or self.framework == "openvino":
            boxes, conf = self.inference_object.preprocess_bounding_box_ssd(image_data, result, confidence_level=0.5)
        elif self.framework == "pytorch":
            boxes, conf = self.inference_object.preprocess_bounding_box_ssd_tensor(image_data, result, confidence_level=0.5)
        elif self.framework == "tensorflow":
            boxes, conf = self.inference_object.preprocess_bounding_box_ssd_tensorflow(image_data, result, confidence_level=0.5)

        return boxes, conf

    def write_video(self, boxes, conf, frame):
        for box in boxes:
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 1)

    