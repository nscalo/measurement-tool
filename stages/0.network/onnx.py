import onnxruntime as nxrun

class ONNX:

    def __init__(self, input_model, input_dict):
        self.input_model = input_model
        self.input_dict = input_dict

    def get_session(self, data):
        sess = nxrun.InferenceSession()

        result = sess.run(None, self.input_dict)

        return result