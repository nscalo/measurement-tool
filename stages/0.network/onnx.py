import onnxruntime as nxrun

class ONNX:

    def __init__(self, input_model, input_dict):
        self.input_model = input_model
        self.input_dict = input_dict

    def load_model(self):
        self.sess = nxrun.InferenceSession(self.input_model)

        return self.sess

    def get_session(self, data):
        result = self.sess.run(None, self.input_dict)

        return result