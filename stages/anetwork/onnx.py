import onnxruntime as nxrun

class ONNX:

    def __init__(self, input_model):
        self.input_model = input_model

    def load_model(self):
        self.sess = nxrun.InferenceSession(self.input_model)
        return self.sess

    def get_session(self, data):
        self.input_dict = data
        result = self.sess.run(None, self.input_dict)

        return result