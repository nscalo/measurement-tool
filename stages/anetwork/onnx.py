import onnxruntime as nxrun

class ONNX:

    def __init__(self, input_model):
        self.input_model = input_model

    def load_model(self):
        self.sess = nxrun.InferenceSession(self.input_model)
        return self.sess

    def get_session(self, input_blob):
        self.input_dict = {'input_blob': input_blob}
        output_name = self.sess.get_outputs()[0].name
        result = self.sess.run([output_name], input_blob)

        return result