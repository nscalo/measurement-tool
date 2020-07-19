from torch import nn

class PyTorch:

    def __init__(self, encoder: nn.Module, output_blob):
        self.encoder = encoder
        if (type(output_blob) != tuple) or (type(output_blob) != list):
            self.output_blob = list(output_blob)

    def load_model(self, encoder, model_file):
        self.checkpoint = torch.load(model_file)
        self.encoder.load_state_dict(checkpoint['model_state_dict'])

        return encoder, checkpoint

    def infer(self, **kwargs):
        self.output_tuple = self.encoder.forward(**kwargs)

    def predict(self):
        if (type(self.output_tuple) != tuple) or (type(self.output_tuple) != list):
            self.output_tuple = list(self.output_tuple)
        
        return zip(self.output_blob, self.output_tuple)
