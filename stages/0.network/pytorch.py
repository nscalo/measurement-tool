from torch import nn

def load_model(encoder, model_file):
    checkpoint = torch.load(model_file)
    encoder.load_state_dict(checkpoint['model_state_dict'])

    return encoder, checkpoint

class PyTorch:

    def __init__(self, encoder: nn.Module, model_file, output_blob):
        encoder, checkpoint = load_model(encoder, model_file)
        self.encoder = encoder
        self.checkpoint = checkpoint
        if (type(output_blob) != tuple) or (type(output_blob) != list):
            self.output_blob = list(output_blob)

    def infer(self, **kwargs):
        self.output_tuple = self.encoder.forward(**kwargs)

    def predict(self):
        if (type(self.output_tuple) != tuple) or (type(self.output_tuple) != list):
            self.output_tuple = list(self.output_tuple)
        
        return zip(self.output_blob, self.output_tuple)
