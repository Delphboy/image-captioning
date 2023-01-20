import torch
import torch.nn as nn

from models.components.vision.cnn import InceptionV3
from models.components.language.lstm import Lstm

class BasicCaptioner(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(BasicCaptioner, self).__init__()
        self.cnn = InceptionV3(embed_size)
        self.lstm = Lstm(embed_size, hidden_size, vocab_size, num_layers)


    def forward(self, images, captions):
        features = self.cnn(images)
        outputs = self.lstm(features, captions)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            x = self.cnn(image).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.lstm.lstm(x, states)
                output = self.lstm.linear(hiddens.squeeze(0))
                predicted = output.argmax()#1)
                result_caption.append(predicted.item())
                x = self.lstm.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]