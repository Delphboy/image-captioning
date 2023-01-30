import torch
import torch.nn as nn

from models.components.vision.cnn import InceptionV3, Resnet152
from models.components.language.lstm import LstmWithDropOut, Lstm

class CaptionWithInceptionV3AndLstmDropout(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithInceptionV3AndLstmDropout, self).__init__()
        self.cnn = InceptionV3(embed_size)
        self.lstm = LstmWithDropOut(embed_size, hidden_size, vocab_size, num_layers)


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
                squeeze = hiddens.squeeze(0)
                output = self.lstm.linear(squeeze)
                predicted = output.argmax(-1)#Was 1
                result_caption.append(predicted.item())
                x = self.lstm.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]


class CaptionWithResnet152AndLstm(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CaptionWithResnet152AndLstm, self).__init__()
        self.encoderCNN = Resnet152(embed_size)
        self.decoderRNN = Lstm(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions, lengths)
        return outputs


    def caption_image(self, image, vocabulary, max_length=50):
        features = self.encoderCNN(image).unsqueeze(0)
        result_caption = self.decoderRNN.sample(features)[0]
        
        return [vocabulary.itos[idx.item()] for idx in result_caption]