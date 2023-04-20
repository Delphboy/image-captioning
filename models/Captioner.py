from typing import Optional
import torch
import torch.nn as nn
from abc import ABC


class BaseCaptioner(ABC, nn.Module):
    def __init__(self,
                 embedding_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_decoder_layers: int) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_decoder_layers = num_decoder_layers

        self.encoder: Optional[nn.Module]
        self.decoder: Optional[nn.Module]


    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs      


    def caption_image(self, image, vocab, max_length=50):
        with torch.no_grad():
            x = self.encoder(image)
            states = None

            result = []
            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(x, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(0)
                result.append(predicted.item())

                x = self.decoder.embedding(predicted).unsqueeze(0)
                if vocab.itos[f"{predicted.item()}"] == "<EOS>":
                    break
            return [vocab.itos[f"{idx}"] for idx in result]

