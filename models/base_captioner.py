from typing import Optional
import torch
import torch.nn as nn
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from constants import Constants as const

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


    def forward(self, input_features, captions):
        features = self.encoder(input_features)
        outputs = self.decoder(features, captions)
        return outputs      


    def caption_image(self, input_features, vocab, max_length=20, method='beam_search'):
        assert method in ['greedy', 'beam_search']
        if method == 'greedy':
            return self.greedy_caption(input_features, vocab, max_length)
        else:
            return self.beam_search_caption(input_features, vocab, max_length)
        
    
    def greedy_caption(self, input_features, vocab, max_length=50):
        with torch.no_grad():
            x = self.encoder(input_features)
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


    @torch.no_grad()
    def beam_search_caption(self, input_features, vocab, max_length=30, beam_size=1):
        input_features = self.encoder(input_features)

        beam = [(torch.tensor([vocab.stoi["<SOS>"]]).to(const.DEVICE),
                  0.0,
                  input_features,
                  None)]
        completed_captions = []

        for _ in range(max_length):
            candidates = []
            for sequence, probability, old_hidden, old_state in beam:
                if vocab.itos[f"{sequence[-1].item()}"] == "<EOS>":
                    completed_captions.append((sequence, probability))
                    continue
                hiddens, states = self.decoder.lstm(old_hidden, old_state)
                output = self.decoder.linear(hiddens.squeeze(0))
                output = F.softmax(output, dim=-1)
                topk = output.topk(beam_size)
                for i in range(beam_size):
                    candidates.append((torch.cat([sequence, topk[1][i].unsqueeze(0)]),
                                       probability + topk[0][i].item(),
                                       self.decoder.embedding(topk[1][i]).unsqueeze(0),
                                       states))
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            beam = candidates[:beam_size]
            if len(completed_captions) == beam_size:
                break

        completed_captions.extend(beam)
        completed_captions = sorted(completed_captions, key=lambda x: x[1], reverse=True)
        return [vocab.itos[f"{idx.item()}"] for idx in completed_captions[0][0]]

