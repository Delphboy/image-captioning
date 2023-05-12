from typing import List, Optional, Tuple
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


    @torch.no_grad()
    def caption_image(self, input_features, vocab, max_length=20, method='greedy'):
        assert method in ['greedy', 'beam_search']
        if method == 'greedy':
            outputs, _ = self.greedy_caption(input_features, vocab, max_length)
            return outputs[0]
        else:
            outputs, _ = self.beam_search_caption(input_features, vocab, max_length)
            return outputs[0]
        
    
    # TODO: Parallelise this properly
    # @torch.no_grad()
    def greedy_caption(self, 
                       input_features, 
                       vocab, 
                       max_length=50) -> List[str]:
        x = self.encoder(input_features)
        states = None

        result = []
        probabilities = []
        for _ in range(max_length):
            if len(x.shape) == 1: x = x.unsqueeze(0)
            hiddens, states = self.decoder.lstm(x, states)
            logits = self.decoder.linear(hiddens.squeeze(0))
            predicted = logits.argmax(-1)
            
            result.append(predicted)
            probability = torch.max(F.softmax(logits, dim=-1), dim=-1)[0]
            probabilities.append(probability)

            x = self.decoder.embedding(predicted)
        probabilities = torch.stack(probabilities, 0).T
        probabilities = probabilities.sum(-1).tolist()

        caption_strings = []
        result = torch.stack(result, 0)

        caption_strings = []
        if len(result.shape) == 1: 
            caption_strings.append([vocab.itos[f"{idx.item()}"] for idx in result])
        else:
            for i in range(result.shape[-1]):
                cap_str = [vocab.itos[f"{idx.item()}"] for idx in result[:, i]]
                caption_strings.append(cap_str)

        return caption_strings, probabilities


    # TODO: Parallelise this properly
    # @torch.no_grad()
    def beam_search_caption(self, 
                            input_features, 
                            vocab, 
                            max_length=30, 
                            beam_size=3) -> Tuple[List[List[str]], List[float]]:
        input_features = self.encoder(input_features)

        batch_caps = []
        batch_probs = []

        for batch_index in range(input_features.shape[0]):
            beam = [(torch.tensor([vocab.stoi["<SOS>"]]).to(const.DEVICE),
                    torch.tensor(0.0),
                    input_features[batch_index].unsqueeze(0),
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
            batch_caps.append([vocab.itos[f"{idx.item()}"] for idx in completed_captions[0][0]])
            batch_probs.append(completed_captions[0][1])
        
        
        
        return batch_caps, batch_probs

