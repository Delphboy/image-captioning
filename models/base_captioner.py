from typing import Optional
import torch
import torch.nn as nn
from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
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
            return self.greedy_caption(input_features, vocab, 12)
        else:
            return self.beam_search_caption(input_features, vocab, 12)
        
    
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
    def beam_search_caption(self, input_features, vocab, max_length=30, beam_size=3):
        class Node:
            def __init__(self, x, states, log_prob, sequence, end):
                self.x = x
                self.states = states
                self.log_prob = log_prob
                self.sequence = sequence
                self.end = end
        
        
        def predict(node, beam_size):
            if node.end:
                return [node]
            hiddens, states = self.decoder.lstm(node.x, node.states)
            output = self.decoder.linear(hiddens.squeeze(0))
            output = F.softmax(output, dim=-1)
            top_probs, top_words = output.topk(beam_size)
            children = []
            for i in range(beam_size):
                sequence = node.sequence.copy()
                sequence.append(top_words[i].item())
                x = self.decoder.embedding(top_words[i]).unsqueeze(0)
                new_node = Node(x, 
                                states, 
                                node.log_prob + top_probs[i], 
                                sequence, 
                                vocab.itos[f"{top_words[i].item()}"] == "<EOS>" or len(sequence) == max_length)
                children.append(new_node)
            return children
        

        root = Node(self.encoder(input_features), None, 0, [], False)
        queue = [root]        

        tree_level = 1
        while tree_level < max_length:
            new_queue = []
            for node in queue:
                new_queue.extend(predict(node, beam_size))
            queue = new_queue
            tree_level += 1


        end_nodes = [node for node in queue if node.end]
        # return the sequence with the highest log_prob
        sorted_end_nodes = sorted(end_nodes, key=lambda x: x.log_prob, reverse=True)
        return [vocab.itos[f"{idx}"] for idx in sorted_end_nodes[0].sequence]

