from typing import Optional
import torch
import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, 
                 embed_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_layers: int, 
                 max_seq_length: Optional[int]=20):
        
        super(Lstm, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seg_length = max_seq_length

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size=self.embed_size, 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers, 
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)


    def forward(self, features, captions):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)

        return outputs


    def sample(self, features, states=None):
        sampled_ids = []
        if len(features.shape) == 2:    
            inputs = features.unsqueeze(1)
        else:
            inputs = features
        
        for i in range(self.max_seg_length):
            hiddens, states = self.lstm(inputs, states)          # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # outputs:  (batch_size, vocab_size)
            predicted = outputs.argmax(1)                        # predicted: (batch_size)
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)                       # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                         # inputs: (batch_size, 1, embed_size)
        sampled_ids = torch.stack(sampled_ids, 1)                # sampled_ids: (batch_size, max_seq_length)
        return sampled_ids