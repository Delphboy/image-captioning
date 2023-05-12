from typing import Optional
import torch
import torch.nn as nn


class Lstm(nn.Module):
    def __init__(self, 
                 embedding_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_layers: int, 
                 max_seq_length: Optional[int]=20):
        
        super(Lstm, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_length

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True,
                            dropout=0.1)
        self.linear = nn.Linear(hidden_size, vocab_size)


    def forward(self, features, captions):
        features = features.unsqueeze(1)
        
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features, embeddings), dim=1)
        
        hidden, _ = self.lstm(embeddings)
        outputs = self.linear(hidden)
        return outputs


    def sample(self, features, states=None):
        sampled_ids = []
        if len(features.shape) == 2:    
            inputs = features.unsqueeze(1)
        else:
            inputs = features
        
        for i in range(self.max_seq_length):
            hiddens, states = self.lstm(inputs, states)          
            outputs = self.linear(hiddens.squeeze(1))            
            predicted = outputs.argmax(1)                        
            sampled_ids.append(predicted)
            inputs = self.embedding(predicted)                   
            inputs = inputs.unsqueeze(1)                         
        sampled_ids = torch.stack(sampled_ids, 1)                
        return sampled_ids

