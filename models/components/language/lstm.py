import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from constants import Constants as const

TEACHER_FORCING_RATIO = 1.0

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


# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/models.py#L54
class Attention(nn.Module):
    def __init__(self, 
                 encoder_dim: int, 
                 decoder_dim: int, 
                 attention_dim: int):
        super(Attention, self).__init__()
        self.encoder_transform = nn.Linear(encoder_dim, attention_dim)
        self.decoder_transform = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.tanh = nn.Tanh()


    def forward(self, 
                image_features, 
                lang_hidden):
        W_s = self.encoder_transform(image_features)
        U_h = self.decoder_transform(lang_hidden).unsqueeze(1)
        att = self.tanh(W_s + U_h)
        e = self.full_att(att).squeeze(2)
        alpha = F.softmax(e, dim=1)
        context = (image_features * alpha.unsqueeze(2)).sum(1)
        return context, alpha


# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/cc9c7e2f4017938d414178d3781fed8dbe442852/models.py#L89
class AttentionLstm(nn.Module):
    def __init__(self, 
                 embedding_size=2048, 
                 hidden_size=2048, 
                 vocab_size=10000,
                 num_layers=1,
                 max_seq_length=20):
        super(AttentionLstm, self).__init__()

        self.encoder_dim = embedding_size
        self.attention_dim = hidden_size
        self.embed_dim = hidden_size
        self.decoder_dim = embedding_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_length

        self.attention = Attention(self.encoder_dim, self.decoder_dim, self.attention_dim)  # attention network

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)  # embedding layer

        self.lstm_cell = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)  # decoding LSTMCell
        
        self.init_h = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(self.encoder_dim, self.decoder_dim)  # linear layer to find initial cell state of LSTMCell
        
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution


    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)


    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        hidden = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        cell_state = self.init_c(mean_encoder_out)
        return hidden, cell_state


    def forward(self, 
                features, 
                captions, 
                caption_lengths):
        is_teacher_forcing = True if random.random() < TEACHER_FORCING_RATIO else False

        batch_size = features.size(0)
        # features = features.view(batch_size, -1, self.encoder_dim)  # (batch_size, num_features, encoder_dim)

        # Initialize LSTM state
        hidden, cell_state = self.init_hidden_state(features)  # (batch_size, decoder_dim)
        max_timespan = max((caption_lengths - 1).tolist())

        embeddings = self.embedding(captions)  # (batch_size, max_caption_length, embed_dim)
        
        logits = torch.zeros(batch_size, max_timespan, self.vocab_size).to(const.DEVICE)
        alphas = torch.zeros(batch_size, max_timespan, features.size(1)).to(const.DEVICE)
        pred_idx = torch.tensor([1] * batch_size).to(const.DEVICE)
        for t in range(max_timespan):
            context, alpha = self.attention(features, hidden)
            gate = self.sigmoid(self.f_beta(hidden)) 
            gated_context = gate * context

            # LSTM
            embedding = embeddings[:, t] if is_teacher_forcing else self.embedding(pred_idx)

            lstm_input = torch.cat([embedding, gated_context], dim=1)
            hidden, cell_state = self.lstm_cell(lstm_input, (hidden, cell_state))
            logit = self.fc(hidden)
            
            logits[:, t] = logit
            alphas[:, t] = alpha
            pred_idx = logit.argmax(dim=-1)

        probs = F.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1)
        caption_idxs = pred_idx.tolist()
        
        return logits#, captions, decode_lengths, sort_ind 