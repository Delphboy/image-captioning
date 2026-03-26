import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderRNN(nn.Module):
    def __init__(self, args, vocab):
        super(DecoderRNN, self).__init__()
        self.vocab_size = len(vocab)
        self.token_dim = args.token_dim
        self.embedding = nn.Embedding(self.vocab_size, self.token_dim)

        self.rnn = nn.ModuleList(
            [
                nn.LSTMCell(self.token_dim, self.token_dim)
                for _ in range(args.dec_num_layers)
            ]
        )
        self.fc = nn.Linear(self.token_dim, self.vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.norm = nn.LayerNorm(self.token_dim)
        self.init_weights()

    def init_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.Embedding):
                    nn.init.normal_(layer.weight, 0, 0.02)
                elif isinstance(layer, nn.LSTMCell):
                    nn.init.xavier_uniform_(layer.weight_ih)
                    nn.init.orthogonal_(layer.weight_hh)
                    nn.init.constant_(layer.bias_ih, 0)
                    nn.init.constant_(layer.bias_hh, 0)

    @property
    def d_model(self):
        return self.token_dim

    def forward(self, features, caption):
        b = features.shape[0]
        f = features.shape[1]
        emb = self.embedding(caption)

        h = torch.zeros(b, self.token_dim).to(features.device)
        c = torch.zeros(b, self.token_dim).to(features.device)

        input = torch.cat([features, emb], dim=1)
        outputs = torch.zeros(b, f + caption.shape[1], self.vocab_size).to(
            features.device
        )
        for t in range(0, f + caption.shape[1]):
            for rnn in self.rnn:
                h, c = rnn(input[:, t, :], (h, c))

            h = self.norm(h)
            outputs[:, t, :] = self.fc(h)

        outputs = F.log_softmax(outputs[:, f:, :], dim=-1)
        return outputs
