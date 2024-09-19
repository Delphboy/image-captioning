import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, token_dim):
        super(Attention, self).__init__()
        self.v_att = nn.Linear(2048, token_dim)
        self.h_att = nn.Linear(token_dim, token_dim)
        self.a_att = nn.Linear(token_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, visual_features, attn_out):
        """
        Compute the full attention between the visual features and the hidden state LSTM
        param: visual_features: Tensor[B, X, 2048]
        param: attn_out: Tensor[B, token_dim]
        return: Tensor[B, X, 1]
        """
        a_v = self.v_att(visual_features)
        a_h = self.h_att(attn_out).unsqueeze(1)

        a = self.tanh(a_v + a_h)
        a = self.a_att(a)
        a_weights = F.softmax(a, dim=-1)
        v_hat = torch.sum(a_weights * visual_features, dim=1)
        return v_hat


class DualLstm(nn.Module):
    def __init__(self, args, vocab):
        super(DualLstm, self).__init__()
        self.vocab_size = len(vocab)
        self.token_dim = args.token_dim
        self.batch_size = args.batch_size
        self.embedding = nn.Embedding(self.vocab_size, self.token_dim)
        self.max_len = args.max_len

        self.lstm_attn = nn.LSTMCell(self.token_dim * 3, self.token_dim)
        self.lstm_lang = nn.LSTMCell(self.token_dim * 2, self.token_dim)
        self.full_attn = Attention(self.token_dim)

        self.fc = nn.Linear(self.token_dim, self.vocab_size, bias=True)
        self.norm = nn.LayerNorm(self.token_dim)

    @property
    def d_model(self):
        return self.token_dim

    def init_hidden_state(self, device="cpu"):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.
        :return: hidden state, cell state
        """
        h = torch.zeros(self.batch_size, self.token_dim).to(device)
        c = torch.zeros(self.batch_size, self.token_dim).to(device)
        return h, c

    def forward(self, features, caption):
        """
        Dual LSTM with Attention, following the Bottom Up Top Down model
        https://arxiv.org/abs/1707.07998
        :param features: Tensor[B, X, 2048]
        :param caption: Tensor[B, X]
        """
        # Initialise hidden states of LSTMs
        h_attn, c_attn = self.init_hidden_state(features.device)
        h_lang, c_lang = self.init_hidden_state(features.device)

        embeddings = self.embedding(caption)  # [B, X, token_dim]
        v_bar = torch.mean(features, dim=1)

        predictions = []
        for t in range(caption.shape[1]):
            x = torch.cat([h_lang, v_bar, embeddings[:, t, :]], dim=1)
            h_attn, c_attn = self.lstm_attn(x, (h_attn, c_attn))

            v_hat = self.full_attn(features, h_attn)
            x = torch.cat([h_attn, v_hat], dim=1)
            h_lang, c_lang = self.lstm_lang(x, (h_lang, c_lang))

            pred_t = self.fc(h_lang)  # .unsqueeze(1)
            predictions.append(pred_t)
        predictions = F.log_softmax(
            torch.stack(predictions, dim=1), dim=2
        )  # .squeeze(2)
        return predictions
