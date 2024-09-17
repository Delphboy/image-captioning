import torch
import torch.nn as nn
import torch.nn.functional as F

from models.beam_search.beam_search import BeamSearch
from models.encoders import gnn
from torch_geometric.data import Batch


class CaptioningModel(nn.Module):
    def __init__(self, encoder, decoder, bos_idx, feature_limit):
        super(CaptioningModel, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.bos_idx = bos_idx
        self.feature_limit = feature_limit
        self.d_model = decoder.d_model
        self.is_graph_encoder = isinstance(encoder, gnn.Gnn)

    def _prepare_graph_features_for_decoder(self, graphs):
        graphs = Batch.to_data_list(graphs)

        X_padded = []
        for g in graphs:
            if g.x.shape[0] < self.feature_limit:
                padding = torch.zeros(
                    self.feature_limit - g.x.shape[0],
                    g.x.shape[1],
                    dtype=torch.float,
                ).to(g.x.device)
                x = torch.cat((g.x, padding), dim=0)
            else:
                x = g.x
            X_padded.append(x[: self.feature_limit, :])

        X = torch.stack(X_padded, dim=0)
        return X

    def forward(self, input_features, targets):
        x = self.encoder(input_features)
        x = self._prepare_graph_features_for_decoder(x) if self.is_graph_encoder else x
        out = self.decoder(x, targets)
        return out

    def step(
        self,
        t,
        prev_output,
        input_features,
        seq,
        mode="teacher_forcing",
        **kwargs,
    ):
        it = None
        if mode == "teacher_forcing":
            raise NotImplementedError
        elif mode == "feedback":
            if t == 0:
                input_features = self.encoder(input_features)

                if self.is_graph_encoder:
                    input_features = self._prepare_graph_features_for_decoder(
                        input_features
                    )

                self.enc_output, self.mask_enc = self.encoder(input_features)
                if isinstance(input_features, torch.Tensor):
                    it = input_features.data.new_full(
                        (input_features.shape[0], 1), self.bos_idx
                    ).long()
                else:
                    it = (
                        input_features[0]
                        .data.new_full((input_features[0].shape[0], 1), self.bos_idx)
                        .long()
                    )
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)

    def beam_search(
        self,
        visual,
        max_len: int,
        eos_idx: int,
        beam_size: int,
    ):
        bs = BeamSearch(self, beam_size, max_len, eos_idx)
        return bs.search(visual)
