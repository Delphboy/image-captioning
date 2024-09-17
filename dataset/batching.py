import torch
from torch_geometric.data import Batch


class Batcher:
    def __init__(self, vocab, feature_limit):
        self.vocab = vocab
        self.feature_limit = feature_limit

    def __call__(self, batch):
        features, seq, captions = zip(*batch)
        X_padded = []
        for feat in features:
            if feat.shape[0] < self.feature_limit:
                padding = torch.zeros(
                    self.feature_limit - feat.shape[0],
                    feat.shape[1],
                    dtype=torch.float,
                ).to(feat.device)
                x = torch.cat((feat, padding), dim=0)
            else:
                x = feat
            X_padded.append(x[: self.feature_limit, :])

        X = torch.stack(X_padded, dim=0)

        max_len = max(len(s) for s in seq)
        seq_padded = []
        for s in seq:
            padded = s + [self.vocab.stoi("<pad>")] * (max_len - len(s))
            seq_padded.append(padded)

        seq_padded = torch.LongTensor(seq_padded)

        return X, seq_padded, captions


class GraphBatcher:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, batch):
        graphs, seq, captions = zip(*batch)

        data_batches = Batch.from_data_list(graphs)

        max_len = max(len(s) for s in seq)
        seq_padded = []
        for s in seq:
            padded = s + [self.vocab.stoi("<pad>")] * (max_len - len(s))
            seq_padded.append(padded)

        seq_padded = torch.LongTensor(seq_padded)

        return data_batches, seq_padded, captions
