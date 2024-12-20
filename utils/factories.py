import torch
from torch.utils.data import DataLoader

from dataset.batching import Batcher, GraphBatcher

from models.captioning_model import CaptioningModel
from models.encoders.gnn import Gnn
from models.encoders.simple import PoolEncoder, NoneEncoder
from models.decoders.lstm import DecoderRNN
from models.decoders.dual_lstm import DualLstm


def get_optim_and_scheduler(args, model):
    optim = None
    scheduler = None
    if args.optim == "Adam":
        optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optim == "AdamW":
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    elif args.optim == "SGD":
        optim = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optim == "RMSProp":
        optim = torch.optim.RMSprop(
            model.parameters(), lr=args.learning_rate, momentum=0.9
        )
    else:
        raise ValueError(f"Optimiser type {args.optim} is not supported")

    if args.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda _: 1)
    elif args.scheduler == "linear":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda epoch: 1 - (epoch / args.epochs)
        )
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optim, 5, 0.8)
    elif args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode="min", factor=0.8, patience=3
        )
    else:
        raise ValueError(f"Scheduler type {args.scheduler} is not supported")

    return optim, scheduler


def get_model(args, vocab):
    encoder, decoder = None, None
    if args.enc_model_type in ["gcn", "gin", "gat"]:
        assert args.input_mode != "butd", "Input mode must be a graph type"
        encoder = Gnn(
            in_features=2048,  # Using ResNet features
            hidden_features=1024,
            out_features=args.token_dim,
            convolution=args.enc_model_type,
            layers=args.enc_num_layers,
            dropout=args.dropout,
        )
    elif args.enc_model_type == "pool":
        assert args.input_mode == "butd", "Pool encoder does not work with graph inputs"
        encoder = PoolEncoder()
    elif args.enc_model_type == "none":
        assert args.input_mode == "butd", "encoder=None does not work with graph inputs"
        encoder = NoneEncoder()
    else:
        raise ValueError(
            f"Encoder model type {args.enc_model_type} is not yet supported"
        )

    if args.dec_lang_model == "lstm":
        decoder = DecoderRNN(args, vocab)
    elif args.dec_lang_model == "dual_lstm":
        decoder = DualLstm(args, vocab)
    else:
        raise ValueError(
            f"Encoder model type {args.dec_lang_model} is not yet supported"
        )
    return CaptioningModel(encoder, decoder, vocab.stoi["<bos>"], args.feature_limit)


def get_datasets(args):
    train_data, val_data, test_data = None, None, None
    from dataset.captioning_dataset import CocoImageGraphDataset

    train_data = CocoImageGraphDataset(
        args.captions_file,
        args.butd_root,
        args.sgae_root,
        args.vsua_root,
        True,
        freq_threshold=5,
        split="train",
        graph_mode=args.input_mode,
    )

    val_data = CocoImageGraphDataset(
        args.captions_file,
        args.butd_root,
        args.sgae_root,
        args.vsua_root,
        True,
        freq_threshold=5,
        split="val",
        graph_mode=args.input_mode,
    )

    test_data = CocoImageGraphDataset(
        args.captions_file,
        args.butd_root,
        args.sgae_root,
        args.vsua_root,
        True,
        freq_threshold=5,
        split="test",
        graph_mode=args.input_mode,
    )

    return train_data, val_data, test_data


def get_dataloader(
    dataset,
    args,
    shuffle: bool = True,
) -> DataLoader:
    if dataset.graph_mode == "butd":
        batcher = Batcher(dataset.vocab, args.feature_limit)
    else:
        batcher = GraphBatcher(dataset.vocab)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=False,
        shuffle=shuffle,
        collate_fn=batcher,
        drop_last=True,
    )
