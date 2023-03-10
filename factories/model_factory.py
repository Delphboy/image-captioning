from typing import Optional
from constants import Constants as const
import models.basic_captioning_models as cap_models

import torch.nn as nn


def _get_resnet152lstm(vocab_size:int, 
                    embed_size: Optional[int]=256, 
                    hidden_size: Optional[int]=256, 
                    num_lstm_layers: Optional[int]=1) -> nn.Module:
    return cap_models.CaptionWithResnet152AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_lstm_layers)


def _get_resnet101lstm(vocab_size:int, 
                    embed_size: Optional[int]=256, 
                    hidden_size: Optional[int]=256, 
                    num_lstm_layers: Optional[int]=1) -> nn.Module:
    return cap_models.CaptionWithResnet101AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_lstm_layers)


def _get_resnet18lstm(vocab_size:int, 
                    embed_size: Optional[int]=256, 
                    hidden_size: Optional[int]=256, 
                    num_lstm_layers: Optional[int]=1) -> nn.Module:
    return cap_models.CaptionWithResnet18AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_lstm_layers)


def _get_inceptionv3lstm(vocab_size:int, 
                    embed_size: Optional[int]=256, 
                    hidden_size: Optional[int]=256, 
                    num_lstm_layers: Optional[int]=1):
    return cap_models.CaptionWithInceptionV3AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_lstm_layers)

def _get_spatialgcn(vocab_size:int, 
                    embed_size: Optional[int]=256, 
                    hidden_size: Optional[int]=256, 
                    num_lstm_layers: Optional[int]=1):
    return cap_models.CaptionWithSpatialGraph(embed_size=embed_size, 
                                              hidden_size=hidden_size, 
                                              vocab_size=vocab_size, 
                                              num_layers=num_lstm_layers)

################################################################################

MODELS = {
    "resnet152lstm": _get_resnet152lstm,
    "resnet101lstm": _get_resnet101lstm,
    "resnet18lstm": _get_resnet18lstm,
    "inceptionv3lstm": _get_inceptionv3lstm,
    "spatialgcn": _get_spatialgcn
}

def get_model(model_name: str,
                vocab_size:int, 
                embed_size: Optional[int]=256, 
                hidden_size: Optional[int]=256, 
                num_lstm_layers: Optional[int]=1) -> nn.Module:
    model_name = model_name.lower()
    
    if model_name not in MODELS:
        raise Exception(f"The model name {model_name} is not supported by the factory. Supported models are {MODELS.keys()}")

    model = MODELS[model_name](vocab_size, embed_size, hidden_size, num_lstm_layers)
    # model= nn.DataParallel(model)
    model.to(const.DEVICE)

    return model
