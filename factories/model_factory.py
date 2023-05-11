from typing import Optional

from constants import Constants as const
from models.base_captioner import BaseCaptioner
import models.basic_captioning_models as basic_models
import models.single_graph_captioners as single_graph_captioners
import models.dual_graph_captioners as dual_graph_captioners


def _get_resnet152lstm(vocab_size:int, 
                    embed_size: Optional[int]=2048, 
                    hidden_size: Optional[int]=1000, 
                    num_lstm_layers: Optional[int]=1)-> BaseCaptioner:
    return basic_models.CaptionWithResnet152AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_lstm_layers)


def _get_resnet101lstm(vocab_size:int, 
                    embed_size: Optional[int]=2048, 
                    hidden_size: Optional[int]=1000, 
                    num_lstm_layers: Optional[int]=1)-> BaseCaptioner:
    return basic_models.CaptionWithResnet101AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_lstm_layers)


def _get_resnet18lstm(vocab_size:int, 
                    embed_size: Optional[int]=2048, 
                    hidden_size: Optional[int]=1000, 
                    num_lstm_layers: Optional[int]=1)-> BaseCaptioner:
    return basic_models.CaptionWithResnet18AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_lstm_layers)


def _get_inceptionv3lstm(vocab_size:int, 
                    embed_size: Optional[int]=2048, 
                    hidden_size: Optional[int]=1000, 
                    num_lstm_layers: Optional[int]=1):
    return basic_models.CaptionWithInceptionV3AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_lstm_layers)


def _get_spatialgat(vocab_size:int, 
                    embed_size: Optional[int]=2048, 
                    hidden_size: Optional[int]=1000, 
                    num_lstm_layers: Optional[int]=1):
    return single_graph_captioners.SemanticGat(embedding_size=embed_size, 
                                              hidden_size=hidden_size, 
                                              vocab_size=vocab_size, 
                                              num_layers=num_lstm_layers)


def _get_semanticgat(vocab_size:int, 
                    embed_size: Optional[int]=2048, 
                    hidden_size: Optional[int]=1000, 
                    num_lstm_layers: Optional[int]=1):
    return single_graph_captioners.SemanticGat(embedding_size=embed_size, 
                                              hidden_size=hidden_size, 
                                              vocab_size=vocab_size, 
                                              num_layers=num_lstm_layers)


def _get_spatialsemanticgat(vocab_size:int, 
                    embed_size: Optional[int]=2048, 
                    hidden_size: Optional[int]=1000, 
                    num_lstm_layers: Optional[int]=1):
    return dual_graph_captioners.SpatialSemanticGat(embedding_size=embed_size, 
                                                    hidden_size=hidden_size, 
                                                    vocab_size=vocab_size, 
                                                    num_layers=num_lstm_layers)



################################################################################

MODELS = {
    "resnet152lstm": _get_resnet152lstm,
    "resnet101lstm": _get_resnet101lstm,
    "resnet18lstm": _get_resnet18lstm,
    "inceptionv3lstm": _get_inceptionv3lstm,
    "spatialgat": _get_spatialgat,
    "semanticgat": _get_semanticgat,
    "spatialsemanticgat": _get_spatialsemanticgat,
}

def get_model(model_name: str,
                vocab_size:int, 
                embed_size: Optional[int]=2048, 
                hidden_size: Optional[int]=1000, 
                num_lstm_layers: Optional[int]=2) -> BaseCaptioner:
    model_name = model_name.lower()
    
    if model_name not in MODELS:
        raise Exception(f"The model name {model_name} is not supported by the factory. Supported models are {MODELS.keys()}")

    model = MODELS[model_name](vocab_size, embed_size, hidden_size, num_lstm_layers)
    
    # TODO: Fix multi-gpu training 
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     torch.distributed.init_process_group(backend="nccl")
    #     model= nn.parallel.DistributedDataParallel(model)       
    model.to(const.DEVICE)

    return model
