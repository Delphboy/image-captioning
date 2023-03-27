import argparse
import os

import torch.nn as nn
import torch.optim as optim

import eval
import train as trainer
from constants import Constants as const
from factories.data_factory import get_data
from factories.model_factory import get_model
from utils import save_and_load_models
from utils.helper_functions import parse_file


def load_and_evaluate(model_name: str, model_save_name: str):        
    _, val_loader, _, val_dataset, _ = get_data(const.DATASET)

    model = get_model(model_name, 
                      len(val_dataset.dataset.vocab))
    
    model = save_and_load_models.load_model(model=model, 
                                            optimiser=None, 
                                            save_name=model_save_name)
    model.eval()

    if const.IS_GRAPH_MODEL:
        eval.evaluate_graph_caption_model(model, val_loader, val_dataset)
    else:
        eval.evaluate_caption_model(model, val_loader, val_dataset)



def build_and_train_model() -> None:
    print(f"Set device to: {const.DEVICE}\n")

    dataset = None
    train_loader, _, dataset, _, pad_index = get_data(const.DATASET)

    # Build Model
    vocab_size = len(dataset.dataset.vocab)
    captioning_model = get_model(const.MODEL, vocab_size)

    cross_entropy = nn.CrossEntropyLoss(ignore_index=pad_index)
    adam_optimiser = optim.Adam(captioning_model.parameters(), lr=const.LEARNING_RATE)

    trained, epoch, loss = trainer.train(model=captioning_model, 
                                        optimiser=adam_optimiser, 
                                        loss_function=cross_entropy, 
                                        data_loader=train_loader, 
                                        epoch_count=const.EPOCHS)

    save_and_load_models.save_model_checkpoint(trained, 
                                               adam_optimiser, 
                                               epoch, 
                                               loss, 
                                               save_name=const.MODEL_SAVE_NAME)
    print('Model fully trained!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and evaluate a captioning model based on a configuation file"
    )

    parser.add_argument("-f", "--file",
        help="Location of config file",
        action="store",
        nargs=1,
        metavar=("config_location")
    )

    args = parser.parse_args()
    parse_file(args.file[0])

    if const.REGIME.__contains__("train"):
        build_and_train_model()
    
    if const.REGIME.__contains__("test"):
        load_and_evaluate(const.MODEL, const.MODEL_SAVE_NAME)



