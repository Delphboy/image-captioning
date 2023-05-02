import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import eval
import train as trainer
from constants import Constants as const
from factories.data_factory import get_data
from factories.model_factory import get_model
from utils.helper_functions import parse_config_file
from utils.save_and_load_models import *


def load_and_evaluate(model_save_name: str) -> None:        
    _, _, test_loader, train_dataset, _, test_dataset, _ = get_data(const.DATASET)

    vocab_size = len(train_dataset.vocab)
    model = get_model(model_name=const.MODEL, 
                    vocab_size=vocab_size,
                    embed_size=2048,
                    hidden_size=1000,
                    num_lstm_layers=2)
    
    model, _, _, _ = load_model(model=model, 
                                optimiser=None, 
                                save_name=model_save_name)
    model.eval()

    eval.evaluate_caption_model(model, test_dataset)



def build_and_train_model() -> None:
    print(f"Set device to: {const.DEVICE}")

    # Get data
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, pad_index = get_data(const.DATASET)
    
    # FIX: This is a hack to get the validation set to be the test set
    train_loader = val_loader
    val_loader = test_loader

    # Build Model
    vocab_size = len(train_dataset.vocab)
    captioning_model = get_model(model_name=const.MODEL, 
                                 vocab_size=vocab_size,
                                 embed_size=2048,
                                 hidden_size=1000,
                                 num_lstm_layers=2)

    cross_entropy = nn.CrossEntropyLoss(ignore_index=pad_index)
    adam_optimiser = optim.Adam(captioning_model.parameters(), 
                                lr=const.LEARNING_RATE, 
                                weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(adam_optimiser, 
                                                     patience=3,
                                                     factor=0.1,
                                                     verbose=True)

    trained, epoch, loss = trainer.train(model=captioning_model, 
                                        optimiser=adam_optimiser, 
                                        scheduler=scheduler,
                                        loss_function=cross_entropy, 
                                        train_data_loader=train_loader, 
                                        val_data_loader=val_loader,
                                        epoch_count=const.EPOCHS)

    save_model_checkpoint(trained, 
                          adam_optimiser, 
                          epoch, 
                          loss, 
                          save_name=const.MODEL_SAVE_NAME)
    print('Model fully trained!')


if __name__ == "__main__":
    torch.manual_seed(1337)
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
    parse_config_file(args.file[0])

    if const.REGIME.__contains__("train"):
        build_and_train_model()
    
    if const.REGIME.__contains__("test"):
        load_and_evaluate(const.MODEL_SAVE_NAME)



