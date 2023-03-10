import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image

import eval
import train as trainer
from constants import Constants as const
from factories.data_factory import get_coco_data, get_flickr8k_data, get_flickr8k_data_with_spatial_graphs
from factories.model_factory import get_model
from models.basic_captioning_models import (CaptionWithInceptionV3AndLstm,
                                      CaptionWithResnet152AndLstm)
from utils import save_and_load_models


def get_test_image(location, transform=None):
    img = Image.open(location).convert("RGB")
    if transform is not None:
        img = transform(img)
        img = img.unsqueeze(0)
    return img


def caption_array_to_string(array: list[str]) -> str:
    caption = ""

    for i in range(1, len(array)):
        item = array[i]

        if item == "<EOS>": break

        # The captions.txt has a space before fullstops
        if item != '.':
            caption += f"{item} "
        else:
            caption += "."

    return caption


def load_and_evaluate(model_name: str, model_save_name: str, is_graph_based: bool = False):        
    if is_graph_based:
        train_loader, test_loader, dataset = get_flickr8k_data_with_spatial_graphs(
        root_folder=const.FLICKR_ROOT,
        annotation_file=const.FLICKR_ANN,
        transform=const.STANDARD_TRANSFORM,
        graph_dir='/homes/hps01/image-captioning/saved_models/flickr_spatial_graphs.pt', # TODO: Move to constants
        train_ratio=0.8,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=True
    )
    else:
        train_loader, test_loader, dataset = get_flickr8k_data(
            root_folder=const.FLICKR_ROOT,
            annotation_file=const.FLICKR_ANN,
            transform=const.STANDARD_TRANSFORM,
            train_ratio=0.8,
            batch_size=1,
            num_workers=1,
            shuffle=False,
            pin_memory=True
        )

    model = get_model(model_name, len(dataset.vocab))
    model = save_and_load_models.load_model(model=model, 
                                            optimiser=None, 
                                            save_name=model_save_name)

    if is_graph_based:
        eval.evaluate_graph_caption_model(model, test_loader, dataset)
    else:
        eval.evaluate_caption_model(model, test_loader, dataset)



# TODO: This should pull from a configuration file rather than take a model name
def build_and_train_model(model_name: str) -> None:
    print(f"Set device to: {const.DEVICE}\n")
    
    # train_dataloader, val_loader, train_dataset, val_dataset = get_coco_data(
    #     root_folder="/import/visionwebdata/MSCOCO/2017/images/",
    #     annotation_file="/import/visionwebdata/MSCOCO/2017/annotations/",
    #     transform=const.STANDARD_TRANSFORM,
    #     batch_size=2,
    #     num_workers=1,
    #     shuffle=True,
    #     pin_memory=True
    # )

    train_loader, val_loader, dataset = get_flickr8k_data(
        root_folder=const.FLICKR_ROOT,
        annotation_file=const.FLICKR_ANN,
        transform=const.STANDARD_TRANSFORM,
        train_ratio=0.8,
        batch_size=8,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    # Hyperparameters
    vocab_size = len(dataset.vocab)
    learning_rate = 3e-4
    epochs=100

    captioning_model = get_model(model_name, vocab_size)

    # TODO: Move this to the InceptionV3 implementation as a parameter
    # is_cnn_training = False
    # Only finetune the CNN (InceptionV3)
    # for name, param in captioning_model.cnn.inception.named_parameters():
    #     if "fc.weight" in name or "fc.bias" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = is_cnn_training

    # captioning_model.train()

    # cross_entropy = nn.CrossEntropyLoss(ignore_index=dataset.word_to_ix["<PAD>"])
    cross_entropy = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    adam_optimiser = optim.Adam(captioning_model.parameters(), lr=learning_rate)

    trained, epoch, loss = trainer.train(model=captioning_model, 
                                        optimiser=adam_optimiser, 
                                        loss_function=cross_entropy, 
                                        data_loader=train_loader, 
                                        epoch_count=epochs)

    save_and_load_models.save_model_checkpoint(trained, adam_optimiser, epoch, loss, save_name=f'{epochs}_epochs_{model_name}')
    print('Model fully trained!')
    return captioning_model


# TODO: This should pull from a configuration file rather than take a model name
def build_and_train_graph_model(model_name: str) -> None:
    print(f"Set device to: {const.DEVICE}\n")

    train_loader, val_loader, dataset = get_flickr8k_data_with_spatial_graphs(
        root_folder=const.FLICKR_ROOT,
        annotation_file=const.FLICKR_ANN,
        transform=const.STANDARD_TRANSFORM,
        graph_dir='/homes/hps01/image-captioning/saved_models/flickr_spatial_graphs.pt',
        train_ratio=0.8,
        batch_size=8,
        num_workers=4,
        shuffle=False,
        pin_memory=True
    )

    # Hyperparameters
    vocab_size = len(dataset.vocab)
    learning_rate = 3e-4
    epochs=100

    captioning_model = get_model(model_name, vocab_size)

    cross_entropy = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    adam_optimiser = optim.Adam(captioning_model.parameters(), lr=learning_rate)

    trained, epoch, loss = trainer.train_graph_model(model=captioning_model, 
                                                     optimiser=adam_optimiser, 
                                                     loss_function=cross_entropy, 
                                                     data_loader=train_loader, 
                                                     epoch_count=epochs)

    save_and_load_models.save_model_checkpoint(trained, adam_optimiser, epoch, loss, save_name=f'{epochs}_epochs_{model_name}')
    print('Model fully trained!')
    return captioning_model


if __name__ == "__main__":
    model_name = "spatialgcn"
    # trained_model = build_and_train_model(model_name)
    load_and_evaluate(model_name, f'100_epochs_{model_name}', is_graph_based=model_name=="spatialgcn")



