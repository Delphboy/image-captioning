import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

from metrics.caption_metrics import bleu_score
from constants import Constants as const
from datasets.data_factory import get_flickr8k_data
from models.model import CaptionWithInceptionV3AndLstmDropout, CaptionWithResnet152AndLstm

import utils
import eval
import train as trainer



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


def load_and_evaluate(model_name: str):
    embed_size = 256 
    hidden_size = 256
    num_layers = 1
        
    transform = transforms.Compose(
            [
                # transforms.PILToTensor(),
                transforms.Resize((356, 356)),
                transforms.RandomCrop((299, 299)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    train_loader, test_loader, dataset = get_flickr8k_data(
        root_folder="/homes/hps01/flickr8k/images",
        annotation_file="/homes/hps01/flickr8k/captions.txt",
        transform=transform,
        train_ratio=0.8,
        batch_size=32,
        num_workers=2,
        shuffle=False,
        pin_memory=True
    )

    basic_caption_model = CaptionWithResnet152AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=len(dataset.vocab), 
                    num_layers=num_layers).to(device=const.DEVICE)

    basic_caption_model = utils.load_model(model=basic_caption_model, 
                                           optimiser=None,
                                           save_name=model_name)

    eval.evaluate_caption_model(basic_caption_model, test_loader, dataset)


def build_and_train_model() -> None:
    # TODO: This should pull from a configuration file
    print(f"Set device to: {const.DEVICE}\n")
    
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataloader, test_loader, dataset = get_flickr8k_data(
        root_folder="/homes/hps01/flickr8k/images",
        annotation_file="/homes/hps01/flickr8k/captions.txt",
        transform=transform,
        train_ratio=0.8,
        batch_size=64,
        num_workers=32,
        shuffle=True,
        pin_memory=True
    )

    # Hyperparameters
    embed_size = 256 
    hidden_size = 256
    num_layers = 1
    vocab_size = len(dataset.vocab)
    learning_rate = 3e-4
    epochs=5

    is_cnn_training = False

    captioning_model = CaptionWithResnet152AndLstm(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_layers).to(device=const.DEVICE)

    # Only finetune the CNN (InceptionV3)
    # for name, param in captioning_model.cnn.inception.named_parameters():
    #     if "fc.weight" in name or "fc.bias" in name:
    #         param.requires_grad = True
    #     else:
    #         param.requires_grad = is_cnn_training

    captioning_model.train()

    cross_entropy = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    adam_optimiser = optim.Adam(captioning_model.parameters(), lr=learning_rate)

    trained, epoch, loss = trainer.train(model=captioning_model, 
                                        optimiser=adam_optimiser, 
                                        loss_function=cross_entropy, 
                                        data_loader=train_dataloader, 
                                        epoch_count=epochs)

    utils.save_model_checkpoint(trained, adam_optimiser, epoch, loss, save_name='5_with_new_model')
    print('Model fully trained!')
    return captioning_model


if __name__ == "__main__":
    trained_model = build_and_train_model()
    # load_and_evaluate('5_with_new_model')
