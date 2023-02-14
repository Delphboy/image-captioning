import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

from metrics.caption_metrics import bleu_score
from constants import Constants as const
from factories.data_factory import get_coco_data, get_flickr8k_data
from factories.model_factory import get_model, MODELS
from models.captioning_models import CaptionWithInceptionV3AndLstm, CaptionWithResnet152AndLstm, CaptionWithSpatialGraph

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


def load_and_evaluate(model_name: str, model_save_name: str):
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
        num_workers=16,
        shuffle=False,
        pin_memory=True
    )

    model = get_model(model_name, len(dataset.vocab))
    model = utils.load_model(model=model, 
                            optimiser=None,
                            save_name=model_save_name)

    eval.evaluate_caption_model(model, test_loader, dataset)


# TODO: This should pull from a configuration file rather than take a model name
def build_and_train_model(model_name: str) -> None:
    print(f"Set device to: {const.DEVICE}\n")
    
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # train_dataloader, val_loader, train_dataset, val_dataset = get_coco_data(
    #     root_folder="/import/visionwebdata/MSCOCO/2017/images/",
    #     annotation_file="/import/visionwebdata/MSCOCO/2017/annotations/",
    #     transform=transform,
    #     batch_size=2,
    #     num_workers=1,
    #     shuffle=True,
    #     pin_memory=True
    # )

    train_loader, val_loader, dataset = get_flickr8k_data(
        root_folder="/homes/hps01/flickr8k/images",
        annotation_file="/homes/hps01/flickr8k/captions.txt",
        transform=transform,
        train_ratio=0.8,
        batch_size=4,
        num_workers=2,
        shuffle=True,
        pin_memory=True
    )

    # Hyperparameters
    vocab_size = len(dataset.vocab)
    learning_rate = 3e-4
    epochs=1

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

    utils.save_model_checkpoint(trained, adam_optimiser, epoch, loss, save_name=f'{epochs}_epochs_{model_name}')
    print('Model fully trained!')
    return captioning_model



if __name__ == "__main__":
    model_name = "spatialgcn"#"inceptionv3lstm"
    trained_model = build_and_train_model(model_name)
    # load_and_evaluate(model_name, f'1_epochs_{model_name}')


