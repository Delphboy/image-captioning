import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image

from constants import Constants as const
from datasets.data_factory import get_flickr8k
from models.model import BasicCaptioner
import train as trainer



def get_test_image(location, transform=None):
    img = Image.open(location).convert("RGB")
    if transform is not None:
        img = transform(img)
        img = img.unsqueeze(0)
    return img



def build_and_train_model():
    # TODO: This should pull from a configuration file
    print(f"Set device to: {const.DEVICE}\n")
    
    transform = transforms.Compose(
        [
            # transforms.PILToTensor(),
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_dataloader, dataset = get_flickr8k(
        root_folder="/homes/hps01/flickr8k/Flicker8k_Dataset",
        annotation_file="/homes/hps01/flickr8k/captions.txt",
        transform=transform,
        batch_size=32,
        num_workers=2,
        shuffle=True,
        pin_memory=True
    )


    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    num_layers = 5
    vocab_size = len(dataset.vocab)
    learning_rate = 3e-4
    epochs=5

    basic_caption_model = BasicCaptioner(embed_size=embed_size, 
                    hidden_size=hidden_size,
                    vocab_size=vocab_size, 
                    num_layers=num_layers).to(device=const.DEVICE)

    cross_entropy = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    adam_optimiser = optim.Adam(basic_caption_model.parameters(), lr=learning_rate)

    trainer.train(model=basic_caption_model, 
                  optimiser=adam_optimiser, 
                  loss_function=cross_entropy, 
                  data_loader=train_dataloader, 
                  epoch_count=epochs)

    

    
if __name__ == "__main__":
    trained_model = build_and_train_model()

    # img_loc = '/homes/hps01/flickr8k/Flicker8k_Dataset/3099694681_19a72c8bdc.jpg'
    # test_image = get_test_image(img_loc, transform).to(const.DEVICE)
    

    # trained_model.eval()
    # caption = trained_model.caption_image(test_image, dataset.vocab)

    # print(caption)
    