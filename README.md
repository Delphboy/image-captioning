# Image Captioning

A collection of image captioning algorithms.

## Dependencies

- Python 3.9.7
- cuda 10.2
- cudnn8.3.0
- Contents of `pip install -r requirements.txt`

### A Note for QMUL Compute Server Users

The above can be loaded using:

```bash
module load python
module load cuda/10.2-cudnn8.3.0
```

⚠️ If you get an error about the `module` command not being available, run `source /etc/profile.d/modules.sh`. If you add that to your `.bashrc` the error should go away permanently.

Specific `torch` install command: `pip install torch==1.12.1+cu102 torchvision==0.13.1+cu102 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu102`

### Datasets

Current datasets supported:
- Flickr8K (`name: "flickr8k"`)
- COCO (`name: "coco"`)
- COCO with Karpathy Split (`name: "coco_karpathy"`)

1. In `datasets/download_scripts` there are a collection of bash scripts for downloading the required datasets. Run the script in the directory you wish to install the dataset to.
2. Update the `dataset` section of the JSON configuration file being run. Note that the `talkfile` will be generated if it doesn't already exist. Name needs to correspond to one of the supported datasets in `data_factory.py`. See example below

```json
"dataset": {
    "name": "flickr8k",
    "root": "/location/to/flickr8k/images",
    "annotations": "/location/to/flickr8k/captions.txt",
    "talk_file": "/location/to/flickr8k/flickrtalk.json"
}
```

## Running the Code

`python3 main.py --file <path to config.json>`
