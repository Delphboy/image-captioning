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

### Data Set

Currently, the code runs on the Flickr8k data set. The main download link has died, so download with the following commands

```bash
mkdir flickr8k
cd flickr8k

wget -c https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip
unzip Flickr8k_Dataset.zip

mv Flicker8k_Dataset/ images/

rm Flickr8k_Dataset.zip 
rm -rf __MACOSX

wget https://gist.githubusercontent.com/Delphboy/4f0ec8a9fd2c3c12eb2df820963176e7/raw/ff2cb0e84f461f9f8efa298aec40dfb94ab8417b/flickr8k_captions.txt

mv flickr8k_captions.txt captions.txt
```

## Running the Code

`python3 main.py`
