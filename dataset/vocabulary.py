import json
import os


class Vocab:
    def __init__(self, dataset_file, freq_threshold, dataset_name="coco"):
        self.dataset_file = dataset_file
        self.itos = {1: "<pad>", 2: "<bos>", 3: "<eos>", 0: "<unk>"}
        self.stoi = {"<pad>": 1, "<bos>": 2, "<eos>": 3, "<unk>": 0}
        self.freq_threshold = freq_threshold
        self.talk_file_location = f"data/{dataset_name}_talk.json"

        if os.path.exists(self.talk_file_location):
            self.load_vocabulary()
        else:
            self.build_vocabulary()
            self.load_vocabulary()

    def __len__(self):
        return len(self.itos)

    def load_vocabulary(self):
        with open(self.talk_file_location, "r") as f:
            self.itos = json.load(f)
            self.stoi = {v: int(k) for k, v in self.itos.items()}
        print(f"Loaded dictionary with {len(self.itos.items())} words")
        return

    def build_vocabulary(self):
        with open(self.dataset_file, "r") as f:
            karpathy_split = json.load(f)

        captions = []
        for image_data in karpathy_split["images"]:
            caps = [
                " ".join(sentence["tokens"]) for sentence in image_data["sentences"]
            ]
            captions.extend(caps)

        print(len(captions))

        caption_dictionary = {}
        for caption in captions:
            for word in caption.split(" "):
                caption_dictionary[word] = caption_dictionary.get(word, 0) + 1

        limited_caption_dictionary = {}
        for k, v in caption_dictionary.items():
            if v >= self.freq_threshold:
                limited_caption_dictionary[k] = v

        for i, k in enumerate(limited_caption_dictionary.keys()):
            self.stoi[k] = i + 4

        self.itos = {v: k for k, v in self.stoi.items()}

        # write self.itos to a json file
        os.mkdir("data") if not os.path.exists("data") else None
        with open(self.talk_file_location, "w+") as f:
            json.dump(self.itos, f)

    def numericalize(self, text):
        # text is a string
        # we want to return a list of integers
        # providing a numericalized version of the text
        tokenized_text = text.split(" ")
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<unk>"]
            for token in tokenized_text
        ]
