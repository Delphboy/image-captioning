import json
import os
from collections import Counter
from typing import List


class Vocab:
    def __init__(self, freq_threshold):
        self._itos = {1: "<pad>", 2: "<bos>", 3: "<eos>", 0: "<unk>"}
        self._stoi = {"<pad>": 1, "<bos>": 2, "<eos>": 3, "<unk>": 0}
        self.freq_threshold = freq_threshold
        self.talk_file_location = "dataset/coco_talk.json"

    def __len__(self):
        return len(self._itos)

    def get_special_token(self, token: str):
        specials = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
        }
        return specials.get(token, None)

    def build_vocabulary(self, sentence_list):
        if os.path.exists(self.talk_file_location):
            with open(self.talk_file_location, "r") as f:
                self._itos = json.load(f)
                self._stoi = {v: int(k) for k, v in self._itos.items()}
            return
        assert (
            sentence_list is not None
        ), "sentence_list must be provided to generate vocab"
        frequencies = Counter(
            word for sentence in sentence_list for word in sentence.split(" ")
        )
        idx = 4  # idx 1, 2, 3, 0 are already taken (PAD, SOS, EOS, UNK)
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self._stoi[word] = int(idx)
                self._itos[idx] = word
                idx += 1

        # write self.itos to a json file
        with open(self.talk_file_location, "w+") as f:
            json.dump(self._itos, f)

    def stoi(self, token):
        return self._stoi.get(token, self._stoi[self.get_special_token("unk_token")])

    def itos(self, id):
        return self._itos.get(id, self._stoi[self.get_special_token("unk_token")])

    def numericalize(self, text):
        tokenized_text = text.split(" ")
        return [
            self._stoi[token] if token in self._stoi else self._stoi["<unk>"]
            for token in tokenized_text
        ]
