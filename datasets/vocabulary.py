import os
import json
from constants import Constants as const
from utils.data_cleaning import preprocess_captions

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold


    def __len__(self):
        return len(self.itos)


    def build_vocabulary(self, sentence_list=None):
        if os.path.exists(const.TALK_FILE):
            with open(const.TALK_FILE, 'r') as f:
                self.itos = json.load(f)
                self.stoi = {v: int(k) for k, v in self.itos.items()}
            return

        sentence_list = preprocess_captions(sentence_list)

        frequencies = {}
        idx = 4 # idx 0, 1, 2, 3 are already taken (PAD, SOS, EOS, UNK)
        for sentence in sentence_list:
            for word in sentence.split(' '):#self.tokenizer_eng(sentence):
                if word == '': continue
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = int(idx)
                    self.itos[idx] = word
                    idx += 1
        
        # write self.itos to a json file in teh datasets folder
        with open(const.TALK_FILE, 'w') as f:
            json.dump(self.itos, f)


    def numericalize(self, text):
        # tokenized_text = self.tokenizer_eng(text)
        tokenized_text = text.split(' ')

        val= [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
        return val
