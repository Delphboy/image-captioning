import torch
from torch import nn

def lstm_test():
    rnn = nn.LSTM(10, 20, 2)
    input = torch.randn(5, 3, 10)
    
    h0 = torch.randn(2, 3, 20)
    c0 = torch.randn(2, 3, 20)
    
    output, (hn, cn) = rnn(input, (h0, c0))

    print(f"Input is shape: {input.shape}")
    print(f"Output is shape: {output.shape}")


def embedding_test():
    vocab_size = 5 # How many words I have
    embedding_dimension = 4

    sentence_1 = torch.tensor([1, 2, 3, 4]) # A completely fake sentence
    sentence_2 = torch.tensor([0, 4, 3, 2]) # Another completely fake sentence

    sentences = torch.stack((sentence_1, sentence_2))

    # nn.Embedding(How many embeddings I want; Embedding Size)
    emb = nn.Embedding(vocab_size, embedding_dimension)

    embedded = emb(sentences)
    
    print(sentences)
    sm = nn.Softmax(dim=-1)
    print(embedded)
    print()
    print(sm(embedded))

embedding_test()
