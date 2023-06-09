from typing import List

import torch
import torch.nn.functional as F
from models.components.gnns.gat import Gat
from models.components.language.lstm import AttentionLstm
from models.base_captioner import BaseCaptioner
from constants import Constants as const


class SingleGraphCaptioner(BaseCaptioner):
    def __init__(self,
                 embedding_size: int,
                 hidden_size: int,
                 vocab_size: int,
                 num_layers: int) -> None:
        super(SingleGraphCaptioner, self).__init__(embedding_size,
                                                   hidden_size,
                                                   vocab_size,
                                                   num_layers)
        self.encoder = Gat(embedding_size, embedding_size)
        self.decoder = AttentionLstm(embedding_size, hidden_size, vocab_size, num_layers)


    def forward(self, graphs, captions, lengths):   
        features = self.encoder(graphs)
        outputs = self.decoder(features, captions, lengths)
        return outputs


    @torch.no_grad()
    def caption_image(self, input_features, vocab, max_length=20, method='greedy'):
        assert method in ['greedy', 'beam_search']
        if method == 'greedy':
            outputs, _ = self.greedy_caption(input_features, vocab, max_length)
            return outputs
        else:
            outputs, _ = self.beam_search_caption(input_features, vocab, max_length)
            return outputs
    

    @torch.no_grad()
    def greedy_caption(self, 
                      input_features: List, 
                      vocab, 
                      max_length=20) -> List[str]:
        convert = lambda idxs: [vocab.itos[f"{int(idx)}"] for idx in idxs]
   
        features = self.encoder(input_features)
        batch_size = features.size(0)

        hidden, cell_state = self.decoder.init_hidden_state(features)
        logits = torch.zeros(batch_size, max_length, self.vocab_size).to(const.DEVICE)
        alphas = torch.zeros(batch_size, max_length, features.size(1)).to(const.DEVICE)
        
        prev_word = torch.tensor([vocab.stoi["<SOS>"]]).to(const.DEVICE)

        for t in range(max_length):
            context, alpha = self.decoder.attention(features, hidden)
            gate = self.decoder.sigmoid(self.decoder.f_beta(hidden)) 
            gated_context = gate * context

            embedding = self.decoder.embedding(prev_word)
            lstm_input = torch.cat([embedding, gated_context], dim=1)
            hidden, cell_state = self.decoder.lstm_cell(lstm_input, (hidden, cell_state))
            logit = self.decoder.fc(hidden)
            
            logits[:, t] = logit
            alphas[:, t] = alpha

            prev_word = logit.argmax(1)

        probs = F.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1)
        caption_idxs = pred_idx.tolist()[0]
        caption = convert(caption_idxs)
        return caption, probs


class SpatialGat(SingleGraphCaptioner):
    def __init__(self, 
                 embedding_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_layers: int) -> None:
        super(SpatialGat, self).__init__(embedding_size, 
                                         hidden_size, 
                                         vocab_size, 
                                         num_layers)
    
    
    def forward(self, graphs, captions, lengths):   
        return super().forward(graphs[0], captions, lengths) 


    def caption_image(self, graphs, vocab, max_length=20):
        return super().caption_image(graphs[0], vocab, max_length)
    


class SemanticGat(SingleGraphCaptioner):
    def __init__(self, 
                 embedding_size: int, 
                 hidden_size: int, 
                 vocab_size: int, 
                 num_layers: int) -> None:
        super(SemanticGat, self).__init__(embedding_size, 
                                         hidden_size, 
                                         vocab_size, 
                                         num_layers)
    
    
    def forward(self, graphs, captions, lengths):   
        return super().forward(graphs[1], captions, lengths) 


    def caption_image(self, graphs, vocab, max_length=20):
        return super().caption_image(graphs[1], vocab, max_length)
