from typing import List

import torch
import torch.nn.functional as F
from datasets.vocabulary import Vocabulary
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
    def caption_image(self, input_features, vocab, max_length=20, method='beam_search'):
        assert method in ['greedy', 'beam_search']
        if method == 'greedy':
            outputs, _ = self.greedy_caption(input_features, vocab, max_length)
            return outputs
        
        if method == 'beam_search':
            outputs, _ = self.beam_search_caption(input_features, vocab, max_length, beam_size=5)
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
    

    @torch.no_grad()
    def beam_search_caption(self, 
                            input_features: List, 
                            vocab: Vocabulary, 
                            max_length=20,
                            beam_size=3) -> List[str]:
        convert = lambda idxs: [vocab.itos[f"{int(idx)}"] for idx in idxs]
   
        k = beam_size
        vocab_size = len(vocab)

        # Move to GPU if available
        features = self.encoder(input_features)
        features = features.to(const.DEVICE)
        features = features.expand(k, features.size(1), features.size(2))

        # Tensor to store top k previous words at each step; now they're just <start>
        k_prev_words = torch.tensor([[vocab.stoi['<SOS>']]] * k, dtype=torch.long).to(const.DEVICE)  # (k, 1)

        # Tensor to store top k sequences; now they're just <start>
        seqs = k_prev_words  # (k, 1)

        # Tensor to store top k sequences' scores; now they're just 0
        top_k_scores = torch.zeros(k, 1).to(const.DEVICE)  # (k, 1)

        # Lists to store completed sequences and scores
        complete_seqs = list()
        complete_seqs_scores = list()

        # Start decoding
        step = 1
        hidden, cell_state = self.decoder.init_hidden_state(features)  # (batch_size, self.decoder_dim)

        # s is a number less than or equal to k, because sequences are removed from this process once they hit <end>
        while True:
            context, alpha = self.decoder.attention(features, hidden)
            gate = self.decoder.sigmoid(self.decoder.f_beta(hidden)) 
            gated_context = gate * context

            embeddings = self.decoder.embedding(k_prev_words).squeeze(1)  # (s, embed_dim)

            lstm_input = torch.cat([embeddings, gated_context], dim=1)
            hidden, cell_state = self.decoder.lstm_cell(lstm_input, (hidden, cell_state))
            logit = self.decoder.fc(hidden)            

            scores = F.log_softmax(logit, dim=1)

            # Add
            scores = top_k_scores.expand_as(scores) + scores  # (s, vocab_size)

            # For the first step, all k points will have the same scores (since same k previous words, h, c)
            if step == 1:
                top_k_scores, top_k_words = scores[0].topk(k, 0, True, True)  # (s)
            else:
                # Unroll and find top scores, and their unrolled indices
                top_k_scores, top_k_words = scores.view(-1).topk(k, 0, True, True)  # (s)

            # Convert unrolled indices to actual indices of scores
            prev_word_inds = top_k_words / vocab_size  # (s)
            prev_word_inds = prev_word_inds.long()
            next_word_inds = top_k_words % vocab_size  # (s)

            # Add new words to sequences
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)  # (s, step+1)decoder.attention

            # Which sequences are incomplete (didn't reach <end>)?
            incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                               next_word != vocab.stoi['<EOS>']]
            complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

            # Set aside complete sequences
            if len(complete_inds) > 0:
                complete_seqs.extend(seqs[complete_inds].tolist())
                complete_seqs_scores.extend(top_k_scores[complete_inds])
            k -= len(complete_inds)  # reduce beam length accordingly

            # Proceed with incomplete sequences
            if k == 0:
                break
            seqs = seqs[incomplete_inds]
            hidden = hidden[prev_word_inds[incomplete_inds]]
            cell_state = cell_state[prev_word_inds[incomplete_inds]]
            features = features[prev_word_inds[incomplete_inds]]
            top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)
            k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)

            # Break if things have been going on too long
            if step > 50:
                break
            step += 1

        if len(complete_seqs_scores) > 0:
            i = complete_seqs_scores.index(max(complete_seqs_scores))
            seq = complete_seqs[i]
            prob = complete_seqs_scores[i]
        else:
            seq = seqs[0].tolist()
            prob = top_k_scores[0].item()

            

        cap = convert(seq)
        return cap, prob

    


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
