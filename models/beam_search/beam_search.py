import torch
import torch.nn.functional as F


class BeamSearch(object):
    def __init__(self, model, beam_size, max_length, eos_idx):
        self.model = model
        self.beam_size = beam_size
        self.max_length = max_length
        self.eos_idx = eos_idx
        self.bos_idx = model.bos_idx

    def search(self, features):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.model.is_graph_encoder:
            batch_size = features.batch_size
        else:
            batch_size = features.shape[0]

        # Initialize the beams
        beams = torch.full(
            (batch_size, self.beam_size, self.max_length),
            self.bos_idx,
            dtype=torch.long,
            device=device,
        )  # [B, beam_size, L]
        beam_log_probs = torch.zeros(
            (batch_size, self.beam_size), device=device
        )  # [B, beam_size]

        beam_candidates = torch.full(
            (batch_size, self.beam_size * self.beam_size, self.max_length),
            0,
            dtype=torch.long,
            device=device,
        )  # [B, beam_size^2, L]
        beam_candidate_log_probs = torch.zeros(
            (batch_size, self.beam_size * self.beam_size), device=device
        )  # [B, beam_size^2]

        features = self.model.encoder(features)
        if self.model.is_graph_encoder:
            features = self.model._prepare_graph_features_for_decoder(features)

        # Handle t=1 to get the initial beam
        seq = torch.full((batch_size, 1), self.bos_idx, dtype=torch.long, device=device)

        output = self.model.decoder(features, seq)
        top_k_probs, top_k_indices = torch.topk(output[:, -1, :], self.beam_size)
        beams[:, :, 1] = top_k_indices
        beam_log_probs += top_k_probs

        # Set beam candidates
        for beam in range(self.beam_size):
            beam_candidates[
                :, beam * self.beam_size : (beam + 1) * self.beam_size, :2
            ] = beams[:, beam, :2].unsqueeze(1)
            beam_candidate_log_probs[
                :, beam * self.beam_size : (beam + 1) * self.beam_size
            ] = beam_log_probs[:, beam].unsqueeze(1)

        # Handle t>1
        for t in range(2, self.max_length):
            for beam in range(self.beam_size):
                # Get the input for the current beam
                seq = beams[:, beam, :t]
                output = self.model.decoder(features, seq)  # [B, t, V]

                # Get the top-k most likely tokens
                top_k_probs, top_k_indices = torch.topk(
                    output[:, -1, :], self.beam_size
                )

                # Set beam candidates
                beam_candidates[
                    :, beam * self.beam_size : (beam + 1) * self.beam_size, t
                ] = top_k_indices
                beam_candidate_log_probs[
                    :, beam * self.beam_size : (beam + 1) * self.beam_size
                ] = beam_log_probs[:, beam].unsqueeze(1) + top_k_probs

            # Sort the beam candidates
            top_probs, top_indices = torch.topk(
                beam_candidate_log_probs, self.beam_size
            )

            # Update the beams
            beams[:, :, :] = beam_candidates[
                torch.arange(batch_size).unsqueeze(1), top_indices, :
            ]
            beam_log_probs = top_probs

            # update the candidates
            for beam in range(self.beam_size):
                beam_candidates[
                    :, beam * self.beam_size : (beam + 1) * self.beam_size, :
                ] = beams[:, beam, :].unsqueeze(1)
                beam_candidate_log_probs[
                    :, beam * self.beam_size : (beam + 1) * self.beam_size
                ] = beam_log_probs[:, beam].unsqueeze(1)

        # Return the beam with the highest log-probability
        top_beam = beam_log_probs.argmax(dim=1)
        b = beams[torch.arange(batch_size), top_beam]
        return b
