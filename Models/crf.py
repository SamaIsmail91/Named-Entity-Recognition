"""
crf.py
======
A from-scratch, batched, numerically-stable linear-chain Conditional Random
Field layer implemented in pure PyTorch (no external CRF library).

This is placed on top of the BiLSTM emission scores to model *label
transition* dependencies (e.g. "I-PER can only follow B-PER or I-PER, never
follow O directly"). This is precisely what fixes the boundary errors that
a plain BiLSTM softmax makes -- softmax decides each token's tag
independently, so it can output an illegal sequence like O -> I-PER; the CRF
learns a transition matrix that makes such sequences (near-)impossible, and
decodes the globally-optimal tag sequence with the Viterbi algorithm instead
of greedy per-token argmax.
"""

import torch
import torch.nn as nn

IMPOSSIBLE = -10000.0


class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first

        # Learnable transition matrix: transitions[i, j] = score of moving
        # from tag i -> tag j.
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags) * 0.01)
        # Score of a sequence *starting* with tag i / *ending* with tag i.
        self.start_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)
        self.end_transitions = nn.Parameter(torch.randn(num_tags) * 0.01)

    def forward(self, emissions, tags, mask):
        """Negative log-likelihood loss (to be minimized)."""
        gold_score = self._score_sentence(emissions, tags, mask)
        forward_score = self._forward_alg(emissions, mask)
        nll = forward_score - gold_score
        return nll.mean()

    def _score_sentence(self, emissions, tags, mask):
        """Score of the gold tag path (numerator, in log-space)."""
        B, T, _ = emissions.shape
        mask = mask.float()

        score = self.start_transitions[tags[:, 0]] + emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)
        for t in range(1, T):
            emit = emissions[:, t].gather(1, tags[:, t:t + 1]).squeeze(1)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            score = score + (trans + emit) * mask[:, t]

        seq_lens = mask.sum(1).long() - 1
        last_tags = tags.gather(1, seq_lens.unsqueeze(1)).squeeze(1)
        score = score + self.end_transitions[last_tags]
        return score

    def _forward_alg(self, emissions, mask):
        """Log-partition function (denominator) via the forward algorithm."""
        B, T, C = emissions.shape
        mask = mask.float()

        alpha = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [B, C]
        for t in range(1, T):
            emit = emissions[:, t].unsqueeze(1)                          # [B, 1, C]
            trans = self.transitions.unsqueeze(0)                        # [1, C, C]
            broadcast = alpha.unsqueeze(2) + trans + emit                # [B, C(from), C(to)]
            new_alpha = torch.logsumexp(broadcast, dim=1)                # [B, C]
            m = mask[:, t].unsqueeze(1)
            alpha = new_alpha * m + alpha * (1 - m)

        alpha = alpha + self.end_transitions.unsqueeze(0)
        return torch.logsumexp(alpha, dim=1)

    def decode(self, emissions, mask):
        """Viterbi decoding: returns the single best tag sequence per example."""
        B, T, C = emissions.shape
        mask = mask.bool()

        backpointers = []
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]  # [B, C]

        for t in range(1, T):
            broadcast = score.unsqueeze(2) + self.transitions.unsqueeze(0)  # [B, C(from), C(to)]
            best_score, best_idx = broadcast.max(dim=1)                     # [B, C]
            emit = emissions[:, t]
            new_score = best_score + emit
            m = mask[:, t].unsqueeze(1)
            score = new_score * m + score * (~m)
            backpointers.append(best_idx)

        score = score + self.end_transitions.unsqueeze(0)
        best_final_score, best_last_tag = score.max(dim=1)

        seq_lens = mask.sum(1)
        best_paths = []
        for b in range(B):
            L = seq_lens[b].item()
            best_tag = best_last_tag[b].item()
            path = [best_tag]
            for t in range(L - 2, -1, -1):
                best_tag = backpointers[t][b, best_tag].item()
                path.append(best_tag)
            path.reverse()
            best_paths.append(path)
        return best_paths
