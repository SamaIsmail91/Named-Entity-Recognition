"""
char_encoder.py
================
Character-level CNN that produces a fixed-size "spelling" embedding for
each word. This is concatenated with the (frozen or fine-tuned) GloVe word
embedding before being fed to the LSTM/BiLSTM encoders.

Why this matters for OOV handling:
A rare/unseen word like "Zelenskyy" has no reliable GloVe vector, but its
*shape* -- capital first letter, length, character n-grams -- is still
informative. A small CNN over the character sequence learns these spelling
patterns end-to-end, giving the tagger a signal even for words it has never
seen in the embedding table.
"""

import torch
import torch.nn as nn


class CharCNNEncoder(nn.Module):
    def __init__(self, char_vocab_size, char_emb_dim=30, num_filters=50, kernel_sizes=(3, 4, 5), dropout=0.3):
        super().__init__()
        self.char_embedding = nn.Embedding(char_vocab_size, char_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=char_emb_dim, out_channels=num_filters, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, char_ids):
        """
        char_ids: [batch, seq_len, max_word_len]
        returns:  [batch, seq_len, output_dim]
        """
        B, T, W = char_ids.shape
        x = char_ids.view(B * T, W)                       # [B*T, W]
        emb = self.char_embedding(x)                        # [B*T, W, char_emb_dim]
        emb = emb.transpose(1, 2)                            # [B*T, char_emb_dim, W]

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(emb))                        # [B*T, num_filters, W']
            p = torch.max(c, dim=2).values                   # max-over-time pooling -> [B*T, num_filters]
            pooled.append(p)

        out = torch.cat(pooled, dim=1)                        # [B*T, output_dim]
        out = self.dropout(out)
        return out.view(B, T, self.output_dim)
