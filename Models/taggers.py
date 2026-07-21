"""
taggers.py
==========
Three from-scratch sequence-labeling architectures, all sharing the same
input representation (pretrained word embedding + char-CNN embedding):

  1. LSTMTagger        - unidirectional LSTM, softmax per token
  2. BiLSTMTagger      - bidirectional LSTM, softmax per token
  3. BiLSTMCRFTagger   - bidirectional LSTM emissions + CRF decoding layer

Design note (why each step improves on the last):
  LSTM -> BiLSTM:   a unidirectional LSTM only sees left context, so at
                    token t it doesn't know what comes after. NER heavily
                    relies on right context too (e.g. "Washington" as PER
                    vs LOC depends on the following words). BiLSTM fixes
                    this by encoding both directions and concatenating.
  BiLSTM -> +CRF:   BiLSTM still tags each token independently (softmax),
                    so it can produce illegal/inconsistent label sequences
                    (I-PER right after O). The CRF layer models transition
                    scores between adjacent labels and decodes the globally
                    best sequence with Viterbi, which cleans up exactly
                    these boundary/consistency errors.
"""

import torch
import torch.nn as nn
from .char_encoder import CharCNNEncoder
from .crf import CRF


class _EmbeddingStack(nn.Module):
    """Shared input layer: pretrained word embedding + char-CNN embedding."""

    def __init__(self, vocab_size, word_emb_dim, char_vocab_size, char_emb_dim=30,
                 char_num_filters=50, pretrained_embeddings=None, freeze_word_emb=False, dropout=0.5):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_emb_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.word_embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))
        if freeze_word_emb:
            self.word_embedding.weight.requires_grad = False

        self.char_encoder = CharCNNEncoder(char_vocab_size, char_emb_dim, char_num_filters)
        self.output_dim = word_emb_dim + self.char_encoder.output_dim
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_ids, char_ids):
        w = self.word_embedding(word_ids)          # [B, T, word_emb_dim]
        c = self.char_encoder(char_ids)             # [B, T, char_out_dim]
        x = torch.cat([w, c], dim=-1)
        return self.dropout(x)


class LSTMTagger(nn.Module):
    """Unidirectional LSTM baseline."""

    def __init__(self, vocab_size, num_tags, char_vocab_size, word_emb_dim=100,
                 hidden_dim=128, pretrained_embeddings=None, dropout=0.5):
        super().__init__()
        self.embed = _EmbeddingStack(vocab_size, word_emb_dim, char_vocab_size,
                                      pretrained_embeddings=pretrained_embeddings, dropout=dropout)
        self.lstm = nn.LSTM(self.embed.output_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(hidden_dim, num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_ids, char_ids, mask=None):
        x = self.embed(word_ids, char_ids)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.classifier(out)   # [B, T, num_tags] emission scores

    def loss(self, emissions, labels, mask):
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        return criterion(emissions.view(-1, emissions.shape[-1]), labels.view(-1))

    def predict(self, emissions, mask):
        preds = emissions.argmax(-1)
        return [preds[b, :mask[b].sum()].tolist() for b in range(preds.shape[0])]


class BiLSTMTagger(nn.Module):
    """Bidirectional LSTM -- sees both left and right context."""

    def __init__(self, vocab_size, num_tags, char_vocab_size, word_emb_dim=100,
                 hidden_dim=128, num_layers=1, pretrained_embeddings=None, dropout=0.5):
        super().__init__()
        self.embed = _EmbeddingStack(vocab_size, word_emb_dim, char_vocab_size,
                                      pretrained_embeddings=pretrained_embeddings, dropout=dropout)
        self.lstm = nn.LSTM(self.embed.output_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
        self.dropout = nn.Dropout(dropout)

    def forward(self, word_ids, char_ids, mask=None):
        x = self.embed(word_ids, char_ids)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.classifier(out)

    def loss(self, emissions, labels, mask):
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        return criterion(emissions.view(-1, emissions.shape[-1]), labels.view(-1))

    def predict(self, emissions, mask):
        preds = emissions.argmax(-1)
        return [preds[b, :mask[b].sum()].tolist() for b in range(preds.shape[0])]


class BiLSTMCRFTagger(nn.Module):
    """BiLSTM emissions + CRF layer for globally-consistent decoding."""

    def __init__(self, vocab_size, num_tags, char_vocab_size, word_emb_dim=100,
                 hidden_dim=128, num_layers=1, pretrained_embeddings=None, dropout=0.5):
        super().__init__()
        self.embed = _EmbeddingStack(vocab_size, word_emb_dim, char_vocab_size,
                                      pretrained_embeddings=pretrained_embeddings, dropout=dropout)
        self.lstm = nn.LSTM(self.embed.output_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0)
        self.classifier = nn.Linear(hidden_dim * 2, num_tags)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, word_ids, char_ids, mask=None):
        x = self.embed(word_ids, char_ids)
        out, _ = self.lstm(x)
        out = self.dropout(out)
        return self.classifier(out)   # emissions, CRF handles the rest

    def loss(self, emissions, labels, mask):
        # CRF requires non-negative label ids for its gather ops; replace
        # ignored positions (-100) with 0 (masked out anyway by `mask`).
        safe_labels = labels.clone()
        safe_labels[safe_labels == -100] = 0
        return self.crf(emissions, safe_labels, mask)

    def predict(self, emissions, mask):
        return self.crf.decode(emissions, mask)


def build_model(arch: str, **kwargs):
    """Factory used by train_scratch.py and app.py to instantiate a model by name."""
    arch = arch.lower()
    if arch == "lstm":
        return LSTMTagger(**kwargs)
    elif arch == "bilstm":
        return BiLSTMTagger(**kwargs)
    elif arch in ("bilstm_crf", "bilstm-crf", "bilstmcrf"):
        return BiLSTMCRFTagger(**kwargs)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
