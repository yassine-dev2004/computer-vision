"""Encoder-decoder Transformer that rewrites ASL gloss sequences as English."""
import math

import torch
from torch import nn

from .data import PAD_IDX


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class GlossTranslator(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=4, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        nn.init.normal_(self.embed.weight, std=d_model ** -0.5)
        with torch.no_grad():
            self.embed.weight[PAD_IDX].zero_()
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers,
                                          dim_feedforward, dropout, batch_first=True)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.embed.weight  # shared vocab, so input and output embeddings are tied

    def _embed(self, ids):
        return self.pos(self.embed(ids) * math.sqrt(self.d_model))

    def encode(self, src):
        src_pad = src == PAD_IDX
        memory = self.transformer.encoder(self._embed(src), src_key_padding_mask=src_pad)
        return memory, src_pad

    def decode(self, tgt, memory, src_pad):
        size = tgt.size(1)
        causal = torch.triu(torch.ones(size, size, dtype=torch.bool, device=tgt.device), diagonal=1)
        h = self.transformer.decoder(self._embed(tgt), memory, tgt_mask=causal,
                                     tgt_key_padding_mask=tgt == PAD_IDX,
                                     memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, tgt_in):
        memory, src_pad = self.encode(src)
        return self.decode(tgt_in, memory, src_pad)

    @torch.no_grad()
    def greedy_decode(self, src, bos_idx, eos_idx, max_len=80):
        memory, src_pad = self.encode(src)
        ys = torch.full((src.size(0), 1), bos_idx, dtype=torch.long, device=src.device)
        done = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            next_tok = self.decode(ys, memory, src_pad)[:, -1].argmax(-1).masked_fill(done, PAD_IDX)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            done |= next_tok == eos_idx
            if done.all():
                break
        return ys[:, 1:]

    @torch.no_grad()
    def beam_search(self, src, bos_idx, eos_idx, beam_size=5, max_len=80, length_penalty=0.6):
        """Beam search for one sentence (src shape (1, S)); used for the live demo and the test set."""
        memory, src_pad = self.encode(src)
        beams, finished = [([bos_idx], 0.0)], []
        for _ in range(max_len):
            tgt = torch.tensor([seq for seq, _ in beams], device=src.device)
            k = len(beams)
            logp = self.decode(tgt, memory.expand(k, -1, -1), src_pad.expand(k, -1))[:, -1].log_softmax(-1)
            candidates = []
            for (seq, score), row in zip(beams, logp):
                top = row.topk(beam_size)
                candidates += [(seq + [tok], score + lp) for lp, tok in zip(top.values.tolist(), top.indices.tolist())]
            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in candidates:
                (finished if seq[-1] == eos_idx else beams).append((seq, score))
                if len(beams) == beam_size:
                    break
            if not beams or len(finished) >= beam_size:
                break
        pool = finished or beams
        best = max(pool, key=lambda c: c[1] / (((5 + len(c[0])) / 6) ** length_penalty))
        return best[0][1:]
