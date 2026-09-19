import re
import torch
import torch.nn as nn

PAD_ID = 0
UNK_ID = 1

def tokenize(text):
    return re.findall(r"[A-Za-z0-9]+", text.lower())

def make_pair(question, context):
    return question + " [SEP] " + context

class Vocabulary:
    def __init__(self, min_freq=2, max_size=30000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.itos = ["<pad>", "<unk>"]
        self.stoi = {"<pad>": PAD_ID, "<unk>": UNK_ID}

    def build(self, texts):
        from collections import Counter
        counts = Counter()
        for text in texts:
            counts.update(tokenize(text))
        for token, count in counts.most_common(self.max_size - 2):
            if count >= self.min_freq and token not in self.stoi:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)

    def encode(self, text, max_len=192):
        ids = [self.stoi.get(t, UNK_ID) for t in tokenize(text)[:max_len]]
        return ids if ids else [UNK_ID]

class GRURanker(nn.Module):
    def __init__(self, vocab_size, emb_dim=96, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.gru = nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 96),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(96, 1)
        )

    def forward(self, ids, lengths):
        x = self.embedding(ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(representation).squeeze(1)
