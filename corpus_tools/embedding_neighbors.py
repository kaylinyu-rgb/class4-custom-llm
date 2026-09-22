"""List the nearest words to a probe word, before and after training, using all
64 embedding numbers (cosine similarity: 1.0 = pointing the same direction).
Reads a run's checkpoint.json; changes nothing."""
import json, sys
import torch
from torch.nn import functional as F

path, words = sys.argv[1], sys.argv[2:] or ["customer"]
ck = json.load(open(path))
vocab = ck["vocabulary"]
for label, table in [("before", ck["initial_embeddings"]), ("after", ck["weights"]["wte"])]:
    t = F.normalize(torch.tensor(table), dim=1)
    for w in words:
        if w not in vocab:
            print(f"{w}: not in vocabulary"); continue
        sims = t @ t[vocab.index(w)]
        top = [(vocab[i], round(sims[i].item(), 2)) for i in sims.argsort(descending=True)[1:7]]
        print(f"{label:6} {w:9} -> {top}")
