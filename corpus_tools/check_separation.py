"""Check that Experiment 2's teaching text is separate from the 48 fixed tests.

This script READS the test file only to compare against it. It never writes
training text or vocabulary. Checks:
1. No test prompt appears inside any teaching passage (same check as the notebook).
2. No passage contains a test's final words followed by that test's answer
   (the last 4 prompt words + answer), which would be a leaked answer key.
3. Negation passages avoid the tests' subjects/pairs; spatial passages avoid the tests' objects.
4. Reports the longest run of consecutive words shared with any test prompt.
5. Reports the vocabulary size so rare words are not silently dropped (limit 509).
"""
import json, re, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
CORPUS = HERE / "experiment2_expanded" / "corpus"
SUITE = json.loads((HERE / "experiment2_expanded" / "evals" / "language_evals.json").read_text())


def tokens(text):
    return re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower(), flags=re.UNICODE)


def passages(text):
    # Same splitting rule as the notebook's chunk_text.
    return [" ".join(tokens(u)) for u in re.split(r"(?<=[.!?])\s+|\n+", text) if tokens(u)]


def longest_shared_run(a, b):
    best, prev = 0, [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


files = sorted(p for p in CORPUS.rglob("*") if p.suffix in {".txt", ".md"} and p.name != "README.md")
problems, all_passages = [], []
for f in files:
    for p in passages(f.read_text(encoding="utf-8")):
        all_passages.append((f.name, p))

rules_negation = [(w,) for w in "box door ava red blue green yellow open closed wide missing tea milk rice bread".split()]
for name, p in all_passages:
    padded, toks = f" {p} ", p.split()
    for case in SUITE["cases"]:
        prompt = " ".join(tokens(case["prompt"]))
        if f" {prompt} " in padded:
            problems.append(f"{name}: contains test prompt {case['id']}: {p}")
        # Last 4 prompt words + answer. (A 3-word tail flagged the generic phrase
        # "is to the right", which every left/right teaching example needs; the
        # 4-word tail includes the test's own object, e.g. "box is to the right".)
        tail = " ".join(tokens(case["prompt"])[-4:] + [case["answer"]])
        if f" {tail} " in padded:
            problems.append(f"{name}: contains prompt ending + answer of {case['id']}: {p}")
    if name.startswith("negation"):
        if any(w in toks for (w,) in rules_negation):
            problems.append(f"{name}: uses a negation test subject or choice word: {p}")
        for no, yes in [("red", "blue"), ("open", "closed"), ("tea", "milk")]:
            if no in toks and yes in toks:
                problems.append(f"{name}: uses the tested pair {no}/{yes}: {p}")
    if name.startswith("spatial"):
        for w in ["book", "bag", "lamp", "desk", "ball", "box", "shelf", "north", "south"]:
            if w in toks:
                problems.append(f"{name}: uses spatial test object '{w}': {p}")
    if name.startswith("neutral"):
        for w in ["not", "above", "below", "inside", "contains", "left", "right", "beside"]:
            if w in toks:
                problems.append(f"{name}: neutral sentence uses relation word '{w}': {p}")

overlaps = []
for name, p in all_passages:
    for case in SUITE["cases"]:
        overlaps.append((longest_shared_run(p.split(), tokens(case["prompt"])), case["id"], name, p))
overlaps.sort(reverse=True)
print("Passages checked:", len(all_passages), "from", len(files), "files")
print("Longest shared word runs with a test prompt (top 5):")
for n, cid, name, p in overlaps[:5]:
    print(f"  {n} words | {cid} | {name}: {p}")
hist = Counter(n for n, *_ in overlaps if n >= 4)
print("Passage-test pairs sharing >=4 consecutive words:", dict(sorted(hist.items())))

# Vocabulary budget: classroom text (136 types) + these files must stay within 509.
new_types = Counter(t for _, p in all_passages for t in p.split())
print("Word/punctuation types in teaching text:", len(new_types))

if problems:
    print("\nSEPARATION PROBLEMS:"); print(*problems, sep="\n"); sys.exit(1)
print("\nOK: no test prompts, no prompt-ending+answer sequences, no forbidden pairs/objects.")
