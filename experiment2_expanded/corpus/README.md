# Experiment 2 teaching text

(The notebook skips this README when it reads `corpus/`, so none of this text is training data.)

Extension skills: **negation** and **spatial relations**.

- `negation_generated.txt`, `spatial_generated.txt`: 420 examples each, made by
  `../../corpus_tools/make_extension_corpus.py` (fixed random seed, so rerunning it produces identical files).
- `negation_handwritten.md`, `spatial_handwritten.md`: 25 + 20 hand-written examples in more natural,
  varied wording (e.g. "hangs above", "sleeps below", "fits well").
- `neutral_vocabulary.txt`: 23 plain sentences that put everyday words into the vocabulary
  (book, bag, desk, lamp, ball, box, door, open, closed, colours, tea, milk, north, south...).
  None of them states a negation or a spatial relationship.

Internal periods have no following space on purpose: the notebook splits passages at
"period + whitespace", and multi-sentence examples must stay in one passage.
The tokens the model sees are unchanged.
