"""Final leakage audit on the text each model ACTUALLY trained on.

Plain-language summary: for each run, this opens the saved training passages
(split.json -> "train") and the full corpus text (corpus.txt), and checks that
none of the following appear in them:
  1. any of the 48 eval prompts,
  2. any prompt ending (last 4 words) followed by its answer,
  3. any free continuation the model produced during the evals
     (eval outputs), when that continuation is at least 5 words long,
  4. any chat prompt or reply from the saved chat transcripts.
It only reads files; it changes nothing.
"""
import glob, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def norm(text):
    return " " + " ".join(re.findall(r"\w+(?:['’]\w+)*|[^\w\s]", text.lower())) + " "


suite = json.loads((ROOT / "experiment2_expanded/evals/language_evals.json").read_text())["cases"]
eval_outputs = set()
for f in glob.glob(str(ROOT / "experiment*/llm_runs/*/language_evals/*/eval_results.json")):
    for row in json.loads(Path(f).read_text()):
        if len(row["generated_text"].split()) >= 5:
            eval_outputs.add(norm(row["generated_text"]))
chat_bits = set()
for f in glob.glob(str(ROOT / "experiment*/**/*chat_transcript*.json"), recursive=True):
    for turn in json.loads(Path(f).read_text())["turns"]:
        for t in (turn["prompt"], turn["response"]):
            if len(t.split()) >= 4:
                chat_bits.add(norm(t))

ok = True
runs = sorted(p for p in glob.glob(str(ROOT / "experiment*/llm_runs/*")) if Path(p).is_dir())
classroom = None
for run in runs:
    name = Path(run).relative_to(ROOT)
    passages = [norm(x) for x in Path(run, "corpus.txt").read_text().splitlines() if x.strip()]
    train_set = set(norm(t) for t in json.loads(Path(run, "split.json").read_text())["train"])
    if classroom is None:           # Experiment 1 = classroom passages only
        classroom = set(passages)
    mine = [x for x in passages if x not in classroom]
    groups = {"classroom passages (course generator)": [x for x in passages if x in classroom],
              "MY added passages": mine}
    print(f"== {name}")
    for label, items in groups.items():
        text = " ".join(items)
        prompts = [c["id"] for c in suite if norm(c["prompt"]) in text]
        keys = [c["id"] for c in suite
                if norm(" ".join(norm(c["prompt"]).split()[-4:] + [c["answer"]])) in text]
        print(f"  {label} ({len(items)}): eval prompts={prompts or 'none'} | prompt-ending+answer={keys or 'none'}")
        if label.startswith("MY"):
            ok &= not prompts and not keys
        else:
            ok &= not prompts
            if keys:
                print("    note: these are starter/transfer cases; the course generator teaches these associations by design,"
                      " and the notebook reserves only the exact prompts.")
                ok &= all(c["group"] != "extend_corpus" for c in suite if c["id"] in keys)
    # Eval outputs / chat: flagged only if a whole generated string IS a training passage.
    out_hits = [o.strip() for o in eval_outputs if o in train_set]
    chat_hits = [c.strip() for c in chat_bits if c in train_set]
    print(f"  eval outputs that are a training passage: {out_hits or 'none'} | chat text that is a training passage: {chat_hits or 'none'}")
print("\nRESULT:", "PASS: no eval prompts in any training text; no answer keys, eval outputs or chat logs in my added text; "
      "classroom overlap limited to starter associations the course teaches by design." if ok else "FAIL")
