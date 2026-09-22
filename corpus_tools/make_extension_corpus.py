"""Build the Experiment 2 teaching text for two extension skills: negation and
spatial relations.

Plain-language summary: this script writes many short practice examples, each
showing the SKILL (believe the positive statement; flip a spatial relation)
with people, objects, colours and places that the 48 tests do not use. It never
reads the test file. A separate script (check_separation.py) compares the
output against the tests afterwards.

Formatting note: the notebook splits text into separate training passages at
every "period followed by whitespace". A multi-sentence example such as
"the cup is not green. it is yellow. the cup is yellow." would be cut into
three unrelated pieces, so internal periods are written WITHOUT a following
space ("green.it is yellow.the cup"). The word tokenizer still produces exactly
the same tokens, including each ".", so the model sees normal sentences.

Separation rules this generator follows (written from my reading of the tests):
- Negation examples never use the tests' subjects (box, door, ava) or ANY of the
  negation tests' answer choices (red, blue, green, yellow, open, closed, wide,
  missing, tea, milk, rice, bread). The pair check below is a second safety net.
- Spatial examples never use the tests' objects or noun choices (book, bag, lamp,
  desk, ball, box, shelf) or the distractors north/south. Only the relation
  words themselves (above, below, left, right, inside, contains, beside) are shared.
- Test words may appear only in neutral_vocabulary.txt (hand-written), in
  sentences that do not state any tested relationship.
"""
import random
from pathlib import Path

rng = random.Random(7)
OUT = Path(__file__).resolve().parent.parent / "experiment2_expanded" / "corpus"


def join(*sentences):
    # "sentence one." + "sentence two." with no space, so the notebook keeps them together.
    return "".join(s.strip() + "." for s in sentences)


# ---------------------------------------------------------------- negation
things = ["cup", "car", "shirt", "chair", "wall", "kite", "hat", "bike", "plate",
          "flower", "coat", "towel", "sofa", "bottle", "blanket", "fence", "boat", "pen"]
colours = ["white", "black", "pink", "brown", "purple", "gray", "silver", "gold"]
states = [("window", ["clean", "dirty", "cracked", "foggy"]),
          ("gate", ["locked", "unlocked", "broken", "painted"]),
          ("room", ["bright", "tidy", "messy", "cozy"]),
          ("road", ["busy", "calm", "icy", "bumpy"]),
          ("drawer", ["stuck", "locked", "tidy", "messy"]),
          ("tap", ["dripping", "fixed", "broken", "rusty"])]
people = [("mia", "she"), ("zoe", "she"), ("lily", "she"), ("ruby", "she"), ("ivy", "she"),
          ("ben", "he"), ("sam", "he"), ("jack", "he"), ("max", "he"), ("theo", "he")]
items = ["coffee", "juice", "soup", "cheese", "eggs", "apples", "pasta", "honey",
         "butter", "yogurt", "cake", "beans"]
verbs = [("buy", "bought"), ("order", "ordered"), ("choose", "chose"), ("eat", "ate"),
         ("cook", "cooked"), ("pack", "packed")]
FORBIDDEN_NEGATION = {("red", "blue"), ("open", "closed"), ("tea", "milk")}


def pick_pair(options):
    while True:
        a, b = rng.sample(options, 2)
        # Block the tested pairs in BOTH directions (not red->blue AND not blue->red).
        if (a, b) not in FORBIDDEN_NEGATION and (b, a) not in FORBIDDEN_NEGATION:
            return a, b


negation = set()
while len(negation) < 420:
    kind = rng.random()
    if kind < .35:                      # colour of an object
        t = rng.choice(things); no, yes = pick_pair(colours)
        form = rng.random()
        if form < .6:
            negation.add(join(f"the {t} is not {no}", f"it is {yes}", f"the {t} is {yes}"))
        elif form < .8:
            negation.add(f"the {t} is not {no} , it is {yes} .")
        else:
            negation.add(join(f"the {t} was not {no}", f"it was {yes}", f"so the {t} was {yes}"))
    elif kind < .6:                     # state of a thing
        t, opts = rng.choice(states); no, yes = pick_pair(opts)
        if rng.random() < .75:
            negation.add(join(f"the {t} is not {no}", f"it is {yes}", f"the {t} is {yes}"))
        else:
            negation.add(join(f"the {t} is {yes} , not {no}", f"the {t} is {yes}"))
    else:                               # what a person did / did not do
        (name, pron), (base, past) = rng.choice(people), rng.choice(verbs)
        no, yes = pick_pair(items)
        form = rng.random()
        if form < .7:
            negation.add(join(f"{name} did not {base} {no}", f"{pron} {past} {yes}", f"{name} {past} {yes}"))
        else:
            negation.add(join(f"{name} {past} {yes} , not {no}", f"{name} {past} {yes}"))

# ----------------------------------------------------------------- spatial
objects = ["cup", "plate", "clock", "picture", "rug", "sofa", "chair", "table", "mirror",
           "plant", "basket", "bowl", "phone", "towel", "vase", "pillow", "bed", "sink",
           "window", "stool", "jar", "kettle", "cat", "dog"]
containers = ["drawer", "basket", "bowl", "jar", "cupboard", "fridge", "closet", "pocket",
              "suitcase", "envelope", "bucket", "pot"]
small = ["key", "coin", "spoon", "letter", "sock", "ring", "phone", "apple", "cup",
         "toy", "photo", "ticket", "shell", "note"]

spatial = set()
while len(spatial) < 420:
    kind = rng.random()
    if kind < .3:                         # above <-> below
        a, b = rng.sample(objects, 2)
        form = rng.random()
        if form < .5:
            spatial.add(join(f"the {a} is above the {b}", f"the {b} is below the {a}"))
        elif form < .8:
            spatial.add(join(f"the {a} is below the {b}", f"the {b} is above the {a}"))
        else:
            spatial.add(f"the {a} is above the {b} , so the {b} is below the {a} .")
    elif kind < .6:                       # left <-> right
        a, b = rng.sample(objects, 2)
        form = rng.random()
        if form < .45:
            spatial.add(join(f"the {a} is left of the {b}", f"the {b} is to the right of the {a}"))
        elif form < .9:
            spatial.add(join(f"the {a} is right of the {b}", f"the {b} is to the left of the {a}"))
        else:
            spatial.add(f"the {a} is left of the {b} , so the {b} is to the right .")
    elif kind < .9:                       # inside <-> contains
        s, c = rng.choice(small), rng.choice(containers)
        form = rng.random()
        if form < .6:
            spatial.add(join(f"the {s} is inside the {c}", f"the {c} contains the {s}"))
        elif form < .85:
            spatial.add(join(f"the {c} contains the {s}", f"the {s} is inside the {c}"))
        else:
            spatial.add(f"the {s} is inside the {c} , so the {c} contains the {s} .")
    else:                                 # beside is symmetric
        a, b = rng.sample(objects, 2)
        spatial.add(join(f"the {a} is beside the {b}", f"the {b} is beside the {a}"))

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "negation_generated.txt").write_text("\n".join(sorted(negation)) + "\n", encoding="utf-8")
(OUT / "spatial_generated.txt").write_text("\n".join(sorted(spatial)) + "\n", encoding="utf-8")
print("negation examples:", len(negation), "| spatial examples:", len(spatial))
