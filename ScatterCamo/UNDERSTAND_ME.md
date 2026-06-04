# UNDERSTAND_ME — ScatterCamo explained from zero

> You said you just started with ML and have no idea what's going on. Good — this
> file assumes **exactly that**. No prior knowledge. We build up from "what is an
> image classifier" all the way to "what this repo actually does," and only at the
> end do we touch the math (and even then, gently). Read it top to bottom.

---

## 1. The 30-second version

There are AI models that look at a photo and say *"this is a cat."* Those are
called **image classifiers**.

It turns out you can add a small, almost-invisible smudge to the photo so the
model suddenly says *"this is a guacamole"* — even though to **your** eyes it's
obviously still a cat. That sneaky smudge is called an **adversarial attack**.

**ScatterCamo is a program that finds such a smudge.** Specifically, it finds a
smudge made of a few small colored blobs scattered around the image, and it tries
to make that smudge as **invisible as possible** while still fooling the model.

That's the whole idea. The rest of this file explains *how* it pulls that off.

---

## 2. The words you need (mini-dictionary)

You'll see these everywhere. Learn them once here.

| Word | What it really means |
|---|---|
| **Model / classifier** | The AI that looks at an image and outputs a guess like "cat (87%)". Here it's `resnet50` or `vgg16_bn` — famous pre-trained models. |
| **Image** | To a computer, a grid of numbers. Each pixel is 3 numbers: how much Red, Green, Blue (each from 0 to 1). |
| **Label / class** | One of 1000 categories the model knows (ImageNet has 1000: goldfish, school bus, etc.). Each has a number 0–999. "Label 8" = a specific category. |
| **Logits** | The raw scores the model gives each of the 1000 classes before picking a winner. Higher = more confident. |
| **Perturbation** | The smudge/change we add to the image. The thing we're searching for. |
| **Adversarial** | An image that *successfully fools* the model (model's top guess is now wrong). |
| **Untargeted attack** | We just want the model to be **wrong** — we don't care *which* wrong answer it gives. (A *targeted* attack would force a specific wrong answer.) |
| **Black-box** | We are **not allowed to look inside** the model. We can only feed it an image and read its output scores. Like poking a vending machine without a manual. |
| **Query** | One single act of "feed image to model, read its scores." Black-box attacks count these because each one costs time/money. We have a **budget** (e.g. 10,000 queries). |

---

## 3. Why is fooling a model even possible?

A classifier draws invisible boundaries in "image space." On one side of a
boundary it says "cat," on the other "dog." These boundaries are weirdly
wiggly. Most photos sit comfortably inside the "cat" region — but the **edge** is
often surprisingly close. A tiny, cleverly-chosen nudge can shove the image
across the boundary into "dog" territory, even though the nudge is too small for a
human to notice.

The attack's job: **find the cheapest direction to nudge.**

---

## 4. What kind of smudge does ScatterCamo use?

Most attacks add random noise everywhere, or paste one obvious square sticker.
ScatterCamo does something in between: it draws **`M` small colored circles**
("shapes") and scatters them on the image. `M` is a number you choose — e.g.
`M=10` means ten little blobs.

Each blob is described by **7 numbers** (this list is the blob's "DNA"):

```
(y, x, radius, R, G, B, alpha)
 │   │    │     └──┬──┘   └── how see-through the blob is (0 = invisible, 1 = solid paint)
 │   │    │        └── the blob's color (Red, Green, Blue)
 │   │    └── how big the blob is
 └───┴── where the blob sits (vertical, horizontal position)
```

A full smudge of `M` blobs is just `M` of these 7-number rows stacked together.
That whole stack is called a **genome** (more on that word in §6). To turn a
genome into an actual image, the code paints each circle onto the original photo,
one after another — that's the `generate_image` function in
`scattercamo/representation/shapes.py`.

> **Why blobs instead of noise?** Because a few translucent blobs that roughly
> match the local colors are *much* harder to spot than full-image static. That's
> the "Camo" (camouflage) in ScatterCamo.

`M` is a dial you can turn:
- **Small `M`, big blobs** → looks like one camouflaged patch.
- **Large `M`, tiny blobs** → looks like sparse scattered specks.
Sweeping `M` lets you explore the trade-off between "few changes" and "invisible."

---

## 5. Two things we want at once (and they fight)

ScatterCamo is trying to satisfy **two goals simultaneously**:

1. **Fool the model** (be adversarial). → measured by the **loss**.
2. **Stay invisible** (change the photo as little as possible). → measured by **L2**.

These pull in opposite directions: the easiest way to fool a model is a *huge*
ugly smudge, but that's very visible. The most invisible smudge is *no* smudge,
but that fools nobody. We want the sweet spot. This "two goals at once" situation
is called **multi-objective optimization**.

A couple of the measuring sticks:

- **Loss** — a single number saying "how close are we to fooling the model?"
  Lower is better. When it crosses zero, the model's top guess has flipped and
  we're adversarial. (Defined in `scattercamo/losses/margin.py`.)
- **L2** — basically "how much did we change the picture, totaled up over every
  pixel?" Lower = more invisible. (See `l2_perturbation` in `shapes.py`.)
- **L0** — "how many pixels did we touch at all?" (sparsity).
- **SSIM** — "how structurally similar does it still look to a human?" Higher = better.

---

## 6. How does it actually *find* a good smudge? (Evolution!)

Here's the clever part. We can't open the model and do calculus on it (it's
black-box). So instead ScatterCamo uses **a genetic algorithm** — it literally
mimics natural selection. This is why blob-DNA is called a "genome."

The loop, in plain English:

1. **Start with a random crowd.** Make ~20 random smudges (`pop_size=20`). This
   crowd is called the **population**. Each individual smudge is a **Solution**.

2. **Score everyone.** For each smudge, feed the smudged image to the model
   (that's one **query** each) and record its two scores: loss and L2.

3. **Pick the fittest as parents.** Hold little "tournaments": grab two random
   smudges, keep the better one, repeat. Winners become **parents**.
   (`tournament_selection`.)

4. **Breed children.**
   - **Crossover**: take two parent smudges and swap some of their blobs to make a
     child (`pc=0.3` controls how many blobs get swapped).
   - **Mutation**: randomly tweak a child's blobs — nudge a position, re-roll a
     color (`pm=0.3` controls how often). (Both in `scattercamo/operators/`.)

5. **Survival of the fittest.** Now you have parents + children. Keep only the
   best `pop_size` of them for the next generation; the rest die off.

6. **Repeat** steps 2–5 until you run out of your query budget.

Over many generations the crowd drifts toward smudges that are both adversarial
**and** nearly invisible — without us ever needing to look inside the model.
This whole loop lives in `scattercamo/attack/scattercamo.py`.

### "How do we rank who's 'best' when there are TWO goals?"

This is the heart of the method, called **NSGA-II** (a famous algorithm). Two ideas:

- **Domination / Pareto fronts.** Smudge A "dominates" smudge B if A is at least
  as good on *both* goals and strictly better on at least one. The smudges that
  nobody dominates form the **Pareto front** — the current best trade-offs. We
  rank the whole population into layers (front 1 = best, front 2 = next, ...).

- **Prioritized domination (the special sauce from SA-MOO).** Normally the two
  goals are treated equally. ScatterCamo cheats *on purpose*: it says **"first
  become adversarial, worry about invisibility later."** So a smudge that fools
  the model always beats one that doesn't, no matter how pretty the non-fooling
  one is. Only *among* the fooling smudges do we then compete on invisibility.
  This is what `is_adversarial` is doing in the loss — it flips the search into
  "now minimize L2" mode the moment we succeed.

- **Crowding distance.** A tie-breaker that prefers smudges in less-crowded parts
  of the trade-off curve, so the crowd stays diverse instead of all clumping in
  one spot.

That's genuinely the entire algorithm. Random crowd → score → breed the winners →
repeat, with a smart two-goal ranking that prioritizes fooling first.

---

## 7. Following one run end-to-end

When you run the program on a cat photo (true label = "cat"):

```
your image (cat)
      │
      ▼
make 20 random blob-smudges  ──────────────┐
      │                                     │
      ▼                                     │
paint each smudge onto the cat photo        │  ← representation/shapes.py
      │                                     │
      ▼                                     │
ask the model: "what is this?" (a query)    │  ← models/imagenet.py
      │  it returns 1000 scores             │
      ▼                                     │
compute loss + L2 for each smudge           │  ← losses/margin.py
      │                                     │  this whole box repeats
      ▼                                     │  every generation until
rank them (NSGA-II, fooling-first)          │  the query budget runs out
      │                                     │  ← search/ + attack/
      ▼                                     │
breed winners → children (crossover+mutate) │  ← operators/
      │                                     │
      ▼                                     │
keep the best 20, discard the rest  ────────┘
      │
      ▼
return the best smudge that fools the model with the SMALLEST visible change
```

The final output tells you: did we succeed, how many queries it took, and the
quality numbers (L0 / L2 / SSIM) of the winning smudge.

---

## 8. A peek at the actual math (optional, gentle)

You can stop at §7 and understand the project fine. But here's the one formula
worth seeing — the **margin loss** (untargeted), from `losses/margin.py`:

```
loss = f_true − f_other
```

- `f_true` = the model's score for the **correct** class (e.g. "cat").
- `f_other` = the model's score for the **best non-correct** class (the runner-up).

Read it like this:
- If the model is still confident it's a cat, `f_true` is big, `f_other` is
  smaller, so `loss` is **positive** → not fooled yet.
- As our smudge eats into the cat-confidence and lifts some other class, `f_true`
  drops and `f_other` rises. When `f_other` overtakes `f_true`, `loss` goes
  **negative** → the model's top guess flipped → **we fooled it.** 🎉

So "minimize the loss until it crosses zero" literally means "push the runner-up
past the true class." That single number is the compass the whole evolution
steers by. (The `log(exp(...))` bits in the code are just numerical-stability
plumbing — ignore them for now.)

---

## 9. Where to look in the code (a tiny map)

You don't need to read all of it. If you're curious, this is the reading order
that matches this document:

| Concept here | File |
|---|---|
| The blob DNA + painting blobs onto an image | `scattercamo/representation/shapes.py` |
| The "are we fooling it yet?" loss | `scattercamo/losses/margin.py` |
| Talking to the actual AI model + counting queries | `scattercamo/models/imagenet.py` |
| One candidate smudge (a Solution) | `scattercamo/search/solution.py` |
| NSGA-II ranking (fronts, crowding, domination) | `scattercamo/search/` |
| Breeding (crossover + mutation) | `scattercamo/operators/` |
| The master loop that ties it all together | `scattercamo/attack/scattercamo.py` |
| The "run it on one image" entry point | `run_attack.py` |

---

## 10. What to do next

- To actually **run** it, read **`USAGE.md`** (sits right next to this file).
- To go deeper on the algorithm and the research it's based on, read
  `docs/method.md`. For *why* the authors made each design choice, `docs/design.md`.

You now know more than enough to follow along. The core mental model to keep:
**"evolve a crowd of tiny camouflaged smudges, keep the ones that fool the model,
and among those keep the most invisible."** Everything else is detail.
