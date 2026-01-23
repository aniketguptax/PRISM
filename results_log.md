# Experimental Results Log
This is basically me keeping track of what I’ve tried, what broke, what worked, and what I didn’t understand at the time. It’s not meant to be polished. Some of these thoughts were wrong when I wrote them, or only made sense after later experiments. The ordering is roughly chronological, but not exact.

## Early baseline: Even Process + Golden Mean

I started by sanity checking the pipeline on standard toy processes, mostly to make sure nothing obvious was broken before moving on to anything more ambitious.

### Golden Mean

Golden Mean behaved almost exactly how I expected, which was reassuring more than anything else. Under `LastK`:

- k=1 already does most of the job
- k≥2 doesn’t meaningfully change log-loss
- The reconstructed machines are clean, unifilar, and stable
- Branching entropy is basically zero once k is large enough

This mostly served as a “nothing is obviously broken” check. If this *hadn’t* worked, I would’ve been worried.

### Even Process

This is where things got interesting, and also confusing.

Prediction improves steadily as k increases. Log-loss keeps going down, which makes sense: longer suffixes let you track parity information better.

But the reconstructed state machines *never* really become computationally closed:

- Unifilarity does not converge to 1
- Branching entropy stays noticeably > 0
- Even when log-loss is very good, the macrostate update remains stochastic

At first I thought this was a bug in my reconstruction code. I double checked transitions, plotted graphs, inspected counts. Everything looked internally consistent.

Eventually it clicked that this is actually the point: the Even Process is not finite-order Markov in the raw observations, so LastK can improve prediction without ever becoming a sufficient statistic for deterministic state updates. This has become an important reference point: prediction accuracy and closure are genuinely different things.

---

## Moving to a positive control: MarkovOrder2

After the Even Process, I really wanted a clean positive control; a process where I know closure _should_ exist at a finite k.

That’s why I implemented MarkovOrder2.

By definition:

- LastK with k=2 is sufficient for one step prediction
- In the infinite-data limit, k>2 should not improve prediction and should not break closure

So this was the right test of whether the reconstruction method can actually recover closure when it exists.

---

## First MarkovOrder2 runs

Initial runs were done at 500k samples, with eps fixed.

Some things worked immediately:

- Log-loss drops sharply from k=1 → k=2
- Log-loss completely plateaus for k≥2
- For k=2 and k=3:
  - Unifilarity = 1.0 across all seeds
  - Branching entropy = 0.0
  - Reconstructed machine has exactly 4 states, with deterministic updates
- The transitions for last_2 are exactly what you’d expect from an order-2 Markov chain

This was a *big* relief. At least at the “correct” scale, everything lines up.

But then…

### The weird part: k>3

For k=4,5,6:

- Log-loss stays flat (so prediction is still fine)
- But unifilarity starts to drop
- Branching entropy becomes nonzero and grows with k
- Number of states increases with k

This initially felt wrong. Why would *adding redundant history* make things worse?

After staring at this for a while, it became clear that this is a reconstruction and finite-sample effect:

- The number of distinct length k suffixes grows exponentially
- Many suffixes are rare, even at 500k samples
- Small estimation noise in \hat{p}(x_{t+1}|suffix) causes suffixes that *should* be equivalent (because only last 2 bits matter) to get split into different epsilon bins
- Once that happens, the induced macrostate update is no longer deterministic

So prediction stays good, but computational closure degrades. What’s interesting is that this looks superficially similar to the Even Process, but for a completely different reason. Here, closure fails because of estimation artefacts, not because closure doesn’t exist in principle.

That distinction feels important righr now.

---

## Increasing data: 1M samples

I reran the same sweep at length = 1M.

This helped, but didn’t completely eliminate the effect:

- k=2,3,4 remain perfectly unifilar
- k=5 starts to show small but nonzero branching
- k=6 still degrades, but less severely than at 500k

So increasing data pushes the problem out, but doesn’t eliminate it. Redundant contexts plus naive binning are still an issue.

At this point it felt clear that the process itself wasn’t the problem. The reconstruction procedure was.

---

## Introducing eps as a control parameter

Up to this point, eps had been fixed. That started to feel like a mistake.

I added a `--eps` flag to the CLI and nd started sweeping it explicitly at length = 1M.

---

### eps = 0.005

- k=2,3 still perfect
- k=4 already shows noticeable branching
- k=5,6 degrade quite badly
- Branching entropy grows fast with k

Eps too small, so over-splitting, so statistically equivalent suffixes are getting separated.

### eps = 0.01

- k=2,3 perfect
- k=4 mostly fine, but occasional small branching
- k=5,6 still degrade, but less violently than eps=0.005

Better, but still sensitive to redundancy.

### eps = 0.02

This one was interesting.

- k=2,3,4 are completely clean
- k=5 has only very small branching
- k=6 still degrades, but much less than before

This feels like the “sweet spot” so far: large enough eps to merge statistically equivalent suffixes, but not so large that it collapses genuinely different predictive contexts.

### eps = 0.05

- Everything up to k=5 is basically unifilar
- k=6 shows only very mild branching

At this point, the reconstruction is arguably _too_ aggressive, but it seems to demonstrate the point pretty clearly: closure at k=2 is real, and loss of closure at larger k is controllable via `eps` and data.

---

## What 500k vs 1M makes me think in general

- Computational closure _is_ emerging at the correct scale (Markov order 2, k = 2)
- Redundant representations are destroying cloesure without hurting predicition
- Whether closure is recovered depends, so far, on:
    - reconstruction choice
    - reconstruction algo
    - eps
    - noise
    - data length
    - number of seeds
    - probably other stuff
- The contrast with `Even Process` is really nicely, where closure geniuinely does not exist at finite `k` under `LastK`

Nice to have controlled experiements that back up the hypothesis.

Comparing the two has made me increasingly confident that the evaluation pipeline is not only "working", but kind of saying something nontrivial.