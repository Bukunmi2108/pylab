# Week 14 — Preference optimization, evaluation & capstone

The final week closes the modern post-training loop: SFT taught your model a format; DPO teaches it a *preference* between better and worse outputs — the same alignment stage (minus PPO machinery) behind the models you ship prompts against at work. Then you build what your day job has taught you to value: an evaluation harness — automatic metrics plus a blind human-preference tool, all offline, with you as the judge. You finish by assembling base→SFT→DPO into a reproducible, documented artifact and proving it reproduces from its own docs. After this week the curriculum ends and your own projects begin.

**Week outcome:** by day 7 you have: `tlib/dpo.py` and `tlib/evalsuite.py` (pytest-covered, with a finite-difference-checked DPO loss), `checkpoints/dpo_final.pt`, a blind A/B judging CLI with a recorded 20-trial SFT-vs-DPO result, a working chat CLI on your best model, `MODEL_CARD.md` from which the entire pipeline reproduces, a tagged v1.0 repo, and `FINAL_REPORT.md`.

## Day 14.1 — From RLHF to DPO: the loss, from the paper (~2.5h)
- [ ] done

**Goal:** implement and *prove correct* the DPO loss and the sequence-logprob function it consumes.

**Learn:**
- **The problem after SFT:** SFT clones demonstration style — it can't express "output A is better than B" because it only ever sees one target per prompt. Ranking information needs a different objective.
- **RLHF in one paragraph (not built):** InstructGPT trains a reward model on human preference pairs, then optimizes the policy against it with PPO, with a KL penalty to a reference model to stop the policy gaming the reward. Effective, but two models + RL machinery.
- **DPO's move:** the KL-constrained RLHF optimum can be inverted — the optimal policy *implicitly defines* the reward as r(x,y) = β·log(π(y|x)/π_ref(y|x)). Substituting into the Bradley–Terry preference model collapses the whole pipeline into one classification-style loss on the policy directly: **L = −log σ(β[(log π(y_w|x) − log π_ref(y_w|x)) − (log π(y_l|x) − log π_ref(y_l|x))])** where y_w/y_l are chosen/rejected.
- **β's role:** how hard you trade preference fit against staying close to the reference. High β = small log-ratio differences already saturate the sigmoid (stay close to ref); low β = policy must move far to reduce the loss. Typical 0.1–0.5.
- **Sequence logprob reuses Week-13 masking:** log π(y|x) = Σ over *response* tokens of log p(token | prefix) — exactly the positions your SFT labels left unmasked. Same shift, same −100 convention; sum instead of mean.
- **Hand-computable example** (verify on paper first, then encode it as a test):
  ```text
  policy_chosen_lp = -2.0    ref_chosen_lp   = -2.5
  policy_rejected_lp = -4.0  ref_rejected_lp = -3.0    beta = 0.5

  chosen   log-ratio = -2.0 - (-2.5) =  0.5
  rejected log-ratio = -4.0 - (-3.0) = -1.0
  margin = 0.5 - (-1.0) = 1.5;   z = beta * margin = 0.75
  loss = -log(sigmoid(0.75)) = log(1 + e^-0.75) ~= 0.38687
  implicit rewards: chosen = beta*0.5 = 0.25, rejected = beta*(-1.0) = -0.5
  ```
  Interpretation: the policy already up-weights the chosen and down-weights the rejected relative to the reference, so the loss sits below the ln 2 ≈ 0.6931 starting point.

**Read (30–45 min):**
- DPO paper §1–4 (the derivation §4 carefully, Eq. 7) + Appendix A.4 if you want the gradient: https://arxiv.org/abs/2305.18290
- InstructGPT §3.5–3.6 for the RLHF pipeline you're replacing: https://arxiv.org/abs/2203.02155
- `F.logsigmoid` (use it — `log(sigmoid(x))` overflows for large negative x): https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html

**Build:**
1. Create `tlib/dpo.py`:
   ```python
   def sequence_logprob(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
       """input_ids (B, T); labels (B, T) = input_ids copy with prompt+pad = -100.
       logits = model(input_ids); use logits[:, :-1] vs labels[:, 1:]:
       per-position log-softmax, gather target logprob where label != -100,
       zero elsewhere, SUM over T. Returns (B,). Differentiable."""

   def dpo_loss(
       policy_chosen_lp: torch.Tensor, policy_rejected_lp: torch.Tensor,
       ref_chosen_lp: torch.Tensor, ref_rejected_lp: torch.Tensor,
       beta: float = 0.1,
   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       """All inputs (B,). Returns (mean loss, chosen_rewards (B,), rejected_rewards (B,))
       where rewards = beta * (policy_lp - ref_lp), detached.
       loss = -F.logsigmoid(beta * ((pc - rc) - (pr - rr))).mean()"""
   ```
   Gather trick for the −100 positions: `safe = labels.clamp(min=0)` for the gather index, then multiply by `(labels != -100)`.
2. Write the worked example above as a test before running it; predict the three numbers, then assert them.

**Verify — done when** (`tlib/tests/test_dpo.py`):
- **Init guarantee:** with `policy_*_lp == ref_*_lp` (any values), loss `allclose(math.log(2))` ≈ 0.6931 — exactly −log σ(0). This is your first assert during real training too.
- Worked example: loss `allclose(0.38687, atol=1e-4)`; rewards `allclose((0.25, -0.5))`.
- **Monotonicity:** holding everything else fixed, increasing policy_chosen_lp strictly decreases the loss; increasing policy_rejected_lp strictly increases it (assert on 3 crafted points each).
- **Finite-difference gradient check (float64):** wrap dpo_loss in a function of the 4 input tensors (`requires_grad_(True)`, dtype double, B=3 random values in [−10, 0]); compare autograd grads to central differences (ε = 1e-6) with `allclose(atol=1e-7)` — or simply `torch.autograd.gradcheck` on it. Analytic spot-check: in the worked example, ∂L/∂policy_chosen_lp = −β·σ(−z) = −0.5·σ(−0.75) ≈ **−0.1604**.
- `sequence_logprob` masking: on a hand-built 1-sequence batch, equals the manual sum of the response tokens' log-softmax entries (`allclose`); changing a *masked* position's token id changes the result only through the model's context (test with a 2-token response where the masked change is *after* all response tokens → result identical).

**If stuck:** DPO paper Eq. 7 and the paragraph after it; `gradcheck` docs: https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html; your Day-13.3 masking test.

## Day 14.2 — Preference data & the DPO training loop (~2.5h)
- [ ] done

**Goal:** construct a synthetic preference dataset honestly suited to your tiny model, and train DPO with a verified-frozen reference.

**Learn:**
- **Real preference data** is humans ranking model outputs. You can't collect that at scale solo, so you'll construct pairs where "chosen > rejected" holds *by construction*: chosen = the SFT dataset's real response; rejected = a degraded version. **Honest framing:** this teaches the mechanics and gives DPO a learnable signal; it does not teach the model human taste. Say so in your MODEL_CARD.
- **Three corruption recipes for rejected:** (a) *truncation* — cut the chosen response at 40% of its tokens, no `<|end|>` (teaches: don't stop mid-thought); (b) *shuffle* — randomly permute the response's words (teaches: coherence); (c) *off-policy degenerate* — generate from the **base** (pre-SFT) model at temperature 1.4 given the same prompt (teaches: prefer on-format over off-format). Use a mix, ~1/3 each, ~1,500 pairs from your Day-13.3 jsonl.
- **The frozen reference:** `ref = copy.deepcopy(policy); ref.eval(); [p.requires_grad_(False) for p in ref.parameters()]`; all ref logprobs computed under `torch.no_grad()`. Only the policy's optimizer exists. The reference is the anchor — if it drifts, the implicit-reward interpretation collapses.
- **What to track:** loss, mean reward margin (chosen − rejected rewards; should rise), and a KL proxy: mean (policy_chosen_lp − ref_chosen_lp) — how far the policy has moved *on chosen sequences* (large positive = moving toward chosen; watch it doesn't explode, which means it's leaving the reference's support).
- **Start DPO from the SFT checkpoint** (policy *and* reference) — DPO assumes the reference already produces in-distribution responses.

**Read (30–45 min):**
- DPO paper §5–6 (experimental setup — note β values used and that ref = SFT model).
- Python `copy.deepcopy` docs: https://docs.python.org/3/library/copy.html
- Your Trainer's API — decide bespoke loop vs Trainer subclass before coding (a bespoke loop is fine here: two forward passes per model per batch don't fit a plain Trainer cleanly).

**Build:**
1. `week14/make_pref_data.py` → `week14/pref_data.jsonl` of `{"prompt": ..., "chosen": ..., "rejected": ..., "recipe": "trunc|shuffle|degen"}`. Reuse the Day-13.3 templates for prompts; for recipe (c), load `base_final.pt` once and batch-generate.
2. `week14/train_dpo.py`:
   - `PrefDataset` tokenizes prompt+chosen and prompt+rejected separately, each via the Day-13.3 path (template, `<|end|>` appended, prompt positions −100); one item = `{chosen_ids, chosen_labels, rejected_ids, rejected_labels}`.
   - Collate pads ids with pad_id and labels with −100, chosen and rejected batched as two (B, T) pairs.
   - Loop per batch: 4 calls to `sequence_logprob` (policy/ref × chosen/rejected; the two ref calls under `torch.no_grad()`), `dpo_loss(β=0.1)`, clip 1.0, AdamW lr 1e-6–5e-6 (DPO moves logits hard; start low), 1–2 epochs.
   - Log loss / mean margin / KL-proxy every 20 steps to `week14/dpo_log.csv`; hold out 10% of pairs for val margin; save `checkpoints/dpo_final.pt`.
3. Plot margin and loss curves → `week14/dpo_curves.png`.

**Verify — done when:**
- **Step-0 sanity (guaranteed):** first logged train loss `allclose(ln 2, atol=1e-3)` — policy == reference. If it isn't, your masks or logprobs differ between the two models; stop and fix.
- **Reference frozen:** hash (or store `.clone()` of two ref tensors) before training; after N steps, `torch.equal` holds. Assert in the script.
- **Margins trend up:** mean train margin over the last 10% of steps > mean over the first 10% (assert; with constructed preferences this is reliable). Val margin also above its initial value — if train margin rises but val doesn't, you memorized pairs; note it honestly.
- Checkpoint saved and loads into `DecoderLM` cleanly.

**If stuck:** DPO §5 hyperparameters; your Day-14.1 init-guarantee test isolates loss-side bugs from data-side bugs; check both sequences end in `<|end|>` and that pad positions are −100 in *both* labels.

## Day 14.3 — Evaluation harness I: automatic metrics (~2h)
- [ ] done

**Goal:** build the offline metric suite and generate the base-vs-SFT-vs-DPO comparison table.

**Learn:**
- **Held-out perplexity** = exp(mean CE over non-masked tokens) on text none of the stages trained on. Lower = better LM, but it cannot see instruction-following quality — a model can have great ppl and ignore instructions.
- **Distinct-n** (diversity): unique n-grams / total n-grams over a set of generations. Detects degeneration/repetition that ppl misses. Hand example with tokens `["the","cat","sat","on","the","cat"]`: distinct-1 = 4 unique/6 total ≈ 0.667; distinct-2 = 4 unique of 5 bigrams = 0.8 (the bigram ("the","cat") repeats).
- **Template compliance**: a regex check that output is non-empty text terminated by `<|end|>` with no template-marker leakage (`### Instruction:` reappearing = failure). Cheap, automatic, and exactly the kind of programmatic check you'd run before an LLM-judge pass at work.
- **Why human/LLM judges exist:** every metric above is gameable and blind to *quality* — a compliant, diverse, low-ppl response can still be nonsense. Your LLM-judge pipelines at work exist because automatic metrics saturate; here, *you* are the judge (Day 14.4), and the same caveats (rubric drift, position bias) apply to you.
- **Generation config is part of the eval:** fix seed, temperature, top-p, max tokens once in a dataclass; every model evaluated under identical config or the comparison is void.

**Read (30–45 min):**
- Li et al., "A Diversity-Promoting Objective..." §4.1 (distinct-n definition): https://arxiv.org/abs/1510.03055
- DPO paper §6 evaluation discussion (what they measured and why win-rates).
- Python `re` module (you'll write the compliance regex): https://docs.python.org/3/library/re.html

**Build:**
1. Create `tlib/evalsuite.py`:
   ```python
   @dataclass(frozen=True)
   class GenConfig:
       temperature: float = 0.8; top_p: float = 0.95
       max_new_tokens: int = 120; seed: int = 14

   def perplexity(model: nn.Module, batches: Iterable[dict], ignore_index: int = -100) -> float:
       """exp of token-weighted mean CE over all batches (sum losses*counts / sum counts)."""

   def distinct_n(token_lists: list[list[str]], n: int) -> float:
       """unique n-grams / total n-grams, pooled across all generations.
       Returns 0.0 if no n-grams."""

   def template_compliance(texts: list[str]) -> float:
       """Fraction matching: at least one non-whitespace char, then '<|end|>',
       and no occurrence of '### Instruction:' or '### Response:' in the body."""

   def length_stats(token_lists: list[list[str]]) -> dict[str, float]:  # mean/min/max

   def run_evals(model: nn.Module, tokenizer, prompts: list[str],
                 ppl_batches: Iterable[dict], cfg: GenConfig) -> dict[str, float]:
       """Generate for all prompts under cfg; return {'ppl', 'distinct1',
       'distinct2', 'compliance', 'len_mean'}."""
   ```
2. `week14/run_eval_table.py`: 30 held-out template prompts (regenerate via `make_sft_data.py` with a different seed; assert no prompt string appears in the training jsonl) + a ppl set of 50 held-out raw stories tokenized with response-style labels. Evaluate all three checkpoints into `week14/EVAL.md`:
   ```text
   GenConfig: temperature=0.8 top_p=0.95 max_new_tokens=120 seed=14
   | model | ppl | distinct-1 | distinct-2 | compliance | mean len |
   | base  |     |            |            |            |          |
   | sft   |     |            |            |            |          |
   | dpo   |     |            |            |            |          |
   ```

**Verify — done when** (`tlib/tests/test_evalsuite.py`):
- `distinct_n([["the","cat","sat","on","the","cat"]], 1) == pytest.approx(4/6)` and `n=2 == pytest.approx(0.8)` — the hand example, exact.
- Compliance on crafted strings: `"A nice story.<|end|>"` → counted; `""`, `"   <|end|>"`, `"text"` (no end), `"ok ### Response: leak<|end|>"` → all rejected. Assert the four cases.
- `perplexity` against a manual computation on one tiny batch: `allclose(math.exp(manual_mean_ce))`.
- `EVAL.md` table complete: 3 models × 5 metrics, identical GenConfig recorded at the top. Expected *shape* (not guaranteed): base fails compliance (~0), SFT/DPO mostly comply; ppl on raw stories may *worsen* slightly after SFT/DPO — that's the alignment tax, note it rather than hiding it.

**If stuck:** your week-11 sampler (seed plumbing); Li et al. §4.1; check tokenizer round-trips `<|end|>` as one token (`encode→decode` identity test).

## Day 14.4 — Evaluation harness II: blind human preference (~2.5h)
- [ ] done

**Goal:** build a blinded A/B judging CLI and run a 20-trial SFT-vs-DPO comparison on yourself, honestly.

**Learn:**
- **Why blind:** you trained these models; you *want* DPO to win. Blinding (random left/right assignment, hidden until after all votes) is the only defense against your own bias — the same reason position bias is randomized in LLM-judge setups you run at work.
- **The binomial yardstick:** under "no real difference", each trial is a fair coin. With n=20 trials, P(≥15 wins | p=0.5) ≈ 0.021 — so the guideline: **20 trials, 15+ wins for one side ⇒ meaningful at p < 0.05 (one-sided)**. 11–14 wins out of 20 is noise; say "no detectable difference". Ties count as half a trial removed (report them separately).
- **Judge with a rubric, decided in advance** — write it into the tool's prompt text so future-you uses the same one:
  ```text
  Pick the response that is better OVERALL, weighing in order:
  1. Did it do what the instruction asked?
  2. Is it coherent (sentences make sense, no word salad, no loops)?
  3. Did it stop cleanly (no trailing off, no template leakage)?
  Vote: l = left better, r = right better, t = genuine tie.
  ```
- **The deliverable is the harness + the discipline**, not a DPO victory. At your model scale, DPO trained on synthetic corruptions may show no human-visible gain over SFT. A recorded null result from a sound protocol is a success.

**Read (30–45 min):**
- Sign test (the n=20/15 arithmetic is a binomial tail): https://en.wikipedia.org/wiki/Sign_test
- `argparse` docs (subcommands not needed; flags are): https://docs.python.org/3/library/argparse.html
- DPO paper §6.2 (their human-eval / win-rate protocol) for protocol flavor.

**Build:**
1. Create `week14/ab_judge.py`:
   ```python
   def build_trials(prompts: list[str], gens_a: list[str], gens_b: list[str], seed: int) -> list[dict]:
       """One dict per prompt: {'prompt', 'left', 'right', 'left_is_a': bool}
       with left/right assignment from random.Random(seed).random() < 0.5."""

   def main() -> None:
       """Flags: --ckpt-a --ckpt-b --prompts FILE --n 20 --seed 14 --out results.json
       1. Load both models + tokenizer; generate for the first n prompts under
          one GenConfig (reuse tlib.evalsuite.GenConfig).
       2. For each trial: print prompt, then 'LEFT:' and 'RIGHT:' outputs, the
          rubric, and read input() in {'l','r','t'}. NEVER print which model is which.
       3. After ALL votes: unblind, compute wins_a, wins_b, ties, win rate
          (ties excluded), and print the guideline verdict (>=15/20 rule).
       4. Write results.json: per-trial assignment + vote + the verdict."""
   ```
   Blinding rule in code: the `left_is_a` field is written to the results dict at build time but **never printed before voting completes**.
2. `week14/eval_prompts.txt`: 20 held-out instruction prompts (from Day 14.3's held-out pool).
3. Run it: `python week14/ab_judge.py --ckpt-a checkpoints/sft_final.pt --ckpt-b checkpoints/dpo_final.pt --prompts week14/eval_prompts.txt --n 20`. Judge all 20. Record the verdict and 3 sentences of qualitative observation in `week14/LOG.md` — whatever the result.

**Verify — done when** (`week14/test_ab_judge.py`):
- **Blinding randomized:** `build_trials` over 100 synthetic prompts with seed 14 yields both `left_is_a=True` and `False` (assert 20 ≤ count ≤ 80), and for every trial the `left` text equals `gens_a[i]` iff `left_is_a` — the stored assignment provably matches what's presented.
- Win-rate math: feed a fabricated vote list (e.g. 12 l, 6 r, 2 t with known assignments) through the tally function; assert exact wins/ties and that the verdict string applies the ≥15/20 rule correctly (parametrize: 15 wins → "significant", 14 → "not detectable").
- `results.json` from your real run exists with 20 trials, votes, assignments, and verdict; LOG.md entry written.

**If stuck:** Sign test article (the table of critical values); your Day-14.3 GenConfig — both models must share it; `random.Random` (instance, not module-level, so tests are reproducible).

## Day 14.5 — Capstone assembly: chat CLI + model card (~2.5h)
- [ ] done

**Goal:** wire your best checkpoint into an interactive chat CLI and document the entire system in a model card sufficient to reproduce it.

**Learn:**
- **Template application belongs in the interface, not the user's hands:** the CLI wraps every user turn in the SFT template, generates, streams tokens as decoded, and halts on `<|end|>` — the user never sees template markup. This is the chat-formatting layer every production LLM hides from you.
- **Stop-token mechanics:** check each sampled id against `end_id` *before* appending/printing; also enforce `max_new_tokens` as backstop. Multi-turn context: either concatenate prior turns (template per turn) up to the context limit, or stateless single-turn — pick one, document it (stateless is fine and honest for a tiny model).
- **Model cards** (Mitchell et al.): intended use, training data, eval results, *and limitations* — written for someone who didn't build it. Yours doubles as the reproduction script: every stage gets its exact command.
- **Bluntness is a feature:** "75-token context windows of TinyStories knowledge, synthetic preferences, will hallucinate freely outside children's-story domain" is a *good* limitations section.

**Read (30–45 min):**
- Mitchell et al., "Model Cards for Model Reporting" §3–4 (the section schema): https://arxiv.org/abs/1810.03993
- Your week-10 chat/generation CLI code — you're extending, not rewriting.

**Build:**
1. `week14/chat.py`: argparse `--ckpt` (default `checkpoints/dpo_final.pt` or whichever Day 14.4 favored — if null result, prefer DPO and say why in the card: tie goes to the later stage only if eval table doesn't object; otherwise SFT), `--temperature --top-p --max-new-tokens --seed`. REPL loop: read user line → apply `PROMPT_TEMPLATE` → stream tokens (print decoded pieces with `flush=True`) → stop on `end_id` or budget → print newline, repeat. `/quit` exits. Plus a non-interactive mode `--once "instruction"` for testing.
2. `MODEL_CARD.md` at repo root. Completion checklist — every box required:
   - [ ] Architecture table: layers / d_model / heads / context length / param count — printed from code (`sum(p.numel() ...)`), not recalled.
   - [ ] Tokenizer: type (BPE), vocab size, corpus it was trained on, special tokens including the `<|end|>` id.
   - [ ] Data per stage: pretrain corpus + token count; SFT jsonl recipe (the 6 templates) + example count; preference pairs recipes + count and mix ratio.
   - [ ] Training config per stage: lr, schedule, steps/epochs, batch size, clipping, seeds, and β for DPO.
   - [ ] Eval table pasted from `week14/EVAL.md` + the A/B verdict from `results.json`.
   - [ ] Limitations, blunt: domain (children's-story English), scale, synthetic preferences (mechanics, not human taste), single-turn, will hallucinate freely off-domain.
   - [ ] **Reproduction commands**: one fenced block per stage — env setup, pretrain (or "reuse `checkpoints/base_final.pt`, sha256 + producing command + git ref"), `make_sft_data.py`, `train_sft.py`, `make_pref_data.py`, `train_dpo.py`, `run_eval_table.py`, `ab_judge.py`, `chat.py` — each with full flags, copy-pasteable.

**Verify — done when:**
- `python week14/chat.py --once "Write a short story about a dog" --seed 14` produces output and **terminates via the stop token** (assert in a pytest that wraps `--once`: returned text contains no `<|end|>` literal — it's consumed as stop — and generation length < max_new_tokens for at least this prompt, demonstrating the stop fired; if it hits the budget, print a warning and test with a prompt that does stop).
- Interactive smoke test: 5-turn conversation runs without crash; streaming visibly streams (tokens appear incrementally).
- `MODEL_CARD.md` checklist: all 7 boxes ticked; a fresh shell can copy-paste any single command block and it parses (run each with `--help` or a dry-run flag where training is too slow to re-run today — Day 14.6 does the real pass).

**If stuck:** your week-11 KV-cache generate loop (the stop-check belongs inside it); Mitchell et al. §4.

## Day 14.6 — Deep build: close the loop — full reproduction, tag, final report (~3.5–4h)
- [ ] done

**Goal:** prove the system reproduces from MODEL_CARD.md alone, then freeze and report it.

**Learn:**
- **A reproduction pass is an integration test for your documentation.** You will find drift: a flag renamed since week 9, a hardcoded path, a seed set in one script but not another. Fixing these *is* the day's work.
- **Reuse vs re-run:** pretraining is the long pole — the card may legitimately say "reuse `base_final.pt`, produced by command X at commit Y"; everything downstream (SFT → DPO → evals → chat) must actually re-run today, into a separate `checkpoints/repro/` directory, and produce *comparable* results (same val-loss ballpark, same compliance behavior — exact bitwise equality is not the bar unless you've pinned every seed and thread count; note which you achieved).
- **Tagging:** a git tag is a permanent name for "the state that produced these numbers". Future-you, three months into a new project, will need it.

**Read (30 min):** git-tag docs: https://git-scm.com/docs/git-tag — plus your own MODEL_CARD.md, read top to bottom as if you'd never seen the repo.

**Build:**
1. Reproduction pass, driven **only** by MODEL_CARD.md commands (no memory, no other docs): data gen → SFT → DPO → eval table → one `chat.py --once` call. Every divergence: fix the script *or* the card (whichever is wrong), and log the fix.
2. Compare `checkpoints/repro/` eval table vs the original; add a "Reproduction" section to MODEL_CARD.md with both tables and one paragraph on deltas and their causes (seeds? data regen order? note honestly).
3. Freeze: `git init` if somehow never done; `.gitignore` checkpoints/data caches (keep `base_final.pt` out of git; record its sha256 in the card instead); commit everything; `git tag -a v1.0 -m "14-week curriculum capstone"`.
4. `FINAL_REPORT.md`, one page, this skeleton:
   ```markdown
   # Final Report — 14-week PyTorch curriculum
   ## What was built
   (1 paragraph: tlib inventory + base->SFT->DPO pipeline + eval harness)
   ## The 5 hardest bugs
   For each: symptom -> root cause -> the test in tlib/tests that now guards it.
   (Mine every weekly LOG.md; pick by hours lost, not by how clever the fix was.)
   ## Measured results
   - Pretrain: final train/val loss, tokens seen        (week 9 logs)
   - Classifier: head/full/random-control table          (week13 D2)
   - Bake-off: LoRA vs full table                        (week13 D6)
   - Eval suite: base/sft/dpo metric table               (week14 D3)
   - Blind A/B: n, wins, ties, verdict                   (week14 D4 results.json)
   ```
   Numbers from logs only; no reconstruction from memory.

**Verify — done when:**
- The downstream pipeline ran end-to-end from card commands with zero manual intervention beyond what the card states (the honest bar: if you had to deviate, the card now contains the deviation).
- `git tag` lists v1.0; `git status` clean; `git show v1.0 --stat` includes MODEL_CARD.md and FINAL_REPORT.md.
- FINAL_REPORT.md ≤ ~60 lines, all three sections, every number traceable to a log file or results.json in the repo.

**If stuck:** the failing script's own `--help`; your weekly LOG.md files; `git tag` docs above.

## Day 14.7 — Program retrospective (~2h)
- [ ] done

**Goal:** measure what stuck, name what didn't, and choose what's next. No new code; no quiz for this week — the whole program is the quiz.

**Do:**
1. **Re-take cold, on paper, the hardest 3 quiz sections from earlier weeks:** Week 3 (autograd/backprop quiz), Week 9 (training-dynamics/optimization quiz), Week 11 (attention/KV-cache quiz) — they're in those files with their answer keys. Score yourself honestly against the keys.
2. **Write `RETROSPECTIVE.md` — gap list:** every question you missed or hesitated on, plus anything from 14 weeks that still feels shaky (be specific: "I can write LoRA but couldn't re-derive why both-zero init stalls" beats "LoRA-ish stuff"). For each gap: the file/test in your own repo that exercises it.
3. **Pick a continuation path** (write your choice + first concrete step into RETROSPECTIVE.md):
   - **Scale up:** rent a GPU (single A100-class), rerun your pretrain at 10–20× tokens/params with your own DDP script; read first: the Chinchilla paper (https://arxiv.org/abs/2203.15556) for the compute/data trade.
   - **Go deeper:** kernels — Triton tutorials (https://triton-lang.org/main/getting-started/tutorials/index.html), then fuse your own attention; distributed beyond DDP — PyTorch FSDP docs (https://docs.pytorch.org/docs/stable/fsdp.html) and the Megatron-LM tensor-parallel paper (https://arxiv.org/abs/1909.08053) as a reading list.
   - **Go wider:** ViT on CIFAR-10 with your tlib (the encoder is your decoder minus the causal mask — measure how little changes); then DDPM diffusion from the paper (https://arxiv.org/abs/2006.11239) with your Trainer.
   - **Apply at work:** port this week's eval discipline into your legal-AI harness — blind A/B for prompt variants, compliance-style programmatic checks before judge calls, binomial yardsticks for win-rates, model-card-style prompt documentation.
4. **Read this and believe it:** *You no longer need a curriculum — you need a project.* Five seeds matched to your tlib:
   - **tlib-serve:** a batched inference server around your KV-cache generator (continuous batching is the interesting part).
   - **Tokenizer lab:** train BPE at 4 vocab sizes on the same corpus; measure downstream ppl and bytes/token; write up the trade-off curve.
   - **Distillation:** your SFT model as teacher, a half-size student, KL-on-logits loss in tlib; compare student-distilled vs student-from-scratch with your Week-14 harness.
   - **Long-context surgery:** extend your model's context 4× via position-interpolation finetuning; build the eval that proves retrieval at distance works (needle-style synthetic task, fully offline).
   - **Preference-data quality study:** vary Day-14.2's corruption recipes and mix ratios; measure which synthetic preferences actually move the Day-14.4 blind win-rate. (This one is your day job wearing a lab coat.)

**Verify — done when:** three old quizzes scored against their keys; RETROSPECTIVE.md contains the gap list (with repo pointers), the chosen path, and the first concrete step with a date on it.
