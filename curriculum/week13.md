# Week 13 — Finetuning: transfer, SFT & LoRA

You have a base LM you pretrained yourself (`checkpoints/base_final.pt`). This week you turn that raw next-token predictor into useful specialized models — the exact pipeline (pretrain → adapt) behind every product LLM you work with. You'll prove transfer learning works with a controlled experiment, build instruction tuning with correct loss masking from scratch, and implement LoRA from the paper. Everything runs on CPU; your model is small enough that full finetunes take minutes, which is precisely why this is the right scale to learn the mechanics.

**Week outcome:** by day 7 you have: `tlib/metrics.py`, `tlib/sft.py`, `tlib/lora.py` (all pytest-covered), a sentiment classifier proving pretrained features beat random features, an SFT'd instruction-following checkpoint `checkpoints/sft_final.pt`, a LoRA-vs-full-FT comparison table, and the skill of adapting any pretrained checkpoint surgically.

## Day 13.1 — Transfer learning mechanics & the classifier head (~2h)
- [ ] done

**Goal:** load your pretrained trunk into a new classification model with surgical state_dict control, and understand exactly what freezes, what trains, and why.

**Learn:**
- **Why pretrained features transfer:** pretraining forces the trunk to build representations of syntax, entities, and sentiment-bearing words just to predict the next token. A new task can reuse those representations instead of learning them from scratch on tiny data.
- **`load_state_dict(strict=False)` semantics:** it returns a NamedTuple of `missing_keys` (params your new model has but the checkpoint doesn't — fine when it's your new head, which stays randomly initialized) and `unexpected_keys` (params in the checkpoint your model lacks — fine when it's the old `lm_head` you removed). Always print and assert on both lists; "fine" means *you expected exactly those keys*.
- **Freezing:** `p.requires_grad_(False)` stops gradient computation for that param entirely — it never appears in `.grad`, and if you only pass trainable params to the optimizer, it's never updated. Frozen trunk + trainable head = "linear probe".
- **Discriminative learning rates:** `optimizer = AdamW([{"params": head, "lr": 1e-3}, {"params": trunk, "lr": 1e-5}])` — the head is random and needs big steps; the trunk is pretrained and big steps would destroy it.
- **Pooling a decoder for classification:** a causal LM's hidden state at position *t* only sees tokens ≤ *t*, so the **last non-pad token's** hidden state is the only one that has read the whole input. With right-padding and a pad mask, the exact recipe is:
  ```python
  h = self.base.hidden(input_ids)            # (B, T, d)
  lengths = pad_mask.sum(dim=1)              # (B,) number of real tokens
  pooled = h[torch.arange(h.size(0)), lengths - 1]   # (B, d)
  return self.head(pooled)                   # (B, num_classes)
  ```
  Pooling the literal last position `h[:, -1]` would pool a **pad token** for every sequence shorter than T — a classic silent bug that still trains, just worse.

**Read (30–45 min):**
- Yosinski et al., "How transferable are features in deep neural networks?" — abstract + §1, §4.1: https://arxiv.org/abs/1411.1792
- PyTorch `Module.load_state_dict` (the `strict` param and return value): https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.load_state_dict
- PyTorch per-parameter optimizer options: https://docs.pytorch.org/docs/stable/optim.html#per-parameter-options
- HF datasets loading guide (Hub + offline sections): https://huggingface.co/docs/datasets/loading

**Build:**
1. If your `DecoderLM` doesn't already expose pre-head hidden states, add `def hidden(self, input_ids: torch.Tensor) -> torch.Tensor: """Return final-layer hidden states (B, T, d) before lm_head."""` (refactor `forward` to call it, then apply `lm_head` — zero behavior change; your existing LM tests must still pass).
2. Create `week13/classify.py` with:
   ```python
   class LMClassifier(nn.Module):
       def __init__(self, base: DecoderLM, num_classes: int, freeze_trunk: bool = True) -> None:
           """Wrap base as trunk; add self.head = nn.Linear(d_model, num_classes).
           If freeze_trunk, set requires_grad_(False) on all trunk params."""
       def forward(self, input_ids: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
           """pad_mask: (B, T) bool, True on real tokens (right-padding).
           Pool last non-pad hidden state, return (B, num_classes) logits."""
   ```
3. Data: `from datasets import load_dataset; ds = load_dataset("glue", "sst2")` — sentence/label (0=neg, 1=pos). Sizes: 67,349 train / 872 validation; the official test split has hidden labels (−1) — never use it. Take `train[:8000]` for training, `train[8000:9000]` as your test set, official `validation` as val. ~5 MB, cached after first download (`HF_HUB_OFFLINE=1` works afterwards).
4. Tokenize with **your** BPETokenizer. Write `def collate(batch: list[dict], pad_id: int, max_len: int = 64) -> dict[str, torch.Tensor]` returning `input_ids` (B, T) right-padded, `pad_mask` (B, T) bool, `labels` (B,). Truncate to `max_len`.
5. Write `def load_trunk(clf: LMClassifier, ckpt_path: str) -> None` that loads `base_final.pt` into the trunk with `strict=False`, asserts `missing_keys` contains only head params and `unexpected_keys` only the old lm_head (if untied), and raises otherwise.

**Verify — done when** (`week13/test_classify.py`):
- Frozen-trunk determinism: `clf.eval()`; two forwards on the same batch are `torch.equal`.
- Optimizer contents: with `freeze_trunk=True`, `[p for p in clf.parameters() if p.requires_grad]` has exactly 2 tensors (head weight + bias); assert count and total numel `== d_model * 2 + 2`.
- State_dict surgery: snapshot `clf.head.weight.clone()` before `load_trunk`; after loading, head weight `torch.equal` to snapshot (untouched) while `clf.base.tok_emb.weight` now equals the checkpoint tensor (`allclose`).
- Pooling correctness: build a batch of two sequences, lengths 3 and 5, padded to 5; assert pooled rows equal `h[0, 2]` and `h[1, 4]` exactly.

**If stuck:** `load_state_dict` docs above; `torch.nn.Module.named_parameters` docs; the autograd notes on locally disabling gradients: https://docs.pytorch.org/docs/stable/notes/autograd.html

## Day 13.2 — Run the finetune, with a control and real metrics (~2.5h)
- [ ] done

**Goal:** run three classification finetunes — including the control that proves pretraining mattered — and evaluate them with metrics you implement and verify by hand.

**Learn:**
- **The control group:** "my finetuned model gets X% accuracy" means nothing alone. A frozen **randomly initialized** trunk + trained head is the control: if the pretrained trunk doesn't beat it, your pretraining transferred nothing. This is the experiment design instinct that separates evals you can trust from evals you can't — same logic as a no-op baseline in your work eval harnesses.
- **Precision / recall / F1:** accuracy lies under class imbalance. Per class *c*: precision = TP/(TP+FP) ("of my predicted *c*, how many were right"), recall = TP/(TP+FN) ("of true *c*, how many did I find"), F1 = harmonic mean. All derivable from a confusion matrix `cm[true, pred]`.
- **Split discipline:** tune hyperparameters on val only; touch test once, at the end, per model. Three runs, one test pass each.
- **Linear probe vs full FT:** head-only training is cheap and can't destroy the trunk; full FT is stronger but can overfit 8k examples fast — watch val accuracy per epoch.

**Read (30–45 min):**
- Howard & Ruder, ULMFiT — §3 (discriminative fine-tuning, gradual unfreezing): https://arxiv.org/abs/1801.06146
- Wikipedia "Precision and recall" (definition tables): https://en.wikipedia.org/wiki/Precision_and_recall
- Skim your own `tlib` Trainer code: you'll reuse it; decide which callbacks you need.

**Build:**
1. Create `tlib/metrics.py`:
   ```python
   def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
       """(N,) int preds/targets -> (C, C) int64 matrix, rows=true, cols=pred."""
   def precision_recall_f1(cm: torch.Tensor) -> dict[str, torch.Tensor]:
       """Per-class 'precision','recall','f1' (C,) plus scalar 'macro_f1','accuracy'.
       Zero-denominator convention: define the metric as 0.0 (document it)."""
   ```
2. Create `week13/train_classify.py` (argparse: `--mode {head,full,random-control}`, `--epochs`, `--lr`). head: frozen pretrained trunk, lr 1e-3, 3 epochs. full: nothing frozen, param groups (trunk 1e-5 / head 1e-3), 2 epochs. random-control: freshly initialized trunk (same config, no checkpoint load), frozen, head lr 1e-3, 3 epochs. Reuse your Trainer; log val accuracy per epoch.
3. Run all three; write a results table (mode, trainable params, val acc, test acc, test macro-F1) to `week13/LOG.md`.

**Verify — done when:**
- Metrics exactness (`tlib/tests/test_metrics.py`): for `cm = torch.tensor([[4, 1], [2, 3]])` assert class-1 precision `== 3/4`, recall `== 3/5`, F1 `allclose 2/3`; accuracy `== 7/10`. Also assert `confusion_matrix` reproduces this cm from raw pred/target vectors you write out by hand (10 elements).
- The week's key evidence: **pretrained frozen trunk val accuracy > random frozen trunk val accuracy** (assert in a script or note in LOG.md with numbers). This is near-guaranteed; random-trunk linear probes on SST-2 hover near majority class (~51%). If it fails, your checkpoint didn't load — re-check Day 13.1's surgery test before blaming the model.
- All three rows of the table filled from actual runs; test set touched exactly once per mode.

**If stuck:** ULMFiT §3; your week-8 Trainer tests for callback wiring; HF datasets loading guide (above) for split slicing syntax.

## Day 13.3 — Instruction SFT: data format, special tokens, loss masking (~2.5h)
- [ ] done

**Goal:** build the SFT dataset pipeline whose core mechanic — loss masking with ignore_index — you can verify exactly.

**Learn:**
- **SFT = supervised finetuning on (instruction, response) pairs.** The model still does next-token prediction; only the data changes. A fixed prompt template makes the format learnable. Yours: `"### Instruction:\n{instruction}\n\n### Response:\n"` followed by the response and `<|end|>`.
- **Why mask the prompt:** you want the model to learn to *produce responses*, not to reproduce instructions. Set prompt-position labels to `ignore_index=-100` so they contribute exactly zero loss and zero gradient.
- **The shift, precisely:** `labels` is a copy of `input_ids` with prompt positions set to −100. Loss compares `logits[:, :-1]` against `labels[:, 1:]`: position *t*'s logits are scored against token *t+1*. **Worked example** — prompt tokenizes to `[12, 7]`, response to `[17, 3]`, `<|end|>` is id 2:
  ```text
  position t:        0     1     2     3     4
  input_ids:        12     7    17     3     2
  labels:         -100  -100    17     3     2
  shifted target  -100    17     3     2     (none)   <- labels[t+1]
  contributes?      no   YES   YES   YES     n/a
  ```
  Exactly 3 positions contribute: `loss = -(log p(17|12,7) + log p(3|12,7,17) + log p(2|12,7,17,3)) / 3`. Note the elegance: position 1 (the *last prompt token*) carries the first response target — the model learns to start responding right where the prompt ends. Write this table out yourself before coding.
- **Extending the vocab:** `<|end|>` needs a fresh token id (`old_vocab_size`), a new embedding row, and — if your lm_head is tied to the embedding — the head grows automatically *only if you re-tie after resizing*. Replacing the embedding module silently breaks tying; you must reassign `lm_head.weight = tok_emb.weight` afterward.
- **Padding ≠ masking confusion:** padded positions in the collated batch must *also* get label −100, or you train the model to emit pad tokens.

**Read (30–45 min):**
- InstructGPT paper §3.5 (SFT) + Figure 2 (pipeline overview): https://arxiv.org/abs/2203.02155
- PyTorch `CrossEntropyLoss` — the `ignore_index` parameter semantics (ignored targets don't contribute to loss *or* the mean's denominator): https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Browse databricks/databricks-dolly-15k to see real instruction-data shape: https://huggingface.co/datasets/databricks/databricks-dolly-15k

**Build:**
1. **Data — use template-generated instructions matched to your base model's domain** (recommended: your tiny model knows TinyStories-world, not world knowledge; Dolly would ask it things it cannot know). Create `week13/make_sft_data.py`: load 2,000 TinyStories (or wikitext paragraphs) and emit `week13/sft_data.jsonl` of `{"instruction": ..., "response": ...}` using ~6 templates: (a) "Continue this story: {first 2 sentences}" → rest of story; (b) "Write a short story about {noun}" (extract a noun from the story with a small stopword-filtered frequency pick) → full story; (c) "Write a story that ends with the word '{last word}'" → full story; (d) "Finish this sentence: {sentence minus last 5 words}" → the 5 words; (e) "Write a story using the words {w1}, {w2}, {w3}" → full story; (f) "Repeat this sentence exactly: {sentence}" → sentence (a free format-following probe). Cap responses at ~150 tokens.
2. Upgrade your `tlib` cross-entropy: add `ignore_index: int = -100`. Implementation: flatten, build `valid = targets != ignore_index`, compute your stable log-softmax CE on valid positions only, mean over `valid.sum()` (guard: if zero valid positions, return `0.0 * logits.sum()` to keep the graph).
3. Create `tlib/sft.py`:
   ```python
   PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"

   def add_end_token(tokenizer: BPETokenizer, model: DecoderLM) -> int:
       """Append '<|end|>' to vocab; grow tok_emb by one row (old rows copied,
       new row ~ N(0, 0.02)); re-tie lm_head if tied. Return new token id."""

   class SFTDataset(Dataset):
       def __init__(self, path: str, tokenizer: BPETokenizer, end_id: int, max_len: int = 256) -> None: ...
       def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
           """input_ids = enc(template) + enc(response) + [end_id], truncated;
           labels = copy with prompt positions = -100."""

   def sft_collate(batch: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
       """Right-pad input_ids with pad_id and labels with -100."""
   ```

**Verify — done when** (`tlib/tests/test_sft.py`):
- **Exact masking test:** build one sample by hand with known prompt length *k* and known response ids; run your full loss path; separately compute CE manually using only logits at positions `k-1 .. len-2` vs the response ids; assert `allclose(loss, manual, atol=1e-6)`. Also assert: zeroing out (replacing with garbage) the logits at *masked* positions changes the loss by exactly 0.
- `ignore_index` upgrade: against `F.cross_entropy(..., ignore_index=-100)` on random logits with some −100 targets, `allclose`.
- Embedding resize: snapshot `tok_emb.weight[:old_vocab].clone()` before `add_end_token`; after, old rows `torch.equal`, shape grew by 1, and (if tied) `model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()`.
- Dataset sanity: for 5 random samples, `(labels != -100).sum() == response_len + 1` (the +1 is `<|end|>`).

**If stuck:** CrossEntropyLoss docs above (read the ignore_index reduction note twice); your week-5 stable-CE tests; InstructGPT §3.5.

## Day 13.4 — Run SFT with a before/after protocol (~2h)
- [ ] done

**Goal:** finetune the base model on your instruction set and capture honest qualitative evidence of behavior change.

**Learn:**
- **Config intuition for SFT:** lr in the 1e-5–1e-4 range (you're sculpting, not learning from scratch — pretraining used ~10× higher), 1–3 epochs, cosine or constant-with-warmup schedule. On 2k examples your model will overfit within a few epochs: val loss rising while train loss falls is *expected*; checkpoint at best val.
- **Before/after discipline:** qualitative claims need fixed conditions — same prompts, same seed, same sampling params, captured *before* training starts. Otherwise you'll see what you want to see. This is the offline analogue of pinning eval inputs at work.
- **What success looks like at this scale:** the model learns the *format* (respond after the template, stop at `<|end|>`) reliably; instruction *semantics* (using the requested words) only partially. Format acquisition is the measurable claim.

**Read (30–45 min):**
- InstructGPT §3.5 again + Appendix C.1 (SFT hyperparameters — note epochs and lr scale relative to model size).
- Your own week-9 LOG.md pretraining config — pick SFT lr ≈ pretrain lr / 10 as a starting point.

**Build:**
1. `week13/prompt_suite.txt` — 10 fixed prompts, one per line (instruction text only; the script applies the template). Use exactly these:
   ```text
   Continue this story: Once upon a time there was a little bird who could not fly.
   Continue this story: Tom found a big red box in the garden.
   Continue this story: Lily and her mom went to the park on a sunny day.
   Write a short story about a dog.
   Write a short story about the moon.
   Finish this sentence: The cat climbed up the tree because
   Finish this sentence: Sara was very happy when she saw
   Write a story using the words cat, river, hat.
   Repeat this sentence exactly: The sky is blue.
   What is the capital of France?
   ```
   The last one is deliberately out-of-distribution — to observe failure honestly, not to pass.
2. `week13/generate_suite.py`: loads a checkpoint + tokenizer, applies the template to each suite prompt, generates with your KV-cache sampler (temperature 0.8, top-p 0.95, `torch.manual_seed(13)`, max 120 new tokens, stop on `<|end|>`), writes a markdown section per prompt. Run it on `base_final.pt` **first** → the "before" half of `week13/before_after.md`.
3. `week13/train_sft.py`: load base, `add_end_token`, SFTDataset (90/10 train/val split of the jsonl), your Trainer with grad clipping 1.0, lr 5e-5, 3 epochs, save best-val checkpoint to `checkpoints/sft_final.pt` and the val-loss curve to `week13/sft_val_loss.png` (matplotlib).
4. Run `generate_suite.py` on `sft_final.pt` → the "after" half. Read both halves side by side; annotate each prompt pass/fail.

**Verify — done when:**
- **Format-following check:** define "follows" = output contains non-empty text then emits `<|end|>` within 120 tokens. SFT model follows on ≥ 5/10 suite prompts (count them; record the count in LOG.md — if under 5, train another epoch or raise lr one notch, and log that you did).
- The base ("before") model follows on ~0/10 — it has never seen the template (this contrast is your evidence).
- `sft_val_loss.png` exists and val loss decreased from epoch 0 before any rise.
- `before_after.md` has all 10 prompts × both models.

**If stuck:** your week-11 generation/KV-cache tests; InstructGPT Appendix C.1; re-check that `<|end|>` got id `old_vocab_size` in both training and generation (a mismatch here is the most common "never stops" bug).

## Day 13.5 — LoRA from the paper (~2.5h)
- [ ] done

**Goal:** implement LoRA exactly as specified in the paper, with the step-0-identity property proven by an exact test.

**Learn:**
- **The low-rank hypothesis:** finetuning changes weights by ΔW that empirically has low "intrinsic rank". So freeze W and learn ΔW = B·A with B ∈ ℝ^{d_out×r}, A ∈ ℝ^{r×d_in}, r ≪ d. Forward: `h = Wx + (α/r)·B(Ax)`.
- **Where to apply:** the paper (§4.2, Table 5) finds adapting attention's **q and v projections** the best param-for-param deal. You'll target those.
- **Why B = 0 at init:** with B zero, ΔW = BA = 0, so the adapted model is *exactly* the base model at step 0 — training starts from the pretrained function, not a perturbed one. A is Gaussian so gradients reach B immediately. (Both zero would give zero gradient to both — convince yourself via the product rule.)
- **Parameter math:** per adapted Linear, trainable params = r·(d_in + d_out). For q,v (each d×d) across L layers: **4·L·r·d**. Compute it for your model at r=8 and divide by total params — that fraction is the story of LoRA.
- **Merge at inference:** W′ = W + (α/r)·B·A is a plain Linear again — zero latency overhead, which is why LoRA beats adapter layers in deployment.

**Read (30–45 min):**
- LoRA paper §1–4.2 + Table 2/3 (GPT-2 results) + Table 5 (which matrices) + §7: https://arxiv.org/abs/2106.09685
- PyTorch `Module.named_modules` (for the surgery) and `setattr`-based module replacement: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_modules

**Build:**
1. Create `tlib/lora.py`:
   ```python
   class LoRALinear(nn.Module):
       def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
           """Freeze base (requires_grad_(False)). A: (r, in_features) ~ N(0, 0.02);
           B: (out_features, r) zeros. Both nn.Parameter."""
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """base(x) + (alpha / r) * F.linear(F.linear(x, A), B)"""

   def apply_lora(model: nn.Module, target_names: tuple[str, ...], r: int, alpha: float) -> list[str]:
       """Replace every nn.Linear whose qualified name ends with a target_name
       (e.g. ('q_proj', 'v_proj')) using named_modules + setattr on the parent.
       Freeze ALL other model params. Return list of replaced names."""

   def merge_lora(model: nn.Module) -> None:
       """In-place: for each LoRALinear, base.weight += (alpha/r) * B @ A;
       swap the LoRALinear back to the bare base Linear."""

   def lora_trainable_params(model: nn.Module) -> list[nn.Parameter]: ...
   ```
   Note the surgery gotcha: you can't `setattr` while iterating `named_modules()`; collect `(parent, attr_name, module)` first, then replace.
2. Compute and record in LOG.md: your model's d_model, n_layers, total params, LoRA trainable at r=8 via 4·L·r·d, and the percentage. (Example shape: d=512, L=8, r=8 → 131,072 trainable; for a 25M model that's ~0.5%.)

**Verify — done when** (`tlib/tests/test_lora.py`):
- **Step-0 identity (exact):** logits on a fixed batch before `apply_lora` vs after are `torch.equal` — not allclose; B=0 makes the delta exactly zero (the (α/r)·B(Ax) term multiplies a zero matrix).
- **Param count:** `sum(p.numel() for p in lora_trainable_params(model)) == 4 * L * r * d` exactly, and every returned param has `requires_grad`.
- **Gradient isolation:** run a forward+backward on a tiny batch; assert every `LoRALinear.base.weight.grad is None` and every A and B has a non-None grad (A's grad non-None even though B=0 — B's grad is what's nonzero first; check both `.grad is not None`).
- **Merge equivalence:** deepcopy the adapted model, train it 3 steps so A,B are nonzero, then `merge_lora` on one copy; merged-model logits `allclose` unmerged-adapter logits (atol 1e-5), and the merged model contains zero `LoRALinear` instances.

**If stuck:** LoRA paper §4.1 (the exact init and scaling are stated there); `named_modules` docs; your week-4 module-registration notes (parameters must be `nn.Parameter` to be found).

## Day 13.6 — Deep build: LoRA vs full-FT bake-off (~3.5–4h)
- [ ] done

**Goal:** run a controlled three-way comparison — full FT vs LoRA r=8 vs LoRA r=64 — on identical SFT data and budget, and report it honestly.

**Learn:**
- **Controlled comparison:** identical data, identical steps, identical seed, identical eval. Only the trainable-parameter structure varies. One knob per experiment.
- **Measuring memory:** GPU: `torch.cuda.reset_peak_memory_stats()` then `torch.cuda.max_memory_allocated()` after an epoch. CPU: sample `psutil.Process().memory_info().rss` (or `resource.getrusage(RUSAGE_SELF).ru_maxrss`) each step and keep the max. LoRA's saving is mostly *optimizer state* (AdamW keeps 2 floats per trainable param) — predict the saving from param counts before measuring, then check.
- **lr for LoRA:** rule of thumb is ~10× the full-FT lr (only small matrices move; the paper uses larger lr too). Use 5e-4 for LoRA vs your 5e-5 full-FT.
- **What r buys:** r=64 has 8× the params of r=8 but often barely better loss — rank is rarely the bottleneck at small scale. Whatever you observe, report it.

**Read (30–45 min):** LoRA paper §5 + Table 6 (effect of r); `torch.cuda.max_memory_allocated`: https://docs.pytorch.org/docs/stable/generated/torch.cuda.max_memory_allocated.html

**Build:**
1. `week13/bakeoff.py` (argparse `--variant {full,lora8,lora64}`): loads `base_final.pt` + `add_end_token`, applies variant (full: all params, lr 5e-5; lora8/64: `apply_lora(model, ("q_proj","v_proj"), r, alpha=2*r)`, lr 5e-4 — note: with the new `<|end|>` row, also leave `tok_emb` trainable in LoRA runs or freeze it and accept the untrained row; pick one, document it, use it for both LoRA runs), trains 2 epochs on the Day-13.3 data with fixed seed 13, identical batch size/clipping/schedule, records: final val loss, trainable params, peak memory, wall time. Saves `checkpoints/bakeoff_{variant}.pt` (for LoRA: merge before saving so all three load identically).
2. Run all three variants. Run `generate_suite.py` on each → `week13/bakeoff_outputs.md`.
3. Fill this table in `week13/LOG.md` from the recorded runs:
   ```text
   | variant | trainable params | final val loss | peak mem (MB) | wall time (s) |
   | full    |                  |                |               |               |
   | lora8   |                  |                |               |               |
   | lora64  |                  |                |               |               |
   ```
4. Below it, a 12-line analysis: val-loss gap LoRA vs full, params ratio, memory delta vs your optimizer-state prediction, time delta, qualitative differences in `bakeoff_outputs.md`, and one sentence on whether the paper's "LoRA gets close at a fraction of params" held at your scale — **report what actually happened**, including "no measurable difference" if so.

**Verify — done when:**
- The table has all 5 columns × 3 rows from real runs; no cell estimated.
- All three checkpoints load into a plain `DecoderLM` (post-merge) and generate from the suite without error.
- Trainable-param cells: full == total model params; lora8/lora64 match the closed-form (+ tok_emb if you chose trainable) — assert in the script, not by eye.
- LOG.md analysis written (12 lines, honest).

**If stuck:** LoRA paper Table 6; your Day-13.5 merge test; `resource` module docs for `ru_maxrss` units (KB on Linux).

## Day 13.7 — Review, quiz, redo-cold (~1.5h)
- [ ] done

**Goal:** consolidate the week; prove you can reproduce the two core mechanics (masking, LoRA) from memory.

**Self-quiz** (write answers, then check the Answers section at the bottom of this file):
1. After `load_state_dict(ckpt, strict=False)`, what's in `missing_keys` vs `unexpected_keys`, and which one is fine when adding a new head?
2. Why pool the *last non-pad* token's hidden state for a decoder classifier, and what exactly goes wrong if you pool position T−1 of a right-padded batch?
3. What did the frozen-random-trunk control prove, and what would it mean if it matched the pretrained trunk?
4. From cm = [[4,1],[2,3]] (rows=true): class-1 precision, recall, F1?
5. Prompt tokens [9, 4, 4], response [7], end id 2. Write `input_ids` and `labels`, and list which logit positions contribute to the loss after the shift.
6. Why must padded positions get label −100 even though pad is "just another token"?
7. After growing the embedding by one row for `<|end|>`, what breaks if lm_head was tied and you forget to re-tie?
8. LoRA: why B=0 and A Gaussian, rather than the reverse or both zero?
9. Trainable LoRA params for d=768, L=12, r=4, targets q+v — compute it.
10. Why does merging LoRA give zero inference overhead, and what do you lose by merging?

**Redo cold (no looking at your code):**
- Given prompt length k=4 and response ids [5, 8, 2(end)] in a length-7 unpadded sequence, write the exact labels vector and the contributing (logit position → target) pairs. Check against your Day-13.3 test.
- Re-derive the LoRA param formula r·(d_in+d_out) and the 4·L·r·d total for q,v from scratch; verify against your Day-13.5 assert.
- Write `LoRALinear.forward` and its init from memory in a scratch file; diff against `tlib/lora.py`.
- State from memory which keys you expect in missing/unexpected when loading base→classifier; check against your Day-13.1 assert.

---

## Answers (Day 13.7)
1. `missing_keys`: params the model has but the checkpoint lacks (your new head — fine, stays randomly init). `unexpected_keys`: checkpoint params the model lacks (old lm_head — fine, intentionally dropped). Both fine *only* when they contain exactly the keys you expect; assert it.
2. Causal attention means only the last real token's hidden state has attended to the full input. Pooling index T−1 pools a **pad token's** representation for any sequence shorter than T — the model classifies padding.
3. It isolates the contribution of pretrained features: same architecture, same head training, only the trunk weights differ. If random matched pretrained, the trunk features add nothing — i.e. the head alone (≈ bag of embeddings through a random projection) explains the score, and pretraining transferred nothing useful.
4. precision = 3/(3+1) = 0.75; recall = 3/(3+2) = 0.6; F1 = 2·0.75·0.6/1.35 = 2/3 ≈ 0.667.
5. `input_ids = [9, 4, 4, 7, 2]`, `labels = [-100, -100, -100, 7, 2]`. Contributing: logits at position 2 → 7, position 3 → 2. (Position t is scored against labels[t+1].)
6. Otherwise the loss teaches the model to emit pad after `<|end|>` — wasted capacity, and worse, pad becomes a high-probability continuation that pollutes generation.
7. The lm_head keeps pointing at the *old* embedding tensor: output dim stays old_vocab, so the model can never emit `<|end|>`, and embedding gradients no longer flow through the head. Re-tie by reassigning the Parameter after resize.
8. B=0 ⇒ ΔW = BA = 0 ⇒ adapted model is exactly the base at step 0 (training starts from the pretrained function). A Gaussian ⇒ ∂loss/∂B = grad·(Ax)ᵀ is nonzero, so learning starts immediately. Both zero ⇒ both gradients vanish (each grad is a product containing the other zero matrix) — stuck at zero.
9. 4·L·r·d = 4·12·4·768 = 147,456.
10. Merged, W′ = W + (α/r)BA is a single ordinary matmul — same FLOPs/latency as the base model. You lose the ability to cheaply detach/swap/stack adapters and to train further in low-rank form (you can re-extract by subtracting W, but you stored full-rank).
