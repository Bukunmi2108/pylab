# Week 5 — Data pipelines & the training loop

You can now build and optimize a network from raw tensors; what you cannot yet do is feed it data professionally or run experiments you trust. This week you learn what `DataLoader` actually does under the hood (sampler → batch sampler → fetch → collate), then build the infrastructure every later week runs on: a reusable `Trainer` with checkpointing, gradient accumulation, and callbacks; a reproducibility/config system; and an LR finder. This is sequenced here because Weeks 6–9 (normalization debugging, convnets, transformers) all assume you can launch a clean, seeded, logged experiment in one command.

**Week outcome:** `tlib/data.py` (CSVDataset, pad collate, weighted + bucket samplers), `tlib/transforms.py`, `tlib/trainer.py` (Trainer + callbacks), `tlib/config.py`, `tlib/lr_finder.py`, all pytest-covered; a 4-run FashionMNIST experiment grid with `results.csv` and an LR-finder plot. Skill: you can explain every stage between "file on disk" and "batch on device", and reproduce any run bit-for-bit.

---

## Day 5.1 — Dataset & DataLoader internals (~2h)
- [ ] done
**Goal:** Understand the full path an example travels from `__getitem__` to a collated batch, and build a custom dataset + collate function.

**Learn:**
- **Map-style vs iterable-style datasets.** A map-style dataset implements `__getitem__(i)` and `__len__` — random access by index, which is what samplers need. An iterable-style dataset implements `__iter__` and is for streams (logs, sharded files) where random access is impossible; sampling and sharding become *your* problem.
- **What DataLoader actually does.** Each epoch: a `Sampler` yields indices → a `BatchSampler` groups them into lists of `batch_size` → a fetcher calls `dataset[i]` for each index → `collate_fn` merges the list of samples into batched tensors. Every stage is replaceable; `DataLoader` is mostly glue.
- **`default_collate`.** Recursively stacks: tensors → `torch.stack` (adds a batch dim, so all samples must be same shape), numbers → tensor, dicts/tuples → collated per-key/per-position. Variable-length samples crash it — that's why custom `collate_fn` exists.
- **`num_workers`.** Worker *processes* each get a copy of the dataset and prefetch batches. They help when `__getitem__` is slow (disk I/O, decoding, augmentation); they hurt for tiny in-memory tensors (IPC overhead dominates). WSL2 note: process startup and shared memory are slower than native Linux — start with `num_workers=0` and try 2 only if loading is the measured bottleneck.
- **`pin_memory=True`** allocates batch tensors in page-locked RAM so CPU→GPU copies can be async (`non_blocking=True`). Pure CPU training: it does nothing useful; leave it off.

**Read (30–45 min):**
- torch.utils.data docs — read the prose top-to-bottom: *Dataset Types*, *Data Loading Order and Sampler*, *Loading Batched and Non-Batched Data* (incl. `collate_fn`), *Single- and Multi-process Data Loading*, *Memory Pinning*: https://docs.pytorch.org/docs/stable/data.html
- Tutorial "Datasets & DataLoaders" (skim — you know most of it; focus on the custom-dataset class): https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html

**Build:**
1. Create `tlib/data.py`. First a generator (so tests are self-contained):
   ```python
   def make_synthetic_csv(path: str, n_rows: int = 256, n_features: int = 4, seed: int = 0) -> None:
       """Write a CSV with header f0,...,f{n-1},target. Features ~ N(0,1) floats,
       target = int(sum of features > 0). Use csv module or numpy.savetxt."""
   ```
2. ```python
   class CSVDataset(torch.utils.data.Dataset):
       def __init__(self, path: str, feature_cols: list[str], target_col: str) -> None:
           """Load whole CSV into memory at init (it's small). Store features as
           float32 tensor (N, F) and targets as int64 tensor (N,)."""
       def __len__(self) -> int: ...
       def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
           """Returns (features (F,) float32, target () int64 scalar)."""
   ```
3. A padding collate for variable-length integer sequences:
   ```python
   def pad_collate(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
       """batch: list of B 1-D int64 tensors with lengths L_1..L_B.
       Returns:
         padded:  (B, L_max) int64, zero-padded on the right
         lengths: (B,) int64, original lengths
         mask:    (B, L_max) bool, True at real tokens, False at padding
       Example: [tensor([1,2,3]), tensor([4])] ->
         padded=[[1,2,3],[4,0,0]], lengths=[3,1], mask=[[T,T,T],[T,F,F]]"""
   ```
4. `tests/test_data.py`: (a) build CSV in `tmp_path`, check `len(ds)`, dtypes, shapes, and that `ds[0]` matches row 0 of the file; (b) `pad_collate` on the docstring example asserts exact tensors; (c) wrap dataset of random-length sequences in a `DataLoader(collate_fn=pad_collate, batch_size=8)` and assert every batch satisfies `mask.sum(1) == lengths` and `padded[~mask].eq(0).all()`.

**Verify — done when:**
- `pytest tests/test_data.py` passes.
- `default_collate` sanity: `next(iter(DataLoader(ds, batch_size=4)))` on CSVDataset gives shapes `(4, F)` and `(4,)` — assert it.
- You can write the four DataLoader stages (sampler → batch_sampler → fetch → collate) from memory.

**If stuck:** the *Loading Batched and Non-Batched Data* section of https://docs.pytorch.org/docs/stable/data.html spells out collate semantics; the fetcher logic lives in `torch/utils/data/_utils/fetch.py` on GitHub (https://github.com/pytorch/pytorch/tree/main/torch/utils/data/_utils) — it's ~30 lines and very readable.

---

## Day 5.2 — Samplers: weighted sampling & length bucketing (~2.5h)
- [ ] done
**Goal:** Replace DataLoader's index-generation stage with your own samplers for class imbalance and padding efficiency.

**Learn:**
- **The Sampler contract.** A `Sampler` is just an iterable of dataset indices with (usually) `__len__`. `SequentialSampler` yields `0..N-1`; `RandomSampler` a permutation. `DataLoader(shuffle=True)` is literally "use RandomSampler".
- **BatchSampler** wraps a sampler and yields *lists* of indices. Pass `batch_sampler=` to DataLoader and `batch_size/shuffle/sampler/drop_last` must all be left at defaults — you've taken over batching entirely.
- **Weighted sampling** fixes class imbalance at the data layer: sample index `i` with probability ∝ `weights[i]` *with replacement*, so rare classes appear as often as you choose without duplicating data on disk.
- **Length bucketing.** With `pad_collate`, a batch costs `B × L_max` cells; mixing length-3 and length-100 sequences wastes most of the compute on padding. Sorting/grouping similar lengths into the same batch cuts the waste, at a small cost in shuffling purity.

**Read (30–45 min):**
- *Data Loading Order and Sampler* section: https://docs.pytorch.org/docs/stable/data.html
- Sampler source — read `RandomSampler`, `WeightedRandomSampler`, `BatchSampler` (each is ~20 lines): https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py

**Build (in `tlib/data.py`):**
1. ```python
   class MyWeightedSampler(torch.utils.data.Sampler[int]):
       def __init__(self, weights: torch.Tensor, num_samples: int,
                    generator: torch.Generator | None = None) -> None:
           """weights: (N,) nonnegative per-index weights (need not sum to 1)."""
       def __iter__(self): ...   # yields num_samples indices, with replacement
       def __len__(self) -> int: ...
   ```
   Internals: do NOT call `torch.multinomial` or torch's `WeightedRandomSampler`. Use the inverse-CDF multinomial you built in Week 2 (`tlib/utils.py`); if you dropped it, re-derive: `cdf = (weights / weights.sum()).cumsum(0)`, then `torch.searchsorted(cdf, torch.rand(num_samples, generator=generator))`.
2. ```python
   class BucketBatchSampler(torch.utils.data.Sampler[list[int]]):
       def __init__(self, lengths: list[int], batch_size: int,
                    bucket_size: int = 100, seed: int = 0) -> None:
           """Shuffle indices; cut into buckets of `bucket_size`; sort each bucket
           by length; cut sorted buckets into batches of batch_size; shuffle the
           order of the batches. Yields lists of indices."""
   ```
3. A waste metric in `tlib/data.py`:
   ```python
   def padding_waste(batches: list[list[int]], lengths: list[int]) -> float:
       """Fraction of padded cells: sum over batches of (B*Lmax - sum(lens)) / total cells."""
   ```
4. `tests/test_samplers.py`.

**Verify — done when:**
- Weighted: 3-class dataset with class counts 800/150/50; set `weights[i] = 1/count[class_of[i]]`; draw 10,000 indices; empirical class frequencies are each within ±0.03 of 1/3 (seeded, this is comfortably satisfied).
- Bucketing: 2,000 fake lengths uniform in [5, 100]; compute `padding_waste` for `BucketBatchSampler(batch_size=32, bucket_size=320)` vs random batching of the same size; assert `waste_bucket / waste_random < 0.5` (typically ~0.1–0.3 vs ~0.3–0.4; the ratio < 0.5 is robust under any seed at these sizes — if it fails, your buckets aren't sorted).
- `DataLoader(ds, batch_sampler=BucketBatchSampler(...), collate_fn=pad_collate)` iterates without error and covers every index exactly once per epoch (assert sorted concatenation of yielded indices == `range(N)`).

**If stuck:** torch's `BatchSampler.__iter__` in sampler.py (link above) shows the chunking idiom; `WeightedRandomSampler` shows the expected constructor semantics you're mirroring.

---

## Day 5.3 — Transforms & normalization (~2h)
- [ ] done
**Goal:** Build your own transform pipeline and compute dataset statistics correctly, without leaking test data.

**Learn:**
- **Transforms are just callables.** A pipeline is function composition: `Compose([f, g])(x) = g(f(x))`. torchvision's versions are conveniences, not magic — yours will be drop-in equivalent for tensors.
- **Dataset mean/std, done right.** Compute per-channel statistics over the *training set only*, then apply the same numbers to val/test. Using test data to compute normalization stats is data leakage: your "unseen" data has influenced preprocessing.
- **Streaming computation.** For datasets too big for one tensor, accumulate `sum(x)` and `sum(x²)` over batches; `mean = s1/n`, `std = sqrt(s2/n − mean²)`. For FashionMNIST it also fits in memory — do it both ways as a cross-check.
- **Augmentation vs eval transforms.** Random augmentation (flips, crops) is train-time-only regularization: it expands the effective dataset. Eval transforms must be deterministic, or your validation metric becomes a random variable.

**Read (30–45 min):**
- torchvision transforms overview (read "Transforms v2" intro + Normalize/RandomHorizontalFlip API): https://docs.pytorch.org/vision/stable/transforms.html
- d2l.ai 14.1 *Image Augmentation* (concepts; ignore the framework-switcher noise): https://d2l.ai/chapter_computer-vision/image-augmentation.html

**Build — `tlib/transforms.py`:**
1. ```python
   class Compose:
       def __init__(self, transforms: list[Callable]) -> None: ...
       def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

   class Normalize:
       def __init__(self, mean: Sequence[float], std: Sequence[float]) -> None:
           """Per-channel: (x - mean[c]) / std[c]. x is (C, H, W) float."""
       def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

   class RandomHorizontalFlip:
       def __init__(self, p: float = 0.5) -> None: ...
       def __call__(self, x: torch.Tensor) -> torch.Tensor:
           """With prob p, flip the last (width) dim: x.flip(-1)."""
   ```
2. Script `scripts/fmnist_stats.py`: load FashionMNIST train split (`torchvision.datasets.FashionMNIST(root=..., train=True)`), convert to float in [0,1], compute per-channel mean and std two ways (full-tensor and streaming sum/sum-of-squares); print both.
3. `tests/test_transforms.py`.

**Verify — done when:**
- Your two stat computations agree to `atol=1e-5`. Expect approximately mean ≈ 0.286, std ≈ 0.353 — verify against *your* computation; the canonical figures floating around the internet are the check, not the ground truth.
- After `Normalize(your_mean, your_std)` over the whole train set: global mean within ±0.01 of 0 and std within ±0.01 of 1 (assert it).
- `RandomHorizontalFlip(p=1.0)` equals `x.flip(-1)` exactly; `p=0.0` is the identity; with `p=0.5` over 10,000 calls on a fixed asymmetric image, flip fraction in [0.47, 0.53].
- Leakage check (conceptual, write it in a comment): which split did your `Normalize` constants come from, and why does it matter?

**If stuck:** torchvision's functional `normalize` source (`torchvision/transforms/_functional_tensor.py` on GitHub) is a 10-line reference implementation.

---

## Day 5.4 — The Trainer (~2.5h)
- [ ] done
**Goal:** Build a reusable, dependency-free training engine you will use for the rest of the curriculum.

**Learn:**
- **Why a Trainer.** Every experiment repeats the same skeleton (loop, eval, logging, checkpointing). Centralizing it kills copy-paste bugs — the classic one being a forgotten `optimizer.zero_grad()` or `model.eval()`.
- **Gradient clipping.** `torch.nn.utils.clip_grad_norm_(params, max_norm)` computes the *global* norm over all gradients and, if it exceeds `max_norm`, rescales every gradient by `max_norm / total_norm`. It bounds the step size against loss spikes; it does not change gradient *direction*.
- **Gradient accumulation.** To emulate batch `B×k` with memory for `B`: run `k` micro-batches, calling `(loss / k).backward()` each time (grads sum across backward calls — Week 3 knowledge), and step+zero only every `k`-th micro-batch. The `/k` is what turns a sum of per-micro-batch means into the mean over the full effective batch.
- **Callbacks.** Instead of hard-coding early stopping or logging into the loop, the Trainer calls hook methods on observer objects at defined points. This is the pattern Lightning/HF Trainer use; you build the minimal version.
- **Checkpoints that actually resume.** Model weights alone are not a checkpoint: Adam's moment buffers, the epoch counter, and RNG state all change what happens next. Save all of them.

**Read (30–45 min):**
- `clip_grad_norm_` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
- Tutorial "Optimizing Model Parameters" — the loop structure you're generalizing: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- "Saving and Loading Models" — *Saving & Loading a General Checkpoint* section: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

**Build — `tlib/trainer.py`:**
1. Callback protocol:
   ```python
   class Callback:
       def on_fit_start(self, trainer: "Trainer") -> None: ...
       def on_epoch_end(self, trainer: "Trainer", epoch: int, logs: dict[str, float]) -> None: ...
       def on_fit_end(self, trainer: "Trainer") -> None: ...
   ```
   `logs` contains at least `train_loss`, `val_loss`, `val_acc`. A callback may set `trainer.should_stop = True`.
2. ```python
   class Trainer:
       def __init__(self, model, optimizer, loss_fn, train_loader, val_loader,
                    device: torch.device, callbacks: list[Callback] = (),
                    clip_norm: float | None = None, accumulate_steps: int = 1) -> None: ...
       def fit(self, epochs: int) -> dict[str, list[float]]:
           """Runs training; returns self.history (metric name -> per-epoch list).
           Per micro-batch: forward, (loss/accumulate_steps).backward();
           every accumulate_steps micro-batches: clip (if set), step, zero_grad."""
       def evaluate(self) -> dict[str, float]:
           """model.eval() + torch.no_grad(); mean val loss and accuracy."""
       def save_checkpoint(self, path: str) -> None:
           """torch.save dict: model_state, optimizer_state, epoch, history,
           torch_rng (torch.get_rng_state()), numpy_rng, python_rng."""
       def load_checkpoint(self, path: str) -> None: ...
   ```
   Details to honor: `model.train()` at the start of each epoch; move batches to `device`; if the loader length isn't divisible by `accumulate_steps`, flush remaining grads at epoch end (step once more).
3. Callbacks in the same file: `EarlyStopping(monitor="val_loss", patience=3, min_delta=0.0)` and `CSVLogger(path)` (header row on fit_start, one row per epoch: epoch + sorted(logs)).
4. `tests/test_trainer.py`.

**Verify — done when:**
- **Parity with Week 4:** re-train your Week-4 FashionMNIST MLP (your `tlib.modules` model, your AdamW) through `Trainer` for the same epochs/seed; final val accuracy within ~1 point of your Week-4 result and ≥ 85%.
- **Accumulation exactness test** (the load-bearing one):
  - Fix `seed_all(0)`. Build two identical models (same init: construct one, `copy.deepcopy` it) and two identical optimizers (plain SGD, lr 0.1).
  - Take one fixed tensor batch `X (64, D), y (64,)`. Run A: one step on the full 64. Run B: 4 micro-batches of 16 (slices `X[0:16]`, `X[16:32]`, …) with `accumulate_steps=4`, one effective step.
  - After stepping, assert `torch.allclose(pA, pB, atol=1e-6)` for every parameter pair.
  - Why the preconditions matter (write as comments): micro-batches must be equal-sized (else `/k` mis-weights them — this is also why real loaders need `drop_last=True`), and the model must contain no batch statistics (BatchNorm computes stats per *micro*-batch, so B ≠ A regardless of loss scaling).
- EarlyStopping test: feed a stub val_loss sequence [1.0, 0.9, 0.9, 0.9, 0.9] with patience=2 → fit stops after epoch 4 (assert epochs run == 4).
- CSVLogger output parses with `csv.DictReader` and has one row per epoch run.

**If stuck:** `clip_grad_norm_` source is in `torch/nn/utils/clip_grad.py` (GitHub); the *General Checkpoint* tutorial section above is the exact save/load dict pattern; your Week-3 notes on grad accumulation across `.backward()` calls.

---

## Day 5.5 — Reproducibility & experiment hygiene (~2h)
- [ ] done
**Goal:** Make any run exactly reproducible and give every experiment a config, a directory, and a row in a results file.

**Learn:**
- **Three RNGs, one seed function.** `random`, `numpy`, and `torch` each have independent generators; your Week-2 `seed_all` must set all three (plus `torch.cuda.manual_seed_all` for the GPU variant).
- **Determinism ≠ seeding.** Some ops use nondeterministic algorithms (atomics on GPU, some scatter ops). `torch.use_deterministic_algorithms(True)` forces deterministic variants or raises if none exists. On CPU with our ops, seeding alone usually suffices — but know the switch.
- **Worker seeding.** Each DataLoader worker derives its seed from `base_seed + worker_id`; with `num_workers>0`, pass a `worker_init_fn` to reseed numpy/random per worker and a seeded `torch.Generator` to the loader, or augmentation randomness won't reproduce.
- **Config as dataclass.** A frozen dataclass is greppable, type-checked, JSON-dumpable, and diff-able — everything a dict of kwargs isn't. One config = one run directory = one results row. Future-you debugging "which run was that?" will thank you.

**Read (30–45 min):**
- Reproducibility note — read it fully, including *DataLoader* workers section: https://docs.pytorch.org/docs/stable/notes/randomness.html
- `dataclasses` stdlib docs (skim if familiar): https://docs.python.org/3/library/dataclasses.html

**Build — `tlib/config.py`:**
1. ```python
   @dataclass(frozen=True)
   class ExperimentConfig:
       run_name: str
       seed: int = 0
       lr: float = 1e-3
       batch_size: int = 64
       epochs: int = 3
       hidden_sizes: tuple[int, ...] = (256, 128)
       weight_decay: float = 0.0
       out_root: str = "runs"

       def run_dir(self) -> pathlib.Path:
           """Create (mkdir -p) and return out_root/{run_name}_seed{seed}."""
       def save(self) -> None:
           """json.dump(dataclasses.asdict(self)) to run_dir()/config.json."""
   ```
2. `seed_worker(worker_id: int) -> None` in `tlib/data.py` (reseed numpy + random from `torch.initial_seed() % 2**32`), and a helper `make_loader(dataset, cfg, shuffle)` that wires `generator=torch.Generator().manual_seed(cfg.seed)`, `worker_init_fn=seed_worker`, `drop_last=True` for train.
3. A `run_experiment(cfg: ExperimentConfig) -> dict[str, float]` function in `scripts/run_fmnist.py`: seed_all → build loaders/model/optim from cfg → Trainer with CSVLogger into `run_dir()` → return final metrics. Append one row (all config fields + final metrics) to a shared `runs/results.csv`.
4. `tests/test_repro.py` on a tiny synthetic dataset (so it runs in seconds).

**Verify — done when:**
- Same seed twice ⇒ for every parameter, `torch.allclose(p1, p2, atol=0, rtol=0)` (bit-identical on CPU; if not, find the unseeded source — usually the DataLoader shuffle generator).
- Different seeds ⇒ at least one parameter pair is NOT allclose.
- `config.json` round-trips: `ExperimentConfig(**json.load(...)) == cfg` (note: tuples become lists in JSON — handle it, this is a classic gotcha).
- Running `run_experiment` twice appends two rows to `results.csv` with identical metric values for identical configs.

**If stuck:** the randomness note (link above) has the exact `seed_worker`/generator recipe under "DataLoader"; `torch.utils.data.get_worker_info` docs explain per-worker state.

---

## Day 5.6 — Deep build: LR finder + experiment grid (~3.5h)
- [ ] done
**Goal:** Implement the LR range test and run a disciplined 4-experiment grid through your full stack.

**Learn:**
- **The LR range test (Smith 2015).** Over ~one epoch, increase lr exponentially from tiny (1e-7) to huge (~10), recording loss per step. Loss plateaus (lr too small to move), then drops steeply (productive zone), then explodes (divergence). Pick an lr in the steep-descent region, before the minimum of the curve — the curve's minimum is already too hot, because loss there is about to blow up.
- **Why exponential, not linear:** good lrs span orders of magnitude; the interesting structure is in log-space.
- **EMA smoothing with bias correction.** Raw per-batch loss is noisy. Keep `avg = beta*avg + (1-beta)*loss` with `beta=0.98`, and report `smoothed = avg / (1 - beta**step)` — the same bias correction as Adam's moments (Week 4): without it, early values are dragged toward the zero-initialized `avg`.
- **Grid discipline.** Vary few things, hold the rest fixed, one results row per run, conclusions written down immediately — the habit matters more than this particular grid.

**Read (30–45 min):**
- Smith 2015, *Cyclical Learning Rates for Training Neural Networks* — Section 3.3 ("How can one estimate reasonable minimum and maximum boundary values?") is the LR range test: https://arxiv.org/abs/1506.01186
- Understanding Deep Learning, Ch. 6 *Fitting models* (SGD/lr discussion) — free PDF at https://udlbook.github.io/udlbook/

**Build:**
1. `tlib/lr_finder.py`:
   ```python
   def find_lr(model, optimizer, loss_fn, train_loader, device,
               lr_start: float = 1e-7, lr_end: float = 10.0,
               num_steps: int = 100, beta: float = 0.98,
               diverge_factor: float = 4.0) -> tuple[list[float], list[float]]:
       """LR range test. Saves model+optimizer state dicts on entry and restores
       them on exit (the test must not pollute the model).
       Per step t (0-indexed): lr_t = lr_start * (lr_end/lr_start) ** (t / (num_steps-1));
       set lr on all param groups; one ordinary train step; update EMA with bias
       correction (see Learn). Stop early if smoothed > diverge_factor * best_smoothed.
       Returns (lrs, smoothed_losses) of equal length."""
   ```
   Plus `def plot_lr_find(lrs, losses, path: str) -> None` — matplotlib, log-x, save PNG.
2. `tests/test_lr_finder.py`: on a trivial least-squares model, assert (a) returned lists are equal length ≤ num_steps, (b) lrs are strictly increasing with the exact exponential formula (`allclose`), (c) model params after `find_lr` equal params before (`allclose`, atol=0) — state restoration works, (d) feeding a loss sequence by hand through your EMA matches a hand-computed bias-corrected value at step 0 (smoothed == raw loss exactly).
3. `scripts/week5_grid.py`: run `find_lr` on the FashionMNIST MLP (AdamW), save `runs/lr_find.png`, eyeball-pick `lr_hi` from the steep region and set `lr_lo = lr_hi / 10`. Then the grid: `{lr_lo, lr_hi} × {batch 64, batch 256}`, 3 epochs each, all through `run_experiment(ExperimentConfig(...))`, names like `grid_lr{lr}_bs{bs}`.
4. Append a ~10-line note to `LOG.md`: which lr the finder suggested, which grid cell won, whether big-batch needed the bigger lr, one surprise.

**Verify — done when:**
- `pytest tests/test_lr_finder.py` passes.
- `runs/lr_find.png` exists and shows the plateau→descent→explosion shape (visual check; loss should visibly exceed its minimum by the end — if it never diverges, raise `lr_end`).
- `runs/results.csv` has 4 grid rows, each containing every config field plus final `train_loss`, `val_loss`, `val_acc`; all four runs reach val_acc > 0.80 except possibly the hottest cell (divergence there is an acceptable, *reportable* outcome — not a bug).
- `LOG.md` note written.

**If stuck:** Smith 2015 §3.3 (link above); your Adam bias-correction code from Week 4 — the EMA here is the same formula; `optimizer.param_groups[g]["lr"] = x` is how you set lr manually.

---

## Day 5.7 — Review, quiz, redo-cold (~2h)
- [ ] done
**Goal:** Consolidate the week; prove the core pieces are in your head, not just in your repo.

**Self-quiz** (write answers down before checking the Answers section at the bottom):
1. Name the four stages between a map-style dataset and a batch on your device, in order.
2. Why does `default_collate` fail on variable-length sequences, and what are the exact three tensors your `pad_collate` returns (shapes + dtypes)?
3. When do `num_workers > 0` make loading *slower*? What's the WSL2-specific advice?
4. What does `pin_memory=True` buy you, and in which hardware setup is it pure overhead?
5. If you pass `batch_sampler=` to DataLoader, which four other arguments must you not set?
6. Write the gradient-accumulation recipe for effective batch `B×k`: what do you scale, when do you step, when do you zero?
7. Why does BatchNorm break exact gradient-accumulation parity even with correct loss scaling?
8. What does `clip_grad_norm_(params, 1.0)` do when the global grad norm is 0.5? When it's 5.0?
9. List everything that must go into a checkpoint for training to resume *identically*.
10. Dataset normalization stats: computed on which split, applied to which splits, and what's the failure called if you get it wrong?
11. In the LR range test, why pick an lr *before* the minimum of the smoothed-loss curve?
12. Why does the EMA need bias correction, and what is the corrected formula at step t?

**Redo cold** (fresh files, no peeking at your week's code; ~20 min each):
- Rewrite `pad_collate` from the spec in your head; check against your tests.
- Rewrite the gradient-accumulation inner loop of `Trainer.fit` (just the loop body) and re-run the batch-64-vs-16×4 parity test against it.
- Rewrite `seed_all` + the DataLoader worker-seeding wiring; verify the same-seed ⇒ identical-weights test still passes.
- Write the LR-finder lr schedule formula and the bias-corrected EMA as two pure functions; test against step 0 and step ∞ limits.

**If stuck on any quiz item:** the day it came from lists the exact doc section; re-read that, not your own code, first.

---

## Answers

1. `Sampler` yields indices → `BatchSampler` groups them into lists → fetcher calls `dataset[i]` per index → `collate_fn` merges samples into batched tensors (then optional pin-memory + transfer to device in your loop).
2. It calls `torch.stack`, which requires identical shapes. `pad_collate` returns `padded (B, L_max) int64` right-zero-padded, `lengths (B,) int64`, `mask (B, L_max) bool` (True = real token).
3. When `__getitem__` is cheap (small in-memory tensors): worker process startup + IPC serialization cost more than they save. On WSL2 these overheads are worse than native Linux — start at 0, try 2 only if loading is the measured bottleneck.
4. Page-locked host memory enabling async CPU→GPU copies (`non_blocking=True`). On CPU-only training there is no transfer, so it's pure overhead.
5. `batch_size`, `shuffle`, `sampler`, `drop_last` (all are forms of batching/ordering control you've taken over).
6. Call `(loss / k).backward()` on each of the k micro-batches (grads sum across backward calls); `optimizer.step()` and `zero_grad()` only after every k-th micro-batch. The `/k` converts the sum of k micro-batch means into the full-batch mean.
7. BatchNorm normalizes with *per-micro-batch* statistics; a batch of 64 and four batches of 16 see different means/vars, so the forward passes differ before loss scaling even enters.
8. Norm 0.5 ≤ 1.0: gradients untouched. Norm 5.0: every gradient multiplied by 1.0/5.0, so the global norm becomes exactly 1.0; direction unchanged.
9. Model state_dict, optimizer state_dict (Adam moments!), epoch/step counter, history/best-metric for early stopping, and RNG states (torch, numpy, python; CUDA too on GPU).
10. Computed on train only; applied identically to train/val/test; the failure is data leakage.
11. At the curve's minimum the loss is about to diverge — that lr is already marginally unstable. The steep-descent region before it is where training makes fast, stable progress.
12. `avg` is initialized at 0, so early EMA values are biased low. Corrected: `smoothed_t = avg_t / (1 - beta**(t+1))` (with t 0-indexed and `avg_t = beta*avg_{t-1} + (1-beta)*loss_t`); at t=0 this returns exactly `loss_0`.
