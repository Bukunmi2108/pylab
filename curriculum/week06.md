# Week 6 — Training dynamics & normalization

You have a training stack; this week you learn to see inside it while it runs, and to build the layers that keep deep networks trainable. You implement a hooks-based diagnostics toolkit, then every major normalization layer (BatchNorm, LayerNorm, RMSNorm, GroupNorm), LR schedules, and dropout — each verified numerically against PyTorch's own implementations. The week ends in a "sick-network clinic" where you diagnose four deliberately broken setups from statistics alone. Sequenced here because Week 7's convnets need BatchNorm + schedules + diagnostics, and Week 9's transformer is built on *your* LayerNorm/RMSNorm and warmup-cosine schedule.

**Week outcome:** `tlib/diagnostics.py` (ActivationStats, GradStats), `tlib/norms.py` (BatchNorm1d, LayerNorm, RMSNorm, GroupNorm — all allclose vs torch), `tlib/schedules.py` (step/cosine/warmup-cosine/one-cycle), Dropout in `tlib/modules.py`, all pytest-covered; four written before/after diagnoses in `LOG.md`. Skill: given a sick training run, you can name the failing layer and statistic before touching a fix.

---

## Day 6.1 — Hooks & a diagnostics toolkit (~2.5h)
- [ ] done
**Goal:** Build context-manager classes that record per-layer activation and gradient statistics via hooks.

**Learn:**
- **Forward hooks.** `module.register_forward_hook(fn)` calls `fn(module, input, output)` after every forward of that module — you observe activations without touching model code. It returns a `RemovableHandle`; you *must* call `handle.remove()` when done or hooks accumulate across runs (a classic leak).
- **Backward hooks.** `module.register_full_backward_hook(fn)` calls `fn(module, grad_input, grad_output)` during backward; `grad_output[0]` is dL/d(module output). Use the `full_` variant — the older `register_backward_hook` is deprecated and gave wrong results on multi-input modules.
- **What to record.** Per layer: activation mean, std, and fraction "dead" (≤ 0 — meaningful after ReLU; for tanh record fraction saturated, |a| > 0.97), plus grad-output norm. These four numbers distinguish vanishing, explosion, saturation, and dead units.
- **Leaf modules.** Hook the modules that do work (`Linear`, activations), not containers (`Sequential`) — iterate `model.named_modules()` and skip anything with children.
- **Context manager = guaranteed cleanup.** `__enter__` attaches hooks, `__exit__` removes them, even on exception. Statistics recording becomes `with ActivationStats(model) as s: model(x)`.

**Read (30–45 min):**
- `nn.Module` docs — `register_forward_hook`, `register_full_backward_hook`, `named_modules` entries: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
- Understanding Deep Learning Ch. 7 *Gradients and initialization* (the variance-propagation story you'll observe empirically today): https://udlbook.github.io/udlbook/

**Build — `tlib/diagnostics.py`:**
1. ```python
   class ActivationStats:
       """Context manager. Attaches forward hooks to all leaf modules of `model`
       (modules with no children). Each forward pass appends one record per layer:
       {"layer": name, "mean": float, "std": float, "frac_dead": float, "frac_sat": float}
       (frac_dead: fraction of output elements <= 0; frac_sat: fraction |out| > 0.97).
       Detach before computing stats — never keep graph references."""
       def __init__(self, model: torch.nn.Module) -> None: ...
       def __enter__(self) -> "ActivationStats": ...
       def __exit__(self, *exc) -> None: ...   # removes ALL handles
       def table(self) -> list[dict]: ...       # latest record per layer, in model order
       def plot_hists(self, path: str) -> None:
           """Matplotlib grid (one subplot per layer) of activation histograms
           from the last forward pass (store a small detached sample, e.g. 5k
           elements per layer, not the full tensor). Saves PNG."""

   class GradStats:
       """Same shape, but register_full_backward_hook; records per-layer
       {"layer", "grad_norm", "grad_std"} from grad_output[0] during backward."""
   ```
2. A pretty-printer `def print_stats(rows: list[dict]) -> None` (aligned columns; no dependencies).
3. The test rig — rebuild the deep tanh MLP from Week 4 (D4.2) in `tests/test_diagnostics.py`:
   ```python
   def deep_tanh_mlp(depth: int = 20, width: int = 256, w_std: float | str = "xavier") -> nn.Sequential:
       """depth × [Linear(width, width), Tanh()]. If w_std is a float, init weights
       N(0, w_std**2), biases 0; if "xavier", use your tlib.init xavier with tanh gain (5/3)."""
   ```
   Drive it with `x = torch.randn(1024, width)` under a fixed seed.

**Verify — done when:**
- **Vanishing case** (`w_std=0.01`): in `ActivationStats.table()`, the std of the layer-20 tanh output is < 1e-6 (each layer multiplies std by roughly `0.01·√256 = 0.16`, so it collapses fast — assert it).
- **Saturation case** (`w_std=1.0`): pre-activations have std ≈ √256 ≈ 16, so tanh pins to ±1 — assert layer-20 `frac_sat > 0.8` and, via `GradStats` after `model(x).sum().backward()`, that the grad norm at layer 1 is < 1e-3 × the grad norm at layer 20 (saturated tanh kills the backward signal).
- **Healthy case** (`"xavier"`): layer-20 activation std > 0.1 (expect roughly 0.3–0.6) and `frac_sat < 0.2`.
- Hook hygiene: after the `with` block, `len(model._forward_hooks) == 0` for every module (assert), and a second `with` block works identically.
- `plot_hists` writes a PNG; the three cases are visually unmistakable (spike at 0 / twin spikes at ±1 / smooth bell).

**If stuck:** the hooks section of the `nn.Module` docs page (link above) shows the exact callback signatures; `RemovableHandle` is documented under torch.utils.hooks; your Week-4 init derivations explain *why* each case behaves as observed.

---

## Day 6.2 — BatchNorm from scratch (~2.5h)
- [ ] done
**Goal:** Implement BatchNorm1d with correct train/eval duality and verify it against `nn.BatchNorm1d` in both modes, including gradients.

**Learn:**
- **The two-mode duality.** Train mode: normalize each feature by the *current batch's* mean/var, and update running estimates. Eval mode: normalize by the *running* estimates — no batch dependence, so a single example gets a deterministic output. Forgetting `model.eval()` is the most common BN bug in the wild.
- **The exact recipe (train).** Per feature j: `x̂ = (x - μ_B) / √(σ²_B + eps)`, then `y = γ·x̂ + β`. Normalization uses the *biased* variance (divide by N); the running-var update uses the *unbiased* variance (divide by N−1) — PyTorch does both, and your parity test will fail if you mix them up.
- **Running stats are buffers, not parameters.** They're state that must live in `state_dict` and move with `.to(device)`, but receive no gradients: `register_buffer("running_mean", ...)`. `momentum=0.1` means `running = (1−0.1)·running + 0.1·batch_stat` (note: torch's "momentum" is the weight on the *new* value — opposite of optimizer momentum conventions).
- **Why it helps — the honest version.** BN lets higher lrs work and makes initialization less critical. The original paper credited reducing "internal covariate shift"; Santurkar et al. 2018 showed ICS barely changes and the measurable effect is a smoother loss landscape (smaller, more predictable gradients). The mechanism story is contested; the empirical benefit isn't.
- **The pathology.** Batch statistics are noisy estimates; with batch size 2–4 the noise swamps the signal and BN *hurts*. This batch-size dependence is the reason LayerNorm/GroupNorm exist (tomorrow).

**Read (30–45 min):**
- Ioffe & Szegedy 2015, *Batch Normalization* — Sections 1–3 and Algorithm 1: https://arxiv.org/abs/1502.03167
- Santurkar et al. 2018, *How Does Batch Normalization Help Optimization?* — abstract + Section 2: https://arxiv.org/abs/1805.11604
- `nn.BatchNorm1d` docs — read the momentum and track_running_stats notes carefully: https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html

**Build — `tlib/norms.py`:**
1. ```python
   class BatchNorm1d(nn.Module):
       def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
           """Parameters: weight=γ (init ones), bias=β (init zeros), shape (num_features,).
           Buffers: running_mean (zeros), running_var (ones), num_batches_tracked (0)."""
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """x: (B, num_features). Training: normalize with batch mean and BIASED
           batch var; update running stats with UNBIASED var:
             running_mean = (1-momentum)*running_mean + momentum*batch_mean
             running_var  = (1-momentum)*running_var  + momentum*batch_var_unbiased
           Eval: normalize with running stats. Both: y = γ*x̂ + β. Returns (B, num_features)."""
   ```
   Use `self.training` (inherited from nn.Module) to branch — do not invent your own flag.
2. `tests/test_batchnorm.py` implementing the parity protocol below.

**Verify — done when:**
- **Train-mode parity:** `mine = BatchNorm1d(32)`, `ref = nn.BatchNorm1d(32)`; copy γ/β over (`load_state_dict` works if you matched names — a worthwhile goal). Feed the *same sequence* of 5 random batches `(64, 32)` through both in train mode; after each batch assert outputs `allclose(atol=1e-6)`.
- **Running-stat parity:** after those 5 batches, `running_mean` and `running_var` allclose vs ref's buffers (`atol=1e-6`). This is the test that catches biased/unbiased mix-ups.
- **Eval-mode parity:** `.eval()` both, feed a fresh batch, outputs allclose; also assert eval output for batch size 1 raises no error and is deterministic (run twice, identical).
- **Gradient parity:** same input with `requires_grad=True` through both (train mode), `out.square().sum().backward()`; assert `x.grad` allclose between mine and ref, and γ/β grads allclose. (Alternative: your Week-3 `tlib/gradcheck.py` in float64.)
- Sanity property: in train mode, output per-feature mean ≈ 0 and std ≈ 1 (within 1e-5) when γ=1, β=0 — guaranteed by construction, assert it.

**If stuck:** Algorithm 1 in the BN paper is the literal spec; `nn.BatchNorm1d` doc's note on momentum semantics; torch's reference forward lives in `torch/nn/modules/batchnorm.py` (GitHub) — read `_BatchNorm.forward` only after you've tried.

---

## Day 6.3 — LayerNorm, RMSNorm, GroupNorm (~2.5h)
- [ ] done
**Goal:** Implement the three batch-independent normalizers and understand which axis each one averages over — the only real difference between them.

**Learn:**
- **It's all about the axis.** BatchNorm averages over the batch dim (each feature normalized across examples). LayerNorm averages over the feature dims (each *example* normalized across its own features). GroupNorm splits channels into groups and normalizes within each group per example. Same formula, different `dim=` — write that sentence on a sticky note.
- **Why transformers use LayerNorm/RMSNorm.** No batch dependence: works at batch size 1, identical in train and eval (no running stats at all), and well-defined per token in a variable-length sequence. BN over padded, variable-length sequences is somewhere between awkward and wrong.
- **RMSNorm (Zhang & Sennrich 2019)** drops the mean-centering and the bias: `y = x / RMS(x) · γ`, where `RMS(x) = √(mean(x²) + eps)`. The paper's claim: re-centering contributes little; re-scaling is what matters — and you save a reduction. It's the default in Llama-family models, hence in your Week-9 transformer.
- **GroupNorm (Wu & He 2018)** is the convnet compromise: batch-independent like LN, but doesn't force all channels into one statistic; `num_groups=1` degenerates to LN-over-channels, `num_groups=C` to InstanceNorm.

**Read (30–45 min):**
- Zhang & Sennrich 2019, *Root Mean Square Layer Normalization* — Sections 3–4: https://arxiv.org/abs/1910.07467
- Ba, Kiros & Hinton 2016, *Layer Normalization* — Sections 1–3: https://arxiv.org/abs/1607.06450
- API references (read the shape semantics): https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html , https://docs.pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html , https://docs.pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html

**Build (extend `tlib/norms.py`):**
1. ```python
   class LayerNorm(nn.Module):
       def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
           """γ ones, β zeros, shape = normalized_shape. Normalizes over the LAST
           len(normalized_shape) dims using biased variance.
           E.g. x (B, T, D), normalized_shape=D: each (b,t) vector normalized over D."""

   class RMSNorm(nn.Module):
       def __init__(self, dim: int, eps: float = 1e-6) -> None:
           """γ ones (dim,). y = x / sqrt(mean(x**2, dim=-1, keepdim=True) + eps) * γ.
           No mean subtraction, no bias."""

   class GroupNorm(nn.Module):
       def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5) -> None:
           """x (B, C, *spatial). Reshape to (B, G, C//G, *spatial), normalize over
           everything except B and G (biased var), reshape back, then per-CHANNEL γ, β."""
   ```
2. `tests/test_norms.py` with parity tests for all three.

**Verify — done when:**
- LayerNorm vs `nn.LayerNorm(D)` on `x = torch.randn(8, 16, 64)`: outputs allclose (atol=1e-6) and input grads allclose after a `.sum().backward()`; also test tuple `normalized_shape=(16, 64)`.
- GroupNorm vs `nn.GroupNorm(4, 32)` on `(8, 32, 10)` and on `(8, 32)`-plus-spatial `(8, 32, 5, 5)`: outputs allclose.
- RMSNorm: `nn.RMSNorm` exists in torch ≥ 2.4 — guard with `hasattr(torch.nn, "RMSNorm")`. If present, compare vs `nn.RMSNorm(64, eps=1e-6)` (pass eps explicitly: torch's default is dtype-dependent machine epsilon, not 1e-6). If absent, verify against the hand formula in float64: `x64 / (x64.pow(2).mean(-1, keepdim=True) + eps).sqrt() * g64`, atol=1e-10.
- Axis-comprehension property test (no torch reference needed): for `x (B, D)`, your LayerNorm output has per-*row* mean ≈ 0 (atol 1e-6), while your BatchNorm1d (train) output has per-*column* mean ≈ 0. Assert both — if this test confuses you, today isn't done.

**If stuck:** the "shape" diagrams in the `nn.GroupNorm` doc page; RMSNorm paper Eq. 4; your LayerNorm can be debugged by checking `out.mean(-1)` and `out.std(-1)` before adding γ/β.

---

## Day 6.4 — LR schedules from scratch (~2h)
- [ ] done
**Goal:** Implement the four standard schedules as plain, testable classes and verify cosine annealing in lockstep against torch.

**Learn:**
- **Schedules are just functions of the step counter.** All the `torch.optim.lr_scheduler` machinery (chaining, state dicts, `.step()` bookkeeping) wraps functions `step → lr`. Implementing them as pure `get_lr(step)` classes makes them trivially plottable and testable.
- **Step decay:** multiply lr by `gamma` every `step_size` steps — the classic staircase; simple, but each drop is a discontinuity.
- **Cosine annealing (Loshchilov & Hutter 2016):** `lr(t) = eta_min + (base − eta_min)·(1 + cos(π·t/T))/2` — smooth decay from base to eta_min over T steps, fast at first, gentle at the end.
- **Warmup + cosine — the LLM-standard schedule.** Adaptive optimizers misbehave in the first steps (moment estimates are garbage; cf. your Week-4 bias-correction work), and early large steps can wreck a fresh network. Fix: ramp lr linearly from 0 over `warmup_steps`, then cosine-decay to `min_lr`. Your Week-9 transformer uses exactly this class.
- **One-cycle (Smith & Topin 2017):** up from `max_lr/div_factor` to `max_lr` over the first `pct_start` of training, then down to `max_lr/final_div_factor`; the claim is faster convergence at high lrs ("super-convergence").

**Read (30–45 min):**
- `CosineAnnealingLR` docs (read the formula note carefully — and note their recursive form equals the closed form only when stepped once per epoch monotonically): https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
- Loshchilov & Hutter 2016, *SGDR* — Section 3 for the cosine formula: https://arxiv.org/abs/1608.03983
- Smith & Topin 2017, *Super-Convergence* — skim Sections 1–3: https://arxiv.org/abs/1708.07120

**Build — `tlib/schedules.py`:**
1. All classes expose `get_lr(self, step: int) -> float` (step is 0-indexed) and share:
   ```python
   def apply_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
       """Sets param_group['lr'] = lr for every param group."""
   ```
2. ```python
   class StepDecay:        # lr = base_lr * gamma ** (step // step_size)
       def __init__(self, base_lr: float, step_size: int, gamma: float = 0.1): ...

   class CosineAnnealing:  # lr = eta_min + (base_lr - eta_min) * (1 + cos(pi * step / t_max)) / 2
       def __init__(self, base_lr: float, t_max: int, eta_min: float = 0.0): ...

   class WarmupCosine:
       """step < warmup_steps:  lr = base_lr * (step + 1) / warmup_steps
          else: u = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
                lr = min_lr + (base_lr - min_lr) * 0.5 * (1 + cos(pi * u))
       So get_lr(warmup_steps-1) == base_lr and get_lr(total_steps-1) == min_lr."""
       def __init__(self, base_lr: float, warmup_steps: int, total_steps: int, min_lr: float = 0.0): ...

   class OneCycle:
       """First pct_start*total_steps: cosine-up from base_lr/div_factor to base_lr;
          remainder: cosine-down from base_lr to base_lr/final_div_factor."""
       def __init__(self, base_lr: float, total_steps: int, pct_start: float = 0.3,
                    div_factor: float = 25.0, final_div_factor: float = 1e4): ...
   ```
3. `scripts/plot_schedules.py`: all four on one matplotlib figure (1,000 steps, labeled legend) → `runs/schedules.png`.
4. `tests/test_schedules.py`.

**Verify — done when:**
- Cosine lockstep: SGD with `lr=0.1`, `torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)`; for step t in 0..99: record `opt.param_groups[0]["lr"]`, compare to `CosineAnnealing(0.1, t_max=100).get_lr(t)` (`math.isclose`, rel_tol=1e-9), then `scheduler.step()`.
- WarmupCosine boundary exactness: `get_lr(0) == base_lr/warmup_steps`, `get_lr(warmup_steps-1) == base_lr`, `get_lr(total_steps-1) == min_lr` (exact, by your formula).
- StepDecay: `get_lr` at steps [0, step_size-1, step_size, 2*step_size] equals `[b, b, b*g, b*g**2]` exactly.
- OneCycle: max over all steps == `base_lr` (reached at the peak step); lr at step 0 == `base_lr/div_factor`; final lr == `base_lr/final_div_factor` (allclose).
- `runs/schedules.png` exists with all four curves.

**If stuck:** the CosineAnnealingLR doc formula box; SGDR paper Eq. 5; print your warmup-cosine at steps {0, w-1, w, total-1} and check against the docstring by hand.

---

## Day 6.5 — Dropout & regularization (~2.5h)
- [ ] done
**Goal:** Implement inverted dropout, verify it statistically, and demonstrate that regularization measurably shrinks the train/val gap on a small-data problem.

**Learn:**
- **Inverted dropout.** Train: zero each element independently with prob p, scale survivors by `1/(1−p)`. The scaling makes `E[output] = input`, so eval mode is the *identity* — no test-time correction needed (that's the "inverted" part; the original 2014 formulation scaled at test time instead).
- **Why it regularizes:** units can't co-adapt — each must be useful when random teammates vanish; it's also an implicit ensemble over thinned subnetworks.
- **Placement:** after activations in MLP hidden layers; never on the output layer; in transformers, on attention weights and residual-branch outputs (Week 9). Dropout + BatchNorm interact poorly (variance mismatch between train and eval) — another reason transformer-era nets pair dropout with LayerNorm.
- **Weight decay recap, sharpened.** With plain SGD, adding `wd·θ` to the gradient (L2) and decoupled decay (`θ ← θ − lr·wd·θ` separate from the gradient step) coincide up to lr scaling. With Adam they genuinely differ: L2-via-grad gets divided by `√v̂`, so heavily-updated weights get *less* decay; AdamW applies decay uniformly. You built both in Week 4 — today you demonstrate the difference numerically.

**Read (30–45 min):**
- Srivastava et al. 2014, *Dropout* — Sections 1, 4 (the model description): https://jmlr.org/papers/v15/srivastava14a.html
- Loshchilov & Hutter 2017, *Decoupled Weight Decay Regularization* — Sections 1–2 + Algorithm 2: https://arxiv.org/abs/1711.05101
- `nn.Dropout` docs (one paragraph; note the train/eval behavior line): https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html

**Build:**
1. In `tlib/modules.py`:
   ```python
   class Dropout(nn.Module):
       def __init__(self, p: float = 0.5) -> None:
           """0 <= p < 1; raise ValueError otherwise."""
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """Training: mask = (torch.rand_like(x) >= p); return x * mask / (1 - p).
           Eval: return x unchanged (same object or clone — but bitwise equal)."""
   ```
2. Adam-vs-AdamW divergence demo in `tests/test_dropout_wd.py`: one `Linear(10, 1)`, identical init copies; train 50 steps on fixed random data with your Week-4 Adam + L2-in-grad (wd=0.1) vs your AdamW (wd=0.1), same lr/seed. Assert final weights are NOT allclose (atol=1e-4) — then, as a control, run both with wd=0.0 and assert they ARE allclose.
3. Overfitting experiment `scripts/week6_overfit.py`: take exactly 1,000 FashionMNIST training images (fixed seeded subset via `torch.randperm(60000)[:1000]`), full 10k val set. Train your Week-4 MLP (e.g. 784→256→256→10) for ~30 epochs through your Trainer, twice: (a) no regularization; (b) `Dropout(0.5)` after each hidden ReLU + AdamW `weight_decay=1e-2`. Log per-epoch train/val accuracy via CSVLogger; plot both gap curves → `runs/overfit_gap.png`; 5-line conclusion in `LOG.md`.

**Verify — done when:**
- Statistical tests on `Dropout(0.3)` over `x = torch.ones(1_000_000)` in train mode: zero fraction within 0.30 ± 0.005; output mean within 1.0 ± 0.01; surviving elements all exactly equal `1/0.7` (assert via unique values).
- Eval mode: `torch.equal(drop(x), x)` exactly, for any p.
- `p=0.0` in train mode is the identity (exact).
- Adam-vs-AdamW test passes both ways (differ with wd>0, match with wd=0).
- Overfitting run: case (a) reaches ≈100% train accuracy with val plateauing far lower (typically ~80–84%); case (b)'s final train−val gap is smaller than (a)'s (assert gap_b < gap_a in the script; expect roughly half, but only assert the inequality — the exact numbers are stochastic).

**If stuck:** the `nn.Dropout` doc paragraph is the full spec; AdamW paper Algorithm 2 line-by-line vs your Week-4 `tlib/optim.py`; if the overfit gap won't appear, your subset is too big or your model too small — shrink data first.

---

## Day 6.6 — Deep build: the sick-network clinic (~3.5h)
- [ ] done
**Goal:** Diagnose four broken training setups using only your diagnostics toolkit — written diagnosis first, fix second, recovery evidence third.

**The discipline (read first):** For each case below: (1) build it exactly as specced; (2) run 50 training steps on FashionMNIST (batch 128, your Trainer or a bare loop) capturing `ActivationStats` + `GradStats` tables at step 0 and step 50, plus the loss curve; (3) write a 2-sentence diagnosis in `LOG.md` naming the failing layer(s) and the statistic that proves it — *before* applying any fix; (4) apply the named fix and re-capture the same evidence. No staring at loss curves alone; the tables must carry the argument.

**The four patients (type these in `scripts/week6_clinic.py`):**
1. **Patient A — saturated tanh.** `deep_tanh_mlp(depth=10, width=256, w_std=1.0)` from D6.1 + `Linear(256, 10)` head, SGD lr=0.1, cross-entropy. Expected evidence: `frac_sat ≈ 1` in deep layers, grad norms decaying ~orders of magnitude from head to layer 1, flat loss. Fix: xavier init with tanh gain.
2. **Patient B — lr 100× too high.** Your healthy Week-4 ReLU MLP (kaiming init) but SGD lr=10.0 (vs the ~0.1 that works). Expected evidence: activation std and grad norms exploding across *steps* (compare step-0 vs step-5 tables), loss → inf/NaN within ~tens of steps. Fix: lr=0.1. (Note the diagnostic signature: per-*step* explosion, vs Patient A's per-*layer* decay.)
3. **Patient C — dead ReLUs.** ReLU MLP, kaiming weights, but every hidden bias initialized to −3.0 (`with torch.no_grad(): layer.bias.fill_(-3.0)`). Expected evidence: `frac_dead > 0.95` in hidden layers at step 0 and *still* at step 50 (dead units get zero gradient — they can't recover), loss nearly flat. Fix: bias zeros.
4. **Patient D — unnormalized features.** Synthetic 2-class data: `X = torch.randn(4096, 20)`, then scale columns 0–9 by 1e3 and columns 10–19 by 1e-3; labels from a fixed random linear rule on the *unscaled* X. Plain MLP, SGD. Expected evidence: layer-1 activation std enormous; loss decreases erratically or stalls; first-layer weight-grad norms wildly imbalanced between the two column groups (extend GradStats or inspect `layer1.weight.grad[:, :10]` vs `[:, 10:]` norms directly). Fix: standardize columns (mean 0, std 1) using train-set stats — your D5.3 reasoning.

**Build artifacts:** `scripts/week6_clinic.py` runs all four patients before+after, dumps each stats table to `runs/clinic/{patient}_{before|after}.txt` and histogram PNGs; `LOG.md` gains four diagnosis entries (each: 2-sentence diagnosis naming layer+statistic, the fix, 1 sentence of recovery evidence).

**Verify — done when:**
- Each patient has before/after stats files + plots on disk.
- Programmatic checks in the script (assert, with the listed expectations): A-before layer-10 `frac_sat > 0.8`, A-after `< 0.2`; B-before loss becomes non-finite or > 10× initial within 50 steps, B-after loss at step 50 < initial loss; C-before hidden `frac_dead > 0.9` at step 50, C-after `< 0.6`; D-before layer-1 input-side std ratio between column groups > 1e4, D-after loss at step 50 < 0.5× initial.
- All four `LOG.md` diagnoses were written before the corresponding fix was applied (honor system — but the habit is the deliverable).
- After fixes: all four runs show monotonically-ish decreasing smoothed loss (eyeball + the asserts above).

**If stuck:** your own D6.1 verify cases are Patients A's twin; UDL Ch. 7 figures for variance propagation; for Patient C recall the ReLU gradient: zero input region ⇒ zero gradient ⇒ permanent death unless the *input distribution* shifts.

---

## Day 6.7 — Review, quiz, redo-cold (~2h)
- [ ] done
**Goal:** Lock in the normalization-and-dynamics toolbox before Week 7 puts it under convnet load.

**Self-quiz** (answers at the bottom — write yours first):
1. Forward hook vs full backward hook: exact callback signatures, and what is `grad_output[0]`?
2. Why must hook handles be removed, and how does your context manager guarantee it?
3. BatchNorm train mode: which variance (biased/unbiased) is used for normalization, and which for the running-var update?
4. Write torch's running-stat update rule with momentum=0.1. How does this "momentum" differ from SGD momentum?
5. Why is BatchNorm problematic at batch size 2? Which two layers from this week are immune, and why?
6. LayerNorm vs BatchNorm on input `(B, D)`: over which axis does each compute its mean? Which produces row-wise zero mean?
7. Write RMSNorm's full formula. What two things does it drop relative to LayerNorm?
8. Write the warmup+cosine schedule as a piecewise formula (the one your Week-9 transformer will use).
9. Why does inverted dropout scale by `1/(1−p)` at train time, and what exactly happens at eval?
10. With Adam, why do L2-in-the-gradient and decoupled weight decay produce different trajectories? Which one is "AdamW"?
11. A deep ReLU net shows `frac_dead = 0.97` in layer 5 at step 0 *and* step 500. Why can't those units recover on their own?
12. Patient A (per-layer gradient decay) vs Patient B (per-step explosion): name the one diagnostic table-comparison that distinguishes them.

**Redo cold** (fresh files, no peeking; ~20–25 min each):
- Rewrite `BatchNorm1d.forward` from memory — both modes, running-stat updates, biased/unbiased distinction — then re-run your D6.2 parity tests against it.
- Write `WarmupCosine.get_lr` from memory; verify the three boundary values (step 0, warmup−1, total−1) by hand before running the test.
- Rewrite `RMSNorm` from memory; verify vs `nn.RMSNorm` (or float64 formula).
- From memory, list the four clinic signatures (statistic + where it shows up) and their fixes; check against `LOG.md`.

---

## Answers

1. Forward: `fn(module, input, output)` called after forward (`input` is a tuple). Full backward: `fn(module, grad_input, grad_output)` during backward; `grad_output[0]` is the gradient of the loss w.r.t. the module's output.
2. Hooks persist on the module and stack up across attachments, double-recording stats and leaking memory. The context manager stores every `RemovableHandle` in `__enter__` and calls `.remove()` on all of them in `__exit__`, which Python runs even if the body raises.
3. Biased variance (divide by N) for normalizing the batch; unbiased variance (divide by N−1) for updating `running_var`.
4. `running = 0.9·running + 0.1·batch_stat` — i.e. torch's momentum is the weight on the *new* observation. SGD momentum is the weight on the *old* accumulated velocity; the conventions are opposite.
5. With N=2 the batch mean/var are extremely noisy estimates, so the "normalization" injects large noise and train statistics diverge badly from running statistics. LayerNorm and RMSNorm (and GroupNorm) are immune: they normalize per-example over features, never touching the batch axis.
6. LayerNorm: mean over D (the feature axis), per row — so LayerNorm gives row-wise zero mean. BatchNorm: mean over B (the batch axis), per column — column-wise zero mean.
7. `y = γ ⊙ x / √(mean(x²) + eps)` with mean over the last dim. Drops mean-centering and the bias β.
8. For `t < W`: `lr(t) = base·(t+1)/W`. For `t ≥ W`: `u = (t−W)/(T−W−1)`, `lr(t) = min_lr + (base − min_lr)·½(1 + cos(πu))` (T = total steps; endpoints: base at t = W−1, min_lr at t = T−1).
9. Scaling survivors by `1/(1−p)` keeps `E[output] = input`, so the network's expected activations are scale-consistent between train and eval; eval mode is then exactly the identity — no correction, no randomness.
10. Adam divides the (gradient + wd·θ) update by `√v̂`, so L2-in-grad decay is shrunk for parameters with large gradient history — decay strength becomes coupled to gradient statistics. Decoupled decay subtracts `lr·wd·θ` outside the adaptive machinery, decaying all weights uniformly. The decoupled version is AdamW.
11. A dead ReLU outputs 0 and has zero local gradient, so its incoming weights and bias receive zero gradient — there is no learning signal to move its pre-activation back above zero. Only a shift in the *input* distribution (driven by upstream layers that still learn) can revive it.
12. Compare the same table across *time* vs across *depth*. Patient A: at any fixed step, grad norms decay layer-20 → layer-1 (depth-wise), stable across steps. Patient B: norms are sane across depth at step 0 but grow explosively from step to step.
