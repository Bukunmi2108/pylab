# Week 4 — nn & optimizers from scratch

Week 3 demystified gradients; this week demystifies everything stacked on top of them. You will reimplement the layers (`Linear`, `ReLU`, `Sequential`), the initialization schemes (Xavier, Kaiming — derived, not copied), the losses (numerically stable BCE-with-logits and cross-entropy with label smoothing), and the optimizers (SGD-with-momentum through AdamW) — verifying each against torch's implementation in lockstep, parameter-for-parameter. It sequences here because each piece is a thin, checkable wrapper around what you already own: modules wrap forward math, optimizers wrap the `p.data -= lr * p.grad` line from Day 3.6. By day 6 you train FashionMNIST end-to-end using only `tlib` components.

**Week outcome:** `tlib/modules.py`, `tlib/init.py`, `tlib/losses.py`, `tlib/optim.py` — each verified bit-for-bit (allclose) against its torch counterpart — plus `week04/fashion_train.py` reaching approximately 85% test accuracy on FashionMNIST with your stack. Skill by day 7: you can write the Adam update from memory and explain why a 20-layer tanh net is untrainable under N(0,1) init.

## Day 4.1 — nn.Module anatomy + your Linear/ReLU/Sequential (~2.5h)
- [ ] done

**Goal:** Understand how `nn.Module` registers parameters and manages state, then implement `Linear`, `ReLU`, and `Sequential` whose outputs and parameter counts match torch's exactly.

**Learn:**
- **`__init__` declares, `forward` computes.** A Module's constructor creates parameters and submodules; `forward` is pure computation on them. You call the module itself (`m(x)`), never `m.forward(x)` directly — `__call__` wraps forward with hook dispatch.
- **The `__setattr__` registration trick.** `nn.Module` overrides `__setattr__`: assigning an `nn.Parameter` to `self.weight` silently inserts it into `self._parameters`; assigning a Module inserts into `self._modules`. That's the entire "magic" behind `.parameters()` finding everything — a recursive walk over these dicts.
- **`Parameter` vs buffer.** `nn.Parameter` is a Tensor subclass meaning "optimize me" (`requires_grad=True`, yielded by `.parameters()`). A buffer (`self.register_buffer("running_mean", t)`) is state that must be saved/moved-to-device but NOT optimized — e.g. BatchNorm's running statistics (you'll build BatchNorm in Week 6).
- **`state_dict()`** flattens parameters AND buffers into an ordered dict of name → tensor (`"net.0.weight"` style dotted names). It is the serialization unit: `load_state_dict` restores values without rebuilding objects.
- **`train()` / `eval()`** just flip a boolean `self.training` recursively on all children. Linear ignores it; Dropout and BatchNorm branch on it. `eval()` does NOT disable gradients — that's `no_grad`'s job (Day 3.1).
- **`nn.Linear`'s actual default init** (from the source you'll read): weight `kaiming_uniform_(a=√5)`, which algebraically reduces to `U(−1/√fan_in, +1/√fan_in)`; bias the same bound. Tomorrow you'll see where such formulas come from.

**Read (30–45 min):**
- `nn.Module` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html — read `parameters`, `named_parameters`, `register_buffer`, `state_dict`, `train`.
- The real source of Linear (short — read all of it, especially `reset_parameters`): https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
- "What is a state_dict" recipe: https://docs.pytorch.org/tutorials/recipes/recipes/what_is_state_dict.html

**Build:**
1. Create `tlib/modules.py`. Subclassing `nn.Module` is allowed — the point is the compute and the init, not reimplementing the registry:
```python
import math, torch
import torch.nn as nn

class Linear(nn.Module):
    """y = x @ W.T + b, matching nn.Linear semantics.
    weight: Parameter of shape (out_features, in_features)
    bias:   Parameter of shape (out_features,) or None
    Default init: weight, bias ~ U(-k, k) with k = 1/sqrt(in_features)."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True): ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...   # use x @ self.weight.T + self.bias; no F.linear

class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """max(x, 0) — implement via torch.clamp_min or torch.where, not F.relu."""

class Sequential(nn.Module):
    """Chains modules in order. Accept *modules; register each via add_module(str(i), m)
    so .parameters() and state_dict() see them; forward folds x through all."""
    def __init__(self, *modules: nn.Module): ...
    def forward(self, x): ...
```
   For bias=False, set `self.bias = None` — registering `None` is fine and `state_dict` skips it. Wrap init in `with torch.no_grad():` or use `nn.init.uniform_`-style in-place ops on `.data`-free Parameters via `torch.nn.Parameter(torch.empty(...).uniform_(-k, k))`.
2. Create `week04/d1_explore.py`: build `net = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))` and print `list(net.named_parameters())` names, `net.state_dict().keys()`, and `net.training` before/after `net.eval()`. Add a buffer to a scratch module and confirm it appears in `state_dict()` but NOT in `.parameters()`.

**Verify — done when:**
```python
torch.manual_seed(0)
mine, ref = Linear(16, 4), nn.Linear(16, 4)
with torch.no_grad():
    mine.weight.copy_(ref.weight); mine.bias.copy_(ref.bias)
x = torch.randn(7, 16)
assert torch.allclose(mine(x), ref(x), atol=1e-6)

net = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
assert len(list(net.parameters())) == 4            # 2 weights + 2 biases
assert sum(p.numel() for p in net.parameters()) == 4*8 + 8 + 8*2 + 2   # 90
assert set(net.state_dict()) == {"0.weight", "0.bias", "2.weight", "2.bias"}

# init distribution sanity: |values| bounded by 1/sqrt(in_features)
w = Linear(100, 50).weight
assert w.abs().max().item() <= 1/math.sqrt(100) + 1e-8
```
- Backward also flows: `net(x).sum().backward()` leaves no parameter with `grad is None`.

**If stuck:**
- `.parameters()` empty → you stored modules in a plain Python list (invisible to `__setattr__`); use `add_module` or `nn.ModuleList`.
- In-place init on a Parameter raises (Day 3.1 exp09!) → do it under `torch.no_grad()` or before wrapping in `Parameter`.

## Day 4.2 — Initialization from first principles (~2.5h)
- [ ] done

**Goal:** Derive Xavier and Kaiming initialization from the variance-propagation argument, implement both from the formulas, and demonstrate experimentally that N(0,1) init destroys a 20-layer network while yours keeps it alive.

**Learn:**
- **The variance-propagation argument.** For `y = W x` with i.i.d. zero-mean weights and `fan_in` inputs, each output is a sum of `fan_in` terms, so `Var(y) = fan_in · Var(w) · Var(x)`. Stack L layers and the activation variance gets multiplied by `fan_in · Var(w)` per layer — anything ≠ 1 compounds exponentially: explode or vanish. Initialization is choosing `Var(w)` to make that factor 1.
- **Xavier/Glorot** balances forward AND backward passes (backward propagates variance with `fan_out`), compromising on `Var(w) = 2/(fan_in + fan_out)`. Uniform version: `U(±√(6/(fan_in+fan_out)))` (the 6 = 3·2 because `Var(U(−a,a)) = a²/3`). Designed for symmetric, roughly-linear-at-0 activations like tanh.
- **Kaiming/He** accounts for ReLU killing half the variance (zeroing negatives halves E[x²]), so it doubles the weight variance: `Var(w) = 2/fan_in`, i.e. normal with `std = √(2/fan_in)`. The general form is `std = gain/√fan` with mode `fan_in` (preserve forward) or `fan_out` (preserve backward).
- **Gain** is the per-activation correction factor: `gain(linear)=1`, `gain(relu)=√2`, `gain(tanh)=5/3` (an empirical fit to tanh's variance shrinkage near its linear regime).
- **Why biases start at 0:** they add no variance and zero is the symmetric choice; symmetry-breaking comes entirely from random weights.

**Read (30–45 min):**
- UDL book, chapter 7 — the initialization section with the variance derivation and the exploding/vanishing figure: https://udlbook.github.io/udlbook/
- Glorot & Bengio 2010, "Understanding the difficulty of training deep feedforward neural networks" — abstract + §4: https://proceedings.mlr.press/v9/glorot10a.html
- He et al. 2015, "Delving Deep into Rectifiers" — abstract + §2.2 (the derivation): https://arxiv.org/abs/1502.01852
- `torch.nn.init` doc page (to compare conventions, NOT to call): https://docs.pytorch.org/docs/stable/nn.init.html

**Build:**
1. Create `tlib/init.py` — formulas only, no `torch.nn.init` calls:
```python
import math, torch

def _fan_in_fan_out(w: torch.Tensor) -> tuple[int, int]:
    """For a 2D weight of shape (out_features, in_features):
    fan_in = in_features, fan_out = out_features."""

def kaiming_normal_(w: torch.Tensor, gain: float = math.sqrt(2.0),
                    mode: str = "fan_in") -> torch.Tensor:
    """In-place fill: w ~ N(0, std^2), std = gain / sqrt(fan).
    Returns w. Must run under torch.no_grad() internally."""

def xavier_uniform_(w: torch.Tensor, gain: float = 1.0) -> torch.Tensor:
    """In-place fill: w ~ U(-a, a), a = gain * sqrt(6 / (fan_in + fan_out))."""
```
2. Create `week04/d2_depth_experiment.py`: build a 20-layer MLP, width 256, tanh activations, input `torch.randn(512, 256)`. Forward under `torch.no_grad()`, recording `x.std().item()` after every tanh. Run twice: (a) every weight ~ N(0,1); (b) every weight via `xavier_uniform_(w, gain=5/3)`. Print a 20-row table: `layer | std_naive | std_xavier`.
3. Bonus row: ReLU net with `kaiming_normal_` vs with `xavier_uniform_(gain=1)` — watch Xavier-for-ReLU slowly shrink (each ReLU costs √2 that Xavier doesn't repay).

**Verify — done when:**
```python
# distribution-level checks (large tensor -> tight statistics)
w = torch.empty(2000, 1000)
kaiming_normal_(w)
assert abs(w.std().item() - math.sqrt(2/1000)) < 0.003   # std ~= 0.0447
xavier_uniform_(w)
a = math.sqrt(6/(1000+2000))
assert w.abs().max().item() <= a and w.abs().max().item() > 0.99*a
```
- In the experiment table, the naive N(0,1) column saturates: tanh clamps everything to ±1, so per-layer std pins near 1.0 while pre-activations are astronomically large — assert pre-activation std at layer 20 exceeds 1e3 under (a). (With sigmoid-free tanh nets the failure shows as saturation, not literal overflow — say so in a comment.)
- Under (b), layer-20 activation std stays in a sane band: assert `0.2 < std20 < 1.5`.
- Gradient check of the claim: run loss=`out.pow(2).mean()` backward through both nets and compare `net[0].weight.grad.abs().mean()` — naive init's first-layer gradient is degenerate (orders of magnitude off), yours is healthy. Print both.

**If stuck:**
- He 2015 §2.2 is two readable paragraphs — the ½ from ReLU is spelled out there.
- If your kaiming std assert fails by ~√2: you used the wrong `fan` for the mode, or applied gain twice.

## Day 4.3 — Losses from scratch (~2h)
- [ ] done

**Goal:** Implement `mse_loss`, a log-sum-exp-stable `bce_with_logits`, and `cross_entropy` with reductions and label smoothing, matching `torch.nn.functional` under extreme inputs.

**Learn:**
- **MSE:** `mean((pred − target)²)`. The only subtlety is the reduction convention: torch's `"mean"` averages over ALL elements, not just the batch dim.
- **Why BCE-on-probabilities is broken:** `−[y·log σ(x) + (1−y)·log(1−σ(x))]` computes `log` of a saturated sigmoid — `σ(20)` rounds to 1.0 in float32 and `log(1−1.0) = −inf`. The fix is algebra, not clamping.
- **The stable BCE-with-logits form.** Substitute σ and simplify: `loss = max(x, 0) − x·y + log(1 + exp(−|x|))`. Derivation sketch you should reproduce on paper: `−log σ(x) = log(1+e^{−x})` and `−log(1−σ(x)) = log(1+e^{x}) = x + log(1+e^{−x})`; combine with weights y and 1−y, then use `log(1+e^{x}) = max(x,0) + log(1+e^{−|x|})` so the exponent is never positive.
- **Cross-entropy = log-softmax + NLL.** Your Week-1 stable form: `CE_i = logsumexp(z_i) − z_i[y_i]`. New today: `reduction="mean"|"sum"`, and label smoothing.
- **Label smoothing** replaces the one-hot target with `(1−ε)·onehot + ε/K · uniform`. Per-sample loss becomes `(1−ε)·(−log p_y) + (ε/K)·Σ_j(−log p_j)`. It penalizes over-confident logits; ε is typically 0.1.

**Read (30–45 min):**
- `F.binary_cross_entropy_with_logits` doc (read the stated formula and the warning on the non-logits version): https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
- `nn.CrossEntropyLoss` doc — the exact label_smoothing formula is in the doc text: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Szegedy et al. 2016 (where label smoothing comes from), abstract + §7: https://arxiv.org/abs/1512.00567

**Build:**
1. Create `tlib/losses.py`:
```python
def mse_loss(pred: torch.Tensor, target: torch.Tensor,
             reduction: str = "mean") -> torch.Tensor:
    """((pred-target)**2) with reduction in {"mean","sum","none"}."""

def bce_with_logits(logits: torch.Tensor, target: torch.Tensor,
                    reduction: str = "mean") -> torch.Tensor:
    """Stable elementwise: max(x,0) - x*y + log1p(exp(-|x|)).
    target in [0,1], same shape as logits. Use torch.log1p."""

def cross_entropy(logits: torch.Tensor, target: torch.Tensor,
                  reduction: str = "mean", label_smoothing: float = 0.0) -> torch.Tensor:
    """logits (N,K) float, target (N,) int64 class indices.
    log_p = logits - logsumexp(logits, dim=1, keepdim=True)   # your W1 stable form
    nll   = -log_p[arange(N), target]
    smooth= -log_p.mean(dim=1)            # = (1/K)*sum_j -log p_j
    per   = (1-eps)*nll + eps*smooth
    then reduce."""
```
2. `week04/d3_losses_test.py` — pytest comparisons vs `torch.nn.functional` (`F.mse_loss`, `F.binary_cross_entropy_with_logits`, `F.cross_entropy`), each at `atol=1e-6`:
   - random inputs, all three reductions;
   - **extreme logits:** `logits = torch.tensor([±30., ±60.])` style for BCE — assert your result is finite AND matches F; also assert the naive `-(y*torch.log(torch.sigmoid(x)) + ...)` formula produces `inf`/`nan` here (the motivation, demonstrated);
   - CE with `label_smoothing=0.1` on `(8, 5)` logits vs `F.cross_entropy(..., label_smoothing=0.1)`;
   - the mathematical anchor: CE of uniform logits (`torch.zeros(4, K)`) equals `ln(K)` exactly — assert `torch.allclose(cross_entropy(torch.zeros(4, 7), targets), torch.log(torch.tensor(7.0)))` for any targets.
   - gradients too: `logits.requires_grad_(); your_loss.backward()` vs same with F — `assert torch.allclose(g_yours, g_torch, atol=1e-6)`.

**Verify — done when:** `pytest week04/d3_losses_test.py -q` passes, including the extreme-logit and label-smoothing cases and at least one gradient comparison per loss.

**If stuck:**
- BCE off by a sign at negative x → you simplified `max(x,0) − x·y` wrong; re-derive from `log(1+e^x) = max(x,0)+log(1+e^{−|x|})`.
- Label smoothing mismatch → torch smooths over ALL K classes including the true one; don't exclude `y` from the uniform term.

## Day 4.4 — Optimizers I: base class + SGD with momentum (~2.5h)
- [ ] done

**Goal:** Build the `Optimizer` base class and an SGD (momentum + weight decay) that tracks `torch.optim.SGD` parameter-for-parameter over 50 steps.

**Learn:**
- **What an optimizer object actually is:** a list of `param_groups` (each a dict: `params` + hyperparameters like `lr`), a `state` dict mapping each parameter to its per-param buffers (momentum etc.), `zero_grad()`, and `step()` which mutates `p` in place under `no_grad` (Day 3.1 exp09 is why the `no_grad` is mandatory).
- **Param groups** exist so different parts of a model can get different hyperparameters (classic use: no weight decay on biases/norm layers). Accept either an iterable of params (one group) or a list of group dicts.
- **L2 weight decay as implemented by torch's SGD** is a gradient modification: `g ← g + wd·p` BEFORE the momentum buffer sees it (this detail matters; AdamW exists because of it — Day 5).
- **Torch's momentum convention** (different from the textbook `v ← μv − lr·g`!):
  `buf ← μ·buf + g` (no `(1−μ)` damping by default; first step: `buf = g`), then `p ← p − lr·buf`. Same trajectory family, different lr scaling — copy torch's exact form or lockstep parity will fail.
- **Why `step()` reads `.grad` and nothing else:** the optimizer is decoupled from the graph; any sequence of ops that left gradients in `.grad` works. This is the contract that makes the training loop composable.

**Read (30–45 min):**
- `torch.optim` doc page, top sections ("How to use an optimizer", per-parameter options) and the `SGD` algorithm box, which states the exact update incl. the momentum detail: https://docs.pytorch.org/docs/stable/optim.html
- `torch.optim.SGD` page (read the Note explaining how torch's momentum differs from Sutskever's): https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html

**Build:**
1. Create `tlib/optim.py`:
```python
class Optimizer:
    def __init__(self, params, defaults: dict):
        """params: iterable of Tensors OR list of {"params": [...], **overrides} dicts.
        Normalize to self.param_groups, each group = {**defaults, **overrides}.
        self.state: dict[Tensor, dict] = {} (keyed by parameter object)."""
    def zero_grad(self) -> None:
        """Set p.grad = None for all params (cheaper than zeroing; matches
        torch's set_to_none=True default)."""
    @torch.no_grad()
    def step(self) -> None:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, params, lr: float, momentum: float = 0.0,
                 weight_decay: float = 0.0): ...
    @torch.no_grad()
    def step(self) -> None:
        """Per param p with p.grad is not None:
        g = p.grad
        if weight_decay: g = g + weight_decay * p
        if momentum:
            buf = state.get("momentum_buffer")  -> first step: buf = g.clone()
            else: buf.mul_(momentum).add_(g)
            g = buf
        p.add_(g, alpha=-lr)"""
```
2. `week04/d4_sgd_parity.py` — the lockstep-parity protocol you'll reuse all week:
```python
def clone_model(src: nn.Module) -> nn.Module:
    dst = copy.deepcopy(src)          # identical weights, independent storage
    return dst

def lockstep(opt_mine, opt_ref, model_mine, model_ref, steps=50):
    torch.manual_seed(0)
    for t in range(steps):
        x, y = torch.randn(32, 16), torch.randn(32, 4)
        for model, opt in ((model_mine, opt_mine), (model_ref, opt_ref)):
            opt.zero_grad()
            loss = tlib.losses.mse_loss(model(x), y)
            loss.backward()
            opt.step()
        for pm, pr in zip(model_mine.parameters(), model_ref.parameters()):
            assert torch.allclose(pm, pr, atol=1e-6), f"diverged at step {t}"
```
   Run it for SGD with three hyperparameter sets: `(lr=0.1)`, `(lr=0.05, momentum=0.9)`, `(lr=0.05, momentum=0.9, weight_decay=1e-2)` vs `torch.optim.SGD` with identical args. Model: your `Sequential(Linear(16,32), ReLU(), Linear(32,4))`.

**Verify — done when:** `pytest week04/d4_sgd_parity.py -q` — all three configurations stay allclose for all 50 steps. If plain SGD passes but momentum diverges at step 2, you implemented textbook momentum instead of torch's (see Learn bullet 4).

**If stuck:**
- Divergence only WITH weight_decay → you applied decay after the momentum buffer, or decayed via `p.mul_` instead of through the gradient.
- The `SGD` doc page's pseudocode block is the ground truth — transcribe it literally.

## Day 4.5 — Optimizers II: RMSprop, Adam, AdamW (~2.5h)
- [ ] done

**Goal:** Implement the three adaptive optimizers with exact bias correction, pass lockstep parity for all, and visualize why bias correction exists.

**Learn:**
- **RMSprop:** keep an EMA of squared gradients, `v ← α·v + (1−α)·g²`, and normalize: `p ← p − lr · g / (√v + ε)`. Per-coordinate adaptive step: coordinates with consistently large gradients get smaller effective lr. Torch default α=0.99; note torch adds ε OUTSIDE the sqrt.
- **Adam = momentum + RMSprop + bias correction.** `m ← β₁m + (1−β₁)g` (EMA of gradients), `v ← β₂v + (1−β₂)g²`. Both start at 0, so early EMAs are biased toward 0 by exactly a factor `(1−βᵗ)`; correct with `m̂ = m/(1−β₁ᵗ)`, `v̂ = v/(1−β₂ᵗ)`. Update: `p ← p − lr·m̂/(√v̂ + ε)`. Defaults β₁=0.9, β₂=0.999, ε=1e-8. Step counter `t` starts at 1.
- **Adam+L2 vs AdamW.** Adding `wd·p` into `g` (Adam's `weight_decay`) feeds the decay through `m` and `v` — so the decay gets DIVIDED by `√v̂`, making it weaker exactly where gradients are large. AdamW decouples it: decay acts directly on the parameter, `p ← p − lr·wd·p`, applied alongside (in torch: before) the Adam update. Result: weight decay strength becomes independent of gradient scale — the reason AdamW is the transformer-era default.
- **Why bias correction matters in practice:** without it, `v` is tiny for the first ~`1/(1−β₂)` steps, so `g/√v` is huge — early steps are erratic unless lr is warmed up. Correction makes step 1 already well-scaled.

**Read (30–45 min):**
- Kingma & Ba 2015, "Adam" — Algorithm 1 box + §2 (initialization bias): https://arxiv.org/abs/1412.6980
- Loshchilov & Hutter 2019, "Decoupled Weight Decay Regularization" — abstract + Algorithm 2: https://arxiv.org/abs/1711.05101
- Algorithm boxes on the torch pages (your parity ground truth): https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html and https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html

**Build:**
1. Add to `tlib/optim.py` (all share the `Optimizer` base; per-param state holds `step`, `exp_avg`, `exp_avg_sq` — match torch's state names if you want `state_dict` interop later):
```python
class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0.0): ...

class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.0): ...
    # weight_decay here = L2-into-gradient: g = g + wd*p BEFORE moments

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2): ...
    # decoupled: p.mul_(1 - lr*wd) first, then the Adam step with raw g
```
2. Extend `week04/d4_sgd_parity.py` (or new `d5_adaptive_parity.py`): lockstep 50 steps vs `torch.optim.RMSprop`, `torch.optim.Adam`, `torch.optim.AdamW`, each with nonzero `weight_decay` so the Adam-vs-AdamW distinction is actually exercised. Add one NEGATIVE test: Adam-with-L2 vs torch AdamW at the same wd must DIVERGE (assert not allclose by step 50) — proof the distinction is real, not notation.
3. Bias-correction visualization, `week04/d5_bias_correction.py`: single parameter, constant gradient `g = 1.0`. Run 20 Adam steps twice — once with bias correction, once with `m̂ = m, v̂ = v` — recording effective step size `|Δp|/lr` each step. Plot both curves (matplotlib, save `week04/bias_correction.png`). With correction the curve starts at approximately 1.0 and stays flat; without, it starts near `(1−β₁)/√(1−β₂)` ≈ 0.1/0.0316 ≈ 3.16... — actually compute and annotate rather than trust this sentence: assert corrected step 1 is within 10% of 1.0 and uncorrected step 1 differs from corrected by more than 2x.

**Verify — done when:**
- All three lockstep parities hold for 50 steps at `atol=1e-6`.
- The negative test (Adam+L2 ≠ AdamW) fails-as-expected.
- The plot shows the corrected curve flat near 1.0 from step 1 and the uncorrected one starting far from it before converging — the two curves visibly merge as `t` grows (`βᵗ → 0`).

**If stuck:**
- Parity off by a tiny constant factor everywhere → ε inside vs outside the sqrt; check the torch algorithm box for each optimizer separately (they differ!).
- Off only at early steps → `t` starting at 0 instead of 1 in bias correction.

## Day 4.6 — Deep build: FashionMNIST end-to-end on YOUR stack (~3.5h)
- [ ] done

**Goal:** Train a 784→256→128→10 MLP on FashionMNIST using ONLY tlib modules/init/losses/optim, hit approximately 85% test accuracy on CPU, and prove save/resume works via loss continuity.

**Learn:**
- **The dataset.** FashionMNIST: 60k train / 10k test grayscale 28×28 images, 10 clothing classes — a drop-in MNIST replacement that isn't saturated. `torchvision.datasets.FashionMNIST(root, train=, download=True, transform=ToTensor())` gives `(C,H,W)` floats in [0,1].
- **Minimal batching without DataLoader.** Week 5 builds Dataset/DataLoader properly; today, load the whole training set as one tensor (`(60000, 784)` after flatten — ~180MB float32, fine on CPU), shuffle indices each epoch with `torch.randperm`, slice batches. Normalize with the dataset's own statistics: subtract mean, divide std, computed from the training tensor.
- **The bare training loop liturgy:** for each batch — `opt.zero_grad()`, forward, loss, `loss.backward()`, `opt.step()`. Evaluation: `model.eval()` + `torch.inference_mode()` (Day 3.1), argmax over logits, compare to labels.
- **Checkpointing = state_dicts, plural.** A resumable checkpoint needs the model's `state_dict` AND the optimizer's state (momentum/Adam moments). Losing optimizer state makes the first resumed steps behave like a cold restart. (Your Day-4 `Optimizer` keeps state keyed by parameter object — for today, serialize it by parameter INDEX into a dict you can reload; matching torch's `state_dict()` format is optional.)

**Read (30–45 min):**
- `torchvision.datasets.FashionMNIST`: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.FashionMNIST.html
- Saving & loading models tutorial — "Save/Load state_dict" and the checkpoint section: https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

**Build:**
1. `uv add torchvision`
2. Create `week04/fashion_train.py`:
```python
def load_fashion(root: str = "data") -> tuple[torch.Tensor, ...]:
    """Returns Xtr (60000,784) float32 normalized, ytr (60000,) int64,
    Xte, yte likewise (normalized with TRAIN stats)."""

def make_model(seed: int = 0) -> nn.Module:
    """Sequential(Linear(784,256), ReLU(), Linear(256,128), ReLU(), Linear(128,10))
    — tlib.modules classes only. Re-init weights with tlib.init.kaiming_normal_
    (fan_in, relu gain) and zero biases. seed via tlib.utils.seed_all."""

def evaluate(model, X, y, batch_size: int = 1000) -> float:
    """Accuracy under inference_mode + eval()."""

def train(epochs: int = 10, batch_size: int = 128, lr: float = 1e-3,
          resume_from: str | None = None) -> None:
    """AdamW(lr=lr, weight_decay=1e-4) from tlib.optim, tlib.losses.cross_entropy.
    Per epoch: shuffle, iterate batches, track running mean loss.
    Print 'epoch E | train_loss L | test_acc A' each epoch.
    Save checkpoint (model state_dict, optimizer state by param index,
    epoch, last train_loss) to week04/ckpt.pt every epoch via torch.save."""
```
3. Resume proof, `test_resume` in the same file: train 2 epochs saving per-BATCH losses of epoch 2's last 50 batches; checkpoint; reconstruct model+optimizer fresh, load checkpoint, train 1 more epoch recording its first 50 batch losses. Assert `mean(first_50_after_resume) < mean(last_50_before) + 0.05` — resumed loss continues from where it left off rather than spiking back toward `ln(10) ≈ 2.303` (the from-scratch starting loss — assert your epoch-1 first-batch loss is approximately 2.0–2.6 as a sanity anchor).
4. While training, watch one epoch's time. GPU variant note: this entire script needs only `device = tlib.utils.get_device()` and `.to(device)` on model and batches to run on CUDA — but CPU finishes 10 epochs in roughly a few minutes at this model size; don't optimize.

**Verify — done when:**
```bash
python week04/fashion_train.py          # trains 10 epochs
pytest week04/fashion_train.py -q       # resume test passes
```
- Test accuracy after 10 epochs is approximately 0.85–0.89 with these settings; the script asserts `acc >= 0.80`.
- First-batch loss ≈ ln(10) (2.0–2.6 band) — your init and CE are honest.
- The resume test passes; deleting the optimizer state from the checkpoint (try it once) makes the post-resume losses visibly worse for the first batches — observe, then restore.
- Everything model/loss/optim-related imports from `tlib` — grep the file for `torch.nn.functional` and `torch.optim`: only allowed hit is `nn.Module` inheritance inside tlib itself.

**If stuck:**
- Accuracy stuck at 0.10 → labels/logits misaligned (check `cross_entropy(logits, target)` argument order) or you normalized test data with test stats.
- Loss is `nan` in epoch 1 → lr too high for un-normalized inputs; confirm the normalization actually ran (train tensor mean ≈ 0, std ≈ 1 after).

## Day 4.7 — Review, quiz, redo-cold (~2h)
- [ ] done

**Goal:** Consolidate modules/init/losses/optimizers and prove retention by rebuilding the load-bearing pieces from memory.

**Learn (review framing):**
- The week in one sentence each: Modules are parameter REGISTRIES around forward math; init keeps the variance product near 1; stable losses are algebra that never exponentiates a large positive number; optimizers are state machines over `.grad`.
- Map each tlib file to its torch counterpart and recall ONE convention detail you had to match (Linear's `U(±1/√fan_in)`; momentum's missing `(1−μ)`; ε outside the sqrt; AdamW's decoupling).
- Foreshadow: Week 5 wraps today's manual batching into Dataset/DataLoader and a reusable Trainer; Week 6 adds BatchNorm/LayerNorm (your first real buffers) and lr schedules (mutating `param_groups["lr"]` — you built the hook for that today).

**Read (30–45 min):** Re-read the torch `optim` doc's algorithm boxes for SGD/Adam/AdamW one final time WITHOUT taking notes (https://docs.pytorch.org/docs/stable/optim.html), then immediately do redo-cold drill 1.

**Redo cold (no peeking at your own code or notes):**
1. Write the full Adam `step()` from memory — both EMAs, bias correction with `t`, the update. Then diff against `tlib/optim.py`.
2. Re-derive the Kaiming variance argument on paper: start from `Var(y) = fan_in·Var(w)·E[x²]`, insert ReLU's ½, conclude `Var(w) = 2/fan_in`.
3. Rewrite `bce_with_logits`'s stable formula from memory, then verify on `logits=torch.tensor([60., -60.])`, `target=torch.tensor([0., 1.])` against F.
4. From memory: which two things must a resumable checkpoint contain at minimum, and what goes wrong with each missing?
5. Build and train a fresh 2-layer net on 200 random points to overfit (loss → ~0) using only tlib, in under 15 minutes, from an empty file.

**Self-quiz:**
1. Mechanically, how does `self.weight = nn.Parameter(...)` end up in `.parameters()` without any explicit registration call?
2. Name two differences between a buffer and a parameter, and give the canonical example of a buffer.
3. What does `model.eval()` actually change, and name one thing people wrongly expect it to change.
4. Derive in two lines why `Var(w) = 1/fan_in` preserves forward activation variance for a linear layer, and where ReLU's extra factor 2 comes from.
5. Why does Xavier init average fan_in and fan_out instead of just using fan_in?
6. Show the algebra step that makes `log(1 + exp(−|x|))` safe for all x, where the naive BCE blows up.
7. What target distribution does label smoothing ε with K classes train toward, and what is the per-sample loss formula?
8. Write torch's SGD-with-momentum update equations exactly, and name the difference from the classical `v ← μv − lr·g` form.
9. Why is Adam's `m` biased toward zero early, and what is the exact correction factor at step t?
10. A colleague says "AdamW is just Adam with weight_decay > 0." Correct them precisely: where does each variant's decay term enter, and why does it produce different effective regularization?
11. Your FashionMNIST net's first-batch loss is 4.7 instead of ≈2.3. Name two likely bugs this single number points to.
12. Why must `optimizer.step()` run under `torch.no_grad()` (two reasons — one is an error from Week 3, one is graph pollution)?

---

### Answers

1. `nn.Module.__setattr__` is overridden: when the assigned value is an `nn.Parameter` it is stored in `self._parameters` (and Modules in `self._modules`). `.parameters()` recursively walks these dicts through all children.
2. Parameters are returned by `.parameters()` and updated by optimizers; buffers are not optimized but ARE saved in `state_dict` and moved by `.to(device)`. Canonical buffer: BatchNorm's `running_mean`/`running_var`.
3. It recursively sets `self.training = False`, changing the behavior of mode-dependent layers (Dropout passes through, BatchNorm uses running stats). It does NOT disable gradient tracking — that's `no_grad`/`inference_mode`.
4. `y = Σᵢ wᵢxᵢ` over fan_in i.i.d. terms ⇒ `Var(y) = fan_in·Var(w)·E[x²]`; setting `fan_in·Var(w) = 1` keeps variance constant. ReLU zeroes the negative half of a symmetric input, halving `E[x²]`, so weights must carry `Var(w) = 2/fan_in` to compensate.
5. Forward signal variance propagates with fan_in but backward gradient variance propagates with fan_out; no single `Var(w)` satisfies both unless fan_in = fan_out, so Glorot compromises with the harmonic-style mean `2/(fan_in+fan_out)`.
6. Naive BCE computes `log(σ(x))` or `log(1−σ(x))`, which is `log` of exactly 0 once σ saturates in float. Using `log(1+e^x) = max(x,0) + log(1+e^{−|x|})` rewrites every log term so the exponential's argument is `−|x| ≤ 0` — `exp` can only underflow to 0 (harmless inside log1p), never overflow.
7. Target: `(1−ε)` on the true class plus `ε/K` on every class (including the true one). Loss: `(1−ε)·(−log p_y) + (ε/K)·Σⱼ(−log p_j)`.
8. `g ← g + wd·p`; `buf ← μ·buf + g` (first step `buf = g`); `p ← p − lr·buf`. Difference: classical momentum scales the gradient term by lr inside the velocity (`v ← μv − lr·g`), torch scales the whole buffer by lr at apply time and has no `(1−μ)` damping — same family, different effective lr per μ.
9. `m` starts at 0 and is an EMA, so after t steps it has only accumulated weight `1−β₁ᵗ` of the true mean (the rest still "remembers" the zero init). Correction: divide by `(1−β₁ᵗ)`; likewise `v` by `(1−β₂ᵗ)`.
10. Adam's `weight_decay` adds `wd·p` to the gradient BEFORE the moment EMAs, so the decay is later divided by `√v̂` — parameters with large gradient variance get LESS decay. AdamW applies `p ← p − lr·wd·p` directly, outside the adaptive normalization, so decay strength is uniform and independent of gradient statistics.
11. With balanced random init, expected initial CE is ≈ ln(10) ≈ 2.3. A value of 4.7 means the logits are far from uniform at init: likely (a) init scale far too large (no Kaiming, or gain misapplied), or (b) inputs not normalized so pre-activations are huge. (A wrong label/logit pairing also inflates it.)
12. (a) Parameters are leaves with `requires_grad=True`; in-place updates on them outside `no_grad` raise the Week-3 exp09 RuntimeError. (b) Even via workarounds, recording the update ops would graft the optimizer arithmetic onto the autograd graph, leaking memory and corrupting the next backward.
