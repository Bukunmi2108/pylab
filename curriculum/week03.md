# Week 3 — Autograd

Weeks 1–2 gave you tensors as data: shapes, strides, and the numerics of softmax and matmul. This week you learn how PyTorch turns a chain of tensor ops into gradients — first by poking the real autograd engine until you can predict every error message it throws, then by building your own reverse-mode engine (`Scalar`) from nothing and training a real classifier with it. Autograd is sequenced here because everything after this point (modules, optimizers, training loops) is just bookkeeping on top of `backward()`; once you have written a backward pass yourself, nothing in `torch.nn` will feel like magic.

**Week outcome:** `tlib/gradcheck.py` (a finite-difference gradient checker you trust), `tlib/engine.py` (a working scalar autograd engine with topological-sort backward), and `week03/mlp_engine.py` (an MLP trained on two-moons to ≥90% accuracy using *your* engine). Skill by day 7: you can explain the recorded DAG, vector-Jacobian products, and gradient accumulation precisely, and you can write a custom `torch.autograd.Function` that passes `gradcheck`.

## Day 3.1 — Autograd mechanics by experiment (~2.5h)
- [ ] done

**Goal:** Build a precise mental model of the autograd DAG by deliberately triggering (and asserting on) ten of its behaviors and errors.

**Learn:**
- **The recorded DAG.** Every op on a tensor with `requires_grad=True` records a node (`grad_fn`) holding the backward formula and references to its inputs' nodes. `backward()` walks this graph in reverse. The graph is rebuilt from scratch on every forward pass — it is a tape, not a static structure.
- **Leaf vs non-leaf.** A leaf is a tensor you created directly (e.g. `torch.randn(3, requires_grad=True)`); its `grad_fn` is `None` and `.grad` gets populated. Non-leaf tensors (results of ops) have a `grad_fn` and by default do *not* keep `.grad` — you must call `.retain_grad()` to see it.
- **Gradient accumulation.** `backward()` *adds into* `.grad`, it never overwrites. This is deliberate (it's how a node feeding two consumers gets the sum of both contributions) but it means you must zero grads between optimization steps or steps blend together.
- **Three ways to stop recording.** `torch.no_grad()` disables recording inside a block (results are normal tensors you can still use in a later graph). `torch.inference_mode()` is stricter and faster: its outputs can *never* re-enter autograd. `.detach()` returns a view of one tensor cut out of the graph (shares storage!).
- **The graph is freed after backward.** Intermediate buffers needed by backward are released the first time you call `backward()`; a second call raises a RuntimeError unless you passed `retain_graph=True`.
- **In-place ops and the version counter.** Each tensor carries a `_version` int bumped by in-place ops (`add_`, `relu_`, slice-assign). If backward needs a tensor that was modified after being saved, autograd raises "one of the variables needed for gradient computation has been modified by an inplace operation".

**Read (30–45 min):**
- Autograd mechanics note — https://docs.pytorch.org/docs/stable/notes/autograd.html — sections "How autograd encodes the history", "Locally disabling gradient computation" (incl. the No-grad vs Inference Mode table), and "In-place operations with autograd".
- `torch.Tensor.backward` and `torch.Tensor.detach` API docs — https://docs.pytorch.org/docs/stable/generated/torch.Tensor.backward.html

**Build:**
1. Create `week03/d1_experiments.py`. Write ten functions, `exp01_...` through `exp10_...`, each self-contained, each ending in asserts, each with a 1–3 line comment in YOUR OWN words explaining why the behavior occurs. Run all from `if __name__ == "__main__":` and also make the file pytest-collectable (name them `test_exp01_...` etc. if you prefer).
2. The ten experiments:
   - `exp01_leaf_vs_nonleaf`: `x = torch.randn(3, requires_grad=True); y = x * 2`. Assert `x.is_leaf`, `not y.is_leaf`, `x.grad_fn is None`, `y.grad_fn is not None`.
   - `exp02_grad_accumulates`: scalar `loss = (x**2).sum()`; call `loss.backward()` twice (rebuild the graph between calls by recomputing `loss`). Assert `torch.allclose(x.grad, 2*2*x.detach())` — proof of accumulation, and why `zero_grad()` exists.
   - `exp03_backward_twice_errors`: compute `loss` once, `loss.backward()`, then `with pytest.raises(RuntimeError): loss.backward()`. Then show `retain_graph=True` on the first call makes the second succeed.
   - `exp04_nonleaf_grad_is_none`: after backward, assert `y.grad is None` (catch/ignore the UserWarning); repeat with `y.retain_grad()` and assert `y.grad is not None`.
   - `exp05_no_grad_block`: inside `torch.no_grad()`, `z = x * 3`; assert `z.grad_fn is None` and `not z.requires_grad`.
   - `exp06_inference_mode_is_stricter`: create `z` under `torch.inference_mode()`; assert that using `z` in an op with a `requires_grad` tensor and calling backward raises `RuntimeError` ("Inference tensors..."). Contrast: the `no_grad` output from exp05 can be used freely.
   - `exp07_detach_shares_storage`: `d = x.detach(); d[0] = 99.0`; assert `x[0].item() == 99.0`. Comment on why this is a footgun.
   - `exp08_inplace_on_saved_tensor_errors`: `y = x * x; y2 = y.exp(); y.add_(1.0)` then `with pytest.raises(RuntimeError): y2.sum().backward()`. (`exp` saves its *output*... use `y2 = y.sin()` if your torch version doesn't raise — `sin` saves its input.) Assert the error mentions "inplace".
   - `exp09_inplace_on_leaf_errors`: `w = torch.randn(3, requires_grad=True)`; `with pytest.raises(RuntimeError): w.add_(1.0)` — a leaf requiring grad refuses in-place edits outside `no_grad`. Then show the sanctioned form: `with torch.no_grad(): w.add_(1.0)` succeeds (this is exactly what optimizers do).
   - `exp10_grad_fn_chain`: for `z = (x * 2 + 1).sum()`, walk `z.grad_fn`, then `z.grad_fn.next_functions`, printing the class names, until you hit `AccumulateGrad`. Assert the chain contains `"Sum"`, `"Add"`, `"Mul"` in the names, and that the terminal node is `AccumulateGrad` (the node that writes into a leaf's `.grad`).

**Verify — done when:**
```bash
pytest week03/d1_experiments.py -q   # all pass
```
- Every experiment has a why-comment you could defend out loud.
- You can answer without running code: "what does `.grad` hold after calling backward twice without zeroing?"

**If stuck:**
- Re-read "In-place correctness checks" in the autograd mechanics note — it explains the `_version` counter exactly.
- Print `tensor._version` before and after an in-place op to watch the counter move.

## Day 3.2 — backward() with vectors: vector–Jacobian products (~2h)
- [ ] done

**Goal:** Understand exactly what `backward(gradient=v)` computes and use repeated VJPs to assemble full Jacobians.

**Learn:**
- **Why scalar outputs are special.** `loss.backward()` works without arguments only because for a scalar output the "vector" is implicitly `1.0`. For a vector output `y ∈ ℝᵐ` there is no single gradient — there's an m×n Jacobian J, and autograd refuses to guess what you want.
- **What v·J means.** `y.backward(v)` computes `vᵀJ` — one row-combination of the Jacobian — in a single backward pass. Reverse-mode autodiff is a VJP machine: cheap when outputs are few (one scalar loss) and inputs are many (millions of weights). That asymmetry is the entire reason deep learning trains this way.
- **Recovering a full Jacobian.** Pass each standard basis vector `e_i` as `v`: `eᵢᵀJ` is row i of J. m backward passes ⇒ the whole m×n Jacobian.
- **`torch.autograd.grad`.** Functional alternative to `.backward()`: returns gradients instead of mutating `.grad`. Cleaner for repeated VJPs (no zeroing dance).
- **`create_graph=True`.** Makes the backward pass itself differentiable, so you can differentiate the gradient — second derivatives and beyond.

**Read (30–45 min):**
- `torch.autograd.grad` and `torch.autograd.backward` — https://docs.pytorch.org/docs/stable/autograd.html (sections at the top of the Autograd page).
- `torch.autograd.functional.jacobian` — same page, "Functional higher level API" section.
- d2l.ai on automatic differentiation (incl. the non-scalar backward discussion) — https://d2l.ai/chapter_preliminaries/autograd.html

**Build:**
1. Create `week03/d2_vjp.py` with:
```python
def jacobian_by_vjp(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Return the m x n Jacobian of f at x (x: shape (n,), f(x): shape (m,))
    by looping basis vectors e_i through torch.autograd.grad."""
```
   Implementation: `y = f(x)`; for each `i`, `v = torch.zeros_like(y); v[i] = 1.0`; `(gi,) = torch.autograd.grad(y, x, grad_outputs=v, retain_graph=True)`; stack rows.
2. Test it on three functions with `x = torch.randn(4, dtype=torch.float64, requires_grad=True)`:
   - `f1(x) = x ** 2` (Jacobian is `diag(2x)`)
   - `f2(x) = A @ x` for a fixed `A = torch.randn(3, 4, dtype=torch.float64)` (Jacobian is `A` exactly)
   - `f3(x) = torch.softmax(x, dim=0)`
3. Second derivative: with scalar `x = torch.tensor(3.0, requires_grad=True)`, compute `y = x**3`, then `(g,) = torch.autograd.grad(y, x, create_graph=True)`, then `(g2,) = torch.autograd.grad(g, x)`. Assert `g.item() == 27.0` (3x² at x=3) and `g2.item() == 18.0` (6x at x=3). Avoid testing at `x = 2.0` — there 3x² and 6x are BOTH 12, so a swapped implementation would still pass; x=3 actually discriminates.

**Verify — done when:**
```python
from torch.autograd.functional import jacobian as torch_jac
for f in (f1, f2, f3):
    x = torch.randn(4, dtype=torch.float64, requires_grad=True)
    assert torch.allclose(jacobian_by_vjp(f, x), torch_jac(f, x), atol=1e-9)
assert torch.allclose(jacobian_by_vjp(f2, x), A)         # exact structure check
# second derivatives: at x=3.0 -> first grad 27.0, second 18.0 (3x^2 and 6x)
```

**If stuck:**
- The error "grad can be implicitly created only for scalar outputs" is the doorway: re-read the `torch.autograd.backward` doc on the `grad_tensors` argument.
- If your second `grad` call fails, you forgot `create_graph=True` on the first.

## Day 3.3 — Your own gradient checker (~2h)
- [ ] done

**Goal:** Write a finite-difference gradient checker in `tlib/` and learn why float64 and a good relative-error metric are non-negotiable.

**Learn:**
- **Central differences.** `df/dx_i ≈ (f(x + ε·e_i) − f(x − ε·e_i)) / (2ε)`. Central (not forward) differences cancel the first-order error term, giving O(ε²) accuracy instead of O(ε) — dramatically better for the same ε.
- **Choosing ε.** Too large → truncation error (the Taylor remainder); too small → catastrophic cancellation when subtracting two nearly equal floats. The sweet spot is around the cube root of machine epsilon: ~1e-5..1e-6 for float64. For float32 (machine eps ≈ 1.2e-7) there is almost no good ε — which is why gradient checking is done in float64.
- **Relative error, not absolute.** Compare `|a − n| / max(|a|, |n|, tiny)` where `a` is autograd's gradient and `n` the numeric one. An absolute diff of 1e-4 is alarming for gradients of size 1e-3 and meaningless for gradients of size 1e3.
- **Check one scalar function of many inputs.** A checker for `f: ℝⁿ → ℝ` is all you need: to check a vector-valued op, compose it with a sum or a random linear functional.

**Read (30–45 min):**
- `torch.autograd.gradcheck` doc (read what torch's own checker does — eps default, dtype warnings): https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html
- UDL book, chapter 7 (Gradients and initialization) — the backpropagation sections, for context on why we trust-but-verify: https://udlbook.github.io/udlbook/

**Build:**
1. Create `tlib/gradcheck.py`:
```python
import torch
from typing import Callable

def numeric_grad(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    """Central-difference gradient of scalar-valued f at x.
    Returns a tensor shaped like x. Does NOT use autograd.
    Works on a float64 clone of x; perturbs one element at a time."""

def check_gradients(f: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor,
                    rtol: float = 1e-5, eps: float = 1e-6) -> bool:
    """Compare autograd's gradient of f at x against numeric_grad.
    Computes max relative error  |a-n| / max(|a|,|n|,1e-12)  elementwise.
    Returns True if max rel err < rtol; on failure raises AssertionError
    with the max error and offending index in the message."""
```
2. In `week03/d3_check.py`, run `check_gradients` on five functions at `x = torch.randn(6, dtype=torch.float64, requires_grad=True)`:
   - `lambda x: (x ** 3).sum()`
   - `lambda x: torch.tanh(x).sum()`
   - `lambda x: (x @ x)` (dot product with itself)
   - `lambda x: torch.logsumexp(x, dim=0)`
   - your Week-1 `stable_softmax` composed with a weighted sum: `lambda x: (stable_softmax(x) * w).sum()` for fixed random `w` (a plain `.sum()` of softmax is constant 1 — gradient ≈ 0 everywhere, a useless check; the weighted sum makes it real).
3. Float32 failure demo: pick `f(x) = torch.exp(x).sum()` at `x = torch.randn(6) * 4` (values up to ~±8 so f is large). Show `check_gradients` with `rtol=1e-5` FAILS on the float32 input but PASSES on `x.double()`. Print both max relative errors.

**Verify — done when:**
```bash
pytest week03/d3_check.py -q
```
- All five float64 checks pass at `rtol=1e-5`.
- The float32 demo prints a relative error orders of magnitude worse than the float64 one (approximately 1e-2..1e-4 vs ≤1e-8 — exact values vary with the random draw).
- Sanity self-test: `numeric_grad(lambda x: (x**2).sum(), torch.tensor([3.0], dtype=torch.float64))` returns approximately `[6.0]`.

**If stuck:**
- If every check fails identically: you're probably mutating `x` itself rather than a clone — perturb `x2 = x.detach().clone()`.
- The `gradcheck` doc's note on float32 inputs says exactly why your float32 demo fails.

## Day 3.4 — Scalar engine part 1: ops and local gradients (~2.5h)
- [ ] done

**Goal:** Implement `Scalar` in `tlib/engine.py` — a node that stores data, a gradient slot, parent links, and a closure that knows its local backward rule.

**Learn:**
- **A node = value + recipe.** Each `Scalar` stores `data` (float), `grad` (float, starts 0.0), `_parents` (the Scalars it was computed from), `op` (a debug string like `"*"`), and `_backward` — a zero-argument closure that, when called, adds this node's contribution into its parents' `.grad`. This mirrors `grad_fn` + `next_functions` from Day 1.
- **The chain rule, locally.** If `out = f(a, b)`, then `a.grad += (∂out/∂a) · out.grad`. Every `_backward` is just this one line per parent. Note the `+=`: accumulation is what makes shared nodes correct (Day 5 tests this).
- **The local gradient table** (memorize these — they're the whole engine):
  - `out = a + b` → `∂out/∂a = 1`, `∂out/∂b = 1`
  - `out = a * b` → `∂out/∂a = b.data`, `∂out/∂b = a.data`
  - `out = a ** n` (n a plain int/float) → `∂out/∂a = n · a.data**(n−1)`
  - `out = relu(a)` → `1 if a.data > 0 else 0` (times `out.grad`)
  - `out = tanh(a)` → `1 − out.data²` (use the *output*, it's cheaper)
  - `out = exp(a)` → `out.data` (the output again)
  - `out = log(a)` → `1 / a.data`
- **Derived ops cost nothing.** `__neg__` is `self * -1`; `__sub__` is `self + (-other)`; `__truediv__` is `self * other**-1`; `__radd__`/`__rmul__` make `3 + x` work. Their gradients come for free through composition.

**Read (30–45 min):**
- "How autograd encodes the history" (re-read with builder's eyes — you are now writing `grad_fn`): https://docs.pytorch.org/docs/stable/notes/autograd.html
- UDL book ch. 7, the backpropagation-as-local-rules presentation: https://udlbook.github.io/udlbook/

**Build:**
1. Create `tlib/engine.py`:
```python
class Scalar:
    """A node in a dynamically built computation graph over Python floats."""
    def __init__(self, data: float, _parents: tuple = (), op: str = ""):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._parents = set(_parents)
        self.op = op

    def __add__(self, other): ...   # coerce: other = other if isinstance(other, Scalar) else Scalar(other)
    def __mul__(self, other): ...
    def __pow__(self, n: int | float): ...   # n must NOT be a Scalar; assert that
    def __neg__(self): ...
    def __sub__(self, other): ...
    def __truediv__(self, other): ...
    def __radd__(self, other): ...
    def __rmul__(self, other): ...
    def __rsub__(self, other): ...
    def relu(self): ...
    def tanh(self): ...
    def exp(self): ...
    def log(self): ...
    def __repr__(self): return f"Scalar(data={self.data:.4g}, grad={self.grad:.4g})"
```
   Pattern for every primitive op (shown for `__mul__`):
```python
def __mul__(self, other):
    other = other if isinstance(other, Scalar) else Scalar(other)
    out = Scalar(self.data * other.data, (self, other), "*")
    def _backward():
        self.grad  += other.data * out.grad
        other.grad += self.data  * out.grad
    out._backward = _backward
    return out
```
2. You can't run full `backward()` yet (Day 5). For today, test each op's LOCAL rule by hand: build `out = a.op(b)`, set `out.grad = 1.0`, call `out._backward()`, and assert the parents' grads match the table. Put these in `week03/d4_ops_test.py` — one test per op (add, mul, pow, neg, sub, div, relu both sides of 0, tanh, exp, log).
3. Cross-check two of them against torch: e.g. for `tanh`, `t = torch.tensor(0.7, requires_grad=True); torch.tanh(t).backward()`; assert your parent grad equals `t.grad.item()` to within 1e-9.
4. Cross-check the table itself with your Day-3 checker: e.g. `check_gradients(lambda x: torch.tanh(x).sum(), torch.randn(3, dtype=torch.float64, requires_grad=True))` — confirming the formulas you hard-coded are the true derivatives.

**Verify — done when:**
```bash
pytest week03/d4_ops_test.py -q
```
- `relu` test covers a negative input (grad 0) and a positive one (grad 1).
- `(Scalar(3) / Scalar(4)).data == 0.75` and `(2 - Scalar(0.5)).data == 1.5` (right-ops work).
- `pow` raises (assert/TypeError) when given a `Scalar` exponent.

**If stuck:**
- Late-binding closure bug: if you write `_backward` in a loop or reuse variable names, the closure may capture the wrong variable. Define `_backward` immediately after computing `out`, referencing only `self`, `other`, `out`.
- Wrong sign in `__rsub__`? `2 - x` must be `(-x) + 2`, not `x - 2`.

## Day 3.5 — Scalar engine part 2: backward() and the diamond test (~2.5h)
- [ ] done

**Goal:** Implement full reverse-mode `backward()` via topological sort and prove gradient accumulation is correct on a diamond-shaped graph.

**Learn:**
- **Why topological order.** A node's `_backward` reads `out.grad` — so a node must not push gradient to its parents until ALL its consumers have pushed gradient to it. Visiting nodes in reverse topological order guarantees every node's `.grad` is final before it fires.
- **Topo sort by DFS post-order.** Recursively visit parents first, append self after — the resulting list has parents before children; iterate it reversed. A `visited` set prevents re-adding shared nodes.
- **Seeding.** `backward()` starts by setting `self.grad = 1.0` — this is the implicit `v = 1` of a scalar output from Day 2.
- **The diamond.** `x → a; out = f(a, a)` (one node feeding two consumers): a correct engine gives `a` the SUM of both consumers' contributions. This is exactly why Day 4's `_backward` used `+=`, and why torch accumulates `.grad`.

**Read (30–45 min):**
- Topological sorting (DFS post-order) — any standard reference; e.g. CLRS-style summary at https://en.wikipedia.org/wiki/Topological_sorting (section "Depth-first search").
- Skim `torch.autograd` package overview for `Function`/graph vocabulary: https://docs.pytorch.org/docs/stable/autograd.html

**Build:**
1. Add to `Scalar` in `tlib/engine.py`:
```python
def backward(self) -> None:
    """Reverse-mode AD from this node. Sets self.grad = 1.0, then calls
    every node's _backward in reverse topological order. Grads ACCUMULATE
    across calls (like torch); caller zeroes if needed."""
    topo: list[Scalar] = []
    visited: set[Scalar] = set()
    def build(node):
        if node not in visited:
            visited.add(node)
            for p in node._parents:
                build(p)
            topo.append(node)
    build(self)
    self.grad = 1.0
    for node in reversed(topo):
        node._backward()
```
2. In `week03/d5_backward_test.py`:
   - **Multi-node parity vs torch.** Build identically in both systems:
     ```python
     # engine                                  # torch
     a = Scalar(1.5); b = Scalar(-2.0)         ta = torch.tensor(1.5, requires_grad=True); tb = torch.tensor(-2.0, requires_grad=True)
     c = a * b + b ** 2                        tc = ta * tb + tb ** 2
     d = (c * 2 + a).tanh()                    td = torch.tanh(tc * 2 + ta)
     e = d.exp() + c                           te = torch.exp(td) + tc
     e.backward()                              te.backward()
     ```
     Assert `abs(a.grad - ta.grad.item()) < 1e-6` and same for `b`; also compare intermediates by calling `tc.retain_grad(); td.retain_grad()` on the torch side and checking `c.grad`, `d.grad`.
   - **The diamond test.** `x = Scalar(0.5); a = x.tanh(); out = a * a + a.exp()`. Here `a` feeds two consumers. Analytic check: `∂out/∂a = 2·a.data + exp(a.data)`; assert `a.grad` equals that after `out.backward()`, and assert `x.grad == a.grad * (1 - a.data**2)` to 1e-9. Then the same graph in torch with `retain_grad`, compare.
   - **Accumulation across backward calls.** Build a fresh tiny graph twice from the SAME leaf `Scalar`s and call backward twice; assert leaf grad doubled. Comment: this is the engine-level reason `zero_grad()` exists.
3. Add `def zero_grad_tree(self)` or simply document the convention: callers reset `node.grad = 0.0` on parameters before each backward (you'll do exactly this in Day 6's training loop).

**Verify — done when:**
```bash
pytest week03/d5_backward_test.py -q
```
- Every node's grad in the multi-node expression matches torch within 1e-6.
- The diamond test passes with BOTH the analytic formula and torch parity.
- Deep-graph smoke test: a chain `x` → 500 sequential `* 1.001` ops; `backward()` completes (if you hit Python's recursion limit in `build`, either raise the limit with `sys.setrecursionlimit` or rewrite `build` iteratively with an explicit stack — note which you chose).

**If stuck:**
- Grads all zero except the output → you iterated `topo` forward instead of `reversed(topo)`.
- Diamond grad exactly half of expected → some `_backward` uses `=` instead of `+=`.

## Day 3.6 — Deep build: train an MLP on two-moons with YOUR engine (~3.5h)
- [ ] done

**Goal:** Wrap `Scalar` into Neuron/Layer/MLP classes, generate a two-moons dataset, and train to ≥95% accuracy with a manual SGD loop — then prove forward/backward parity against torch.

**Learn:**
- **Neuron = weighted sum + nonlinearity.** `out = act(Σ wᵢxᵢ + b)` with each `wᵢ`, `b` a `Scalar`. A Layer is a list of Neurons sharing inputs; an MLP is Layers composed. Parameters are just "all the Scalars I should nudge".
- **Hinge loss (the one you'll use).** With labels `y ∈ {−1, +1}` and raw score `s`, per-example loss is `max(0, 1 − y·s)` — zero once the example is on the correct side with margin ≥ 1, linear penalty otherwise. In engine terms: `(1 - y * s).relu()` — no new ops needed. Total loss: mean over examples (+ optional L2 term `α·Σw²`).
- **The training loop skeleton.** forward → loss → zero all param grads → `loss.backward()` → for each param `p: p.data -= lr * p.grad`. That last line IS the optimizer; Week 4 dresses it up.
- **Why this is slow.** Every Scalar op is a Python object; a 200-point epoch builds tens of thousands of nodes. That's fine — the point is correctness, and it motivates why real autograd records ops on whole tensors.

**Read (30–45 min):**
- UDL book ch. 5 (Loss functions) — margin/hinge discussion — and ch. 6 (Fitting models, SGD): https://udlbook.github.io/udlbook/
- d2l.ai MLP chapter for the architecture vocabulary: https://d2l.ai/chapter_multilayer-perceptrons/mlp.html

**Build:**
1. Create `week03/mlp_engine.py` with:
```python
def make_moons(n: int = 200, noise: float = 0.1, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Two interleaved half-circles. Returns X (n,2) float32, y (n,) in {-1.,+1.}.
    Recipe: n//2 points per class. Class +1: angle t ~ U(0, pi),
    (cos t, sin t). Class -1: (1 - cos t, 0.5 - sin t)  [shifted/flipped half-circle].
    Add N(0, noise^2) to both coords. Use a torch.Generator seeded with `seed`."""

class Neuron:
    def __init__(self, n_in: int, nonlin: bool = True, rng: random.Random | None = None):
        """Weights ~ U(-1,1) as Scalars, bias Scalar(0). nonlin -> apply relu."""
    def __call__(self, x: list[Scalar]) -> Scalar: ...
    def parameters(self) -> list[Scalar]: ...

class Layer:
    def __init__(self, n_in: int, n_out: int, nonlin: bool = True, rng=None): ...
    def __call__(self, x: list[Scalar]) -> list[Scalar]: ...
    def parameters(self) -> list[Scalar]: ...

class MLP:
    def __init__(self, n_in: int, layer_sizes: list[int], rng=None):
        """e.g. MLP(2, [16, 16, 1]): two hidden relu layers, linear output."""
    def __call__(self, x: list[Scalar]) -> Scalar:  # returns the single output scalar
    def parameters(self) -> list[Scalar]: ...
```
2. Training script (in the same file, under `__main__`): model `MLP(2, [16, 16, 1])`; loss = mean hinge + `1e-4 * Σ w²` L2; full-batch SGD, `lr = 0.05` (optionally decay: `lr = 0.05 * (1 - epoch/max_epochs)`); ~100 epochs. Each epoch: zero grads (`for p in model.parameters(): p.grad = 0.0`), forward all points, backward, step. Print epoch, loss, accuracy every 10 epochs. Accuracy: `sign(s) == y`.
3. **Torch parity check** (write as a pytest test in the same file): build `MLP(2, [4, 1])` with a seeded rng; copy its weights into torch tensors — for each layer build `W` of shape `(n_out, n_in)` from `neuron.w[j].data` and `b` from biases, with `requires_grad=True`. Forward ONE point through both (`torch.relu(W1 @ x + b1)` etc.), apply the same hinge loss, backward both. Assert output scalars match to 1e-6 and EVERY weight gradient matches its `Scalar.grad` to 1e-5.
4. Decision boundary plot: evaluate the trained model on a 100×100 grid over `[-1.5, 2.5] × [-1.0, 1.5]`, `plt.contourf` the sign, scatter the data colored by label, save to `week03/moons_boundary.png`.

**Verify — done when:**
```bash
python week03/mlp_engine.py     # trains, saves plot
pytest week03/mlp_engine.py -q  # parity test passes
```
- Final training accuracy is approximately 0.95–1.00 with the suggested settings; the script asserts `acc >= 0.90`.
- The parity test passes: identical forward value, all grads matching ≤1e-5.
- The PNG shows a wiggly nonlinear boundary separating the moons — a linear cut cannot, which is the visual proof your hidden layers do something.

**If stuck:**
- Loss stuck at 1.0 with zero grads everywhere → all examples may start in the flat relu region; lower the init scale or check the hinge sign (`1 - y*s`, not `1 + y*s`).
- Parity mismatch only in grads, not outputs → weight-copy transposed: torch's `Linear` convention is `W @ x` with `W` shape `(out, in)`; neuron j's weights are ROW j.

## Day 3.7 — Review, custom autograd.Function, quiz (~2h)
- [ ] done

**Goal:** Consolidate the week, then cross the bridge back into torch by writing a custom `autograd.Function` — the official version of what your engine does.

**Learn:**
- **`torch.autograd.Function`** lets you define a forward and a hand-written backward that autograd splices into its graph — used for ops autograd can't derive (custom CUDA kernels, numerically smarter formulas) or shouldn't (straight-through estimators).
- **`ctx` is your `_parents` + saved state.** `ctx.save_for_backward(t)` in forward, `ctx.saved_tensors` in backward — torch's version of your closures capturing `out.data`.
- **backward receives `grad_output` and returns one gradient per forward input** — exactly your `parent.grad += local · out.grad`, expressed functionally.
- **`torch.autograd.gradcheck`** is the industrial version of your Day-3 checker: float64 inputs, central differences, relative-error comparison. Yours and theirs should agree about your `MyExp`.

**Read (30–45 min):**
- Extending PyTorch note, section "Extending torch.autograd" (read the full `Function` example and the gradcheck subsection): https://docs.pytorch.org/docs/stable/notes/extending.html
- `torch.autograd.Function` API page: https://docs.pytorch.org/docs/stable/autograd.html#function

**Build:**
1. `week03/d7_myexp.py`:
```python
class MyExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        out = torch.exp(x)
        ctx.save_for_backward(out)   # save the OUTPUT: d/dx exp(x) = exp(x)
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        (out,) = ctx.saved_tensors
        return grad_output * out

myexp = MyExp.apply
```
2. Verify three ways: (a) `torch.autograd.gradcheck(myexp, (torch.randn(5, dtype=torch.float64, requires_grad=True),))` returns True; (b) your own `check_gradients(lambda x: myexp(x).sum(), x64)`; (c) forward/grad parity vs `torch.exp` on the same input.
3. Skim your week's files once and tighten any comment you can no longer defend.

**Verify — done when:** all three checks in step 2 pass; you can answer the quiz below from memory, then check the Answers section.

**Redo cold (no peeking at your own code):**
1. Rewrite `Scalar.__mul__` including its `_backward` closure, from memory.
2. Rewrite `Scalar.backward()` (topo sort + seed + reverse loop), from memory.
3. Write `numeric_grad` for a scalar function of a vector, from memory.
4. On paper: draw the DAG for `e = tanh(a*b + b**2)` and hand-compute `∂e/∂b` at `a=1, b=2`, then confirm with torch in three lines.

**Self-quiz:**
1. Why does calling `backward()` twice on the same graph raise a RuntimeError by default, and what are the two ways around it?
2. After two `backward()` calls (graph rebuilt between them) without zeroing, what does a leaf's `.grad` contain, and why is this behavior a feature rather than a bug?
3. `y = model(x)` where `y` has shape `(8,)`. What exactly does `y.backward(v)` compute, and what shape must `v` be?
4. State two practical differences between `torch.no_grad()` and `torch.inference_mode()`.
5. Why are central differences preferred over forward differences in a gradient checker, and why must the check run in float64?
6. In your engine, why must `_backward` use `+=` into parent grads instead of `=`? Name the graph shape that exposes the bug.
7. Why must `backward()` traverse nodes in reverse topological order rather than, say, BFS from the output?
8. `tanh`'s backward uses `1 − out²` while `log`'s uses `1/a.data` — one saves the output, the other the input. What determines which a given op should save?
9. In `MyExp`, what does `backward` receive as `grad_output`, in terms of the chain rule?
10. Your Day-6 MLP trains in seconds with torch but minutes with `Scalar`s. What is the structural reason (not "Python is slow" — be specific about graph granularity)?

---

### Answers

1. Backward frees the intermediate tensors saved for gradient computation as it consumes them. Workarounds: `loss.backward(retain_graph=True)`, or recompute the forward pass to build a fresh graph.
2. The SUM of both passes' gradients — backward always accumulates into `.grad`. It's a feature because accumulation is exactly what makes (a) shared nodes/parameters correct within one backward and (b) gradient accumulation across micro-batches possible; the cost is that you must `zero_grad()` between steps.
3. It computes `vᵀJ` — the vector–Jacobian product, i.e. the gradient of the scalar `v·y` with respect to every leaf. `v` must have the same shape as `y`, here `(8,)`.
4. (a) Tensors created in `inference_mode` can never be used in a graph that later requires grad (they have no version counter); `no_grad` outputs can. (b) `inference_mode` is faster — it skips version-counter and view tracking. Use it for pure inference, `no_grad` when results might feed later training code.
5. Central differences cancel the second-order Taylor term, giving O(ε²) truncation error vs O(ε). Float64 is required because the subtraction `f(x+ε)−f(x−ε)` loses ~half the significand to cancellation; float32's ~7 decimal digits leave nothing for ε ≈ 1e-6.
6. Because a node may feed several consumers, each contributing a chain-rule term; the total derivative is their sum. The diamond graph (one node, two consumers, recombined) exposes `=`: it keeps only the last consumer's contribution.
7. A node's `_backward` reads its own `.grad` and pushes to parents. If it fires before all its consumers have contributed, it propagates a partial gradient. Reverse topological order guarantees every consumer has fired first. BFS by levels can interleave incorrectly when path lengths differ (the diamond again).
8. Whatever makes the local derivative cheapest to evaluate: `d tanh = 1 − tanh²` is a function of the output, `d log = 1/x` of the input. Ops save whichever tensor their derivative formula needs (autograd's "saved tensors" do exactly this).
9. `grad_output = ∂L/∂out`, the gradient of the final scalar with respect to this op's output. Backward returns `∂L/∂x = grad_output · ∂out/∂x` — your engine's `out.grad` times the local gradient.
10. Graph granularity: torch records one node per TENSOR op (a 200×16 matmul is one node executing vectorized C++), while the Scalar engine records one Python object plus one closure per ELEMENTARY float op — thousands of nodes and Python-level calls per layer per example.
