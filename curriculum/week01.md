# Week 1 — Tensor Fundamentals

This week builds the substrate every later week stands on: creating tensors with the right dtype and shape, indexing them without loops, predicting broadcast shapes in your head, reducing along dimensions without numerical blowups, and reshaping data between layouts. It comes first because autograd (Week 3) and nn-from-scratch (Week 4) are unlearnable if every tensor expression still requires trial and error. By Day 7 tensor manipulation should feel like list comprehensions do in Python: automatic.

**Week outcome:** a tested `week01/` package (creation functions, ~25 katas, stable softmax/cross-entropy, a tensor toolbox with a loop-free kNN classifier), the start of your personal `tlib/` library, and the skill of writing loop-free, shape-correct tensor code and predicting result dtypes and shapes before running.

Setup once: `mkdir -p week01 tlib && touch tlib/__init__.py`. Run all tests with `pytest week01/ -q` from the repo root.

Conventions for the whole program:
- Test files are named `test_<module>.py` next to the module; one test function per built function, named `test_<function>`.
- "No loops" means no Python `for`/`while`/comprehensions over tensor elements in the implementation; loops in test files (as naive references) are encouraged.
- Default comparison is `torch.allclose(actual, expected, atol=1e-6)` for float32; use `torch.equal` for integer/bool results where bit-exactness is the contract.
- When a check fails, print `actual.shape`, `actual.dtype`, and the first few elements before reaching for the debugger — most Week 1 bugs are shape or dtype, not values.

## Day 1.1 — Tensor creation & dtypes (~2h)
- [x] done
**Goal:** create tensors of any shape/dtype/device deliberately, and know which constructors copy memory and which share it.
**Learn:**
- *Factory functions* — `torch.zeros/ones/full/eye/arange/linspace/rand/randn/randint/empty` each take a shape (or range) plus `dtype=` and `device=` kwargs. `empty` allocates without initializing: its contents are whatever bytes were in memory, so never read it before writing.
- *The dtype zoo* — `float32` is the deep-learning default (weights, activations); `float64` is NumPy's default (a frequent silent mismatch); `int64` ("long") is required for indices and class labels; `bool` for masks; `float16`/`bfloat16` for mixed precision later.
- *Type promotion* — mixing dtypes in one op promotes to the "larger" type by fixed rules (bool < int < float; within a category, wider wins). `torch.result_type(a, b)` tells you the answer without computing. A Python scalar (`x + 1.5`) promotes more weakly than a tensor of the same value.
- *`torch.tensor` vs `torch.Tensor` vs `as_tensor`/`from_numpy`* — `torch.tensor(data)` always copies and infers dtype from the data; `torch.Tensor(...)` is a legacy float32 constructor — avoid it; `torch.as_tensor(arr)` and `torch.from_numpy(arr)` share memory with a NumPy array when dtypes allow, so mutating one mutates the other.
- *Device argument* — every factory accepts `device="cpu"` / `device="cuda"`. On CPU-only machines, writing `device=` through your code anyway is what makes it portable later (GPU variant: the same code runs unchanged with `device="cuda"`).
**Read (30–45 min):**
- Tensors tutorial: https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html (sections "Initializing a Tensor", "Attributes", "Bridge with NumPy").
- Tensor attributes & promotion rules: https://docs.pytorch.org/docs/stable/tensor_attributes.html (section "torch.dtype", subsection on type promotion).
- Skim the "Creation Ops" table: https://docs.pytorch.org/docs/stable/torch.html#creation-ops
**Build:** create `week01/d1_creation.py` with these functions (type hints + one-line docstrings stating the contract), and `week01/test_d1_creation.py` with one test per function checking shape, dtype, and values:
1. `def board(n: int) -> torch.Tensor` — n×n float32 zeros.
2. `def ones_long(shape: tuple[int, ...]) -> torch.Tensor` — int64 ones.
3. `def sevens(rows: int, cols: int) -> torch.Tensor` — float32 filled with 7.0 via `full`.
4. `def identity_f64(n: int) -> torch.Tensor` — float64 identity via `eye`.
5. `def evens_below(n: int) -> torch.Tensor` — int64 tensor `[0, 2, 4, ...) < n` via `arange`.
6. `def grid(start: float, stop: float, num: int) -> torch.Tensor` — `num` evenly spaced points inclusive of both ends via `linspace`.
7. `def uniform(shape: tuple[int, ...], seed: int) -> torch.Tensor` — call `torch.manual_seed(seed)` then `rand`.
8. `def gaussian(shape: tuple[int, ...], seed: int) -> torch.Tensor` — seeded `randn`.
9. `def dice(n: int, seed: int) -> torch.Tensor` — n seeded die rolls in {1..6} via `randint` (note the exclusive high bound).
10. `def from_list32(data: list) -> torch.Tensor` — `torch.tensor(data, dtype=torch.float32)`.
11. `def shared_view(arr: "np.ndarray") -> torch.Tensor` — wrap WITHOUT copying (`from_numpy`).
12. `def private_copy(arr: "np.ndarray") -> torch.Tensor` — wrap WITH a copy (`torch.tensor`).
13. Script `week01/d1_promotion.py`: loop over `[torch.bool, torch.int32, torch.int64, torch.float32, torch.float64]` pairs, print a table of `torch.result_type(torch.empty(1, dtype=a), torch.empty(1, dtype=b))`. Add one row showing `(torch.ones(1, dtype=torch.int64) + 1.5).dtype` vs `torch.result_type(torch.ones(1, dtype=torch.int64), torch.tensor(1.5))` and write a 2-line comment explaining the scalar rule.
**Verify — done when:** `pytest week01/test_d1_creation.py -q` passes, including these exact checks:
```python
import numpy as np, torch
from week01.d1_creation import *
assert board(3).dtype == torch.float32 and board(3).shape == (3, 3)
assert dice(1000, 0).min() >= 1 and dice(1000, 0).max() <= 6
assert torch.equal(uniform((2, 2), 42), uniform((2, 2), 42))   # same seed, same draws
a = np.zeros(3); t = shared_view(a); a[0] = 5.0
assert t[0].item() == 5.0                                       # memory is shared
a2 = np.zeros(3); t2 = private_copy(a2); a2[0] = 5.0
assert t2[0].item() == 0.0                                      # memory is not shared
assert torch.result_type(torch.empty(1, dtype=torch.int64), torch.empty(1, dtype=torch.float32)) == torch.float32
```
**If stuck:** the promotion subsection of https://docs.pytorch.org/docs/stable/tensor_attributes.html; the `torch.from_numpy` doc (https://docs.pytorch.org/docs/stable/generated/torch.from_numpy.html) states the sharing behavior explicitly.

## Day 1.2 — Indexing & slicing (~2h)
- [x] done
**Goal:** select any sub-tensor without writing a Python loop, and know whether the result shares memory with the source.
**Learn:**
- *Basic slicing returns views* — `t[1:4]`, `t[:, ::2]`, `t[..., -1]` create no new data; they reinterpret the same storage. Mutating the slice mutates the original.
- *Integer ("fancy") indexing returns copies* — indexing with a tensor of indices, e.g. `t[torch.tensor([0, 2])]`, gathers elements into fresh memory. Pairs of index tensors broadcast against each other: `m[rows, cols]` picks element `(rows[i], cols[i])` for each i — the workhorse for "per-row pick".
- *Boolean masks* — `t[t > 0]` flattens and returns a copy of the selected elements; `t[mask] = 0` writes through to the original (assignment to an indexed expression is always in-place on `t`).
- *Ellipsis and `None`* — `...` means "all remaining dims" (`t[..., 0]` works for any rank); indexing with `None` inserts a size-1 dim at that position (`t[:, None]` turns shape `(n,)` into `(n, 1)`), equivalent to `unsqueeze`.
- *Negative indices and step slicing* — `t[-1]` is the last row; `t[::2]` every other element. PyTorch slices cannot use negative steps (use `torch.flip`).
**Read (30–45 min):**
- Indexing section of the tensors deep-dive: https://docs.pytorch.org/tutorials/beginner/introyt/tensors_deeper_tutorial.html
- NumPy indexing reference (PyTorch follows it almost exactly): https://numpy.org/doc/stable/user/basics.indexing.html (sections "Basic indexing", "Advanced indexing", "Boolean array indexing").
- Tensor views list (which ops view, which copy): https://docs.pytorch.org/docs/stable/tensor_view.html
**Build:** `week01/d2_indexing.py` + `week01/test_d2_indexing.py`. No Python loops anywhere. Each function: docstring with a 2×2 or 3×3 worked example.
1. `def anti_diagonal(m: torch.Tensor) -> torch.Tensor` — top-right to bottom-left of an n×n matrix. Hint: two `arange`s.
2. `def select_per_row(m: torch.Tensor, idx: torch.Tensor) -> torch.Tensor` — `out[i] = m[i, idx[i]]`.
3. `def reverse_every_other_row(m: torch.Tensor) -> torch.Tensor` — rows 0, 2, 4… left-right reversed; returns a new tensor, input untouched.
4. `def positives(x: torch.Tensor) -> torch.Tensor` — 1-D tensor of strictly positive elements.
5. `def relu_(x: torch.Tensor) -> torch.Tensor` — in-place: negative entries set to 0 via mask assignment; return x.
6. `def border(m: torch.Tensor) -> torch.Tensor` — clone of m with interior zeroed (slice assignment on `m[1:-1, 1:-1]`).
7. `def last_col(t: torch.Tensor) -> torch.Tensor` — last element along the final dim, any rank, via `...`.
8. `def as_column(x: torch.Tensor) -> torch.Tensor` — `(n,)` → `(n, 1)` using `None` indexing only.
9. `def swap_halves(x: torch.Tensor) -> torch.Tensor` — even-length 1-D: second half then first half (use `torch.cat` of two slices).
10. `def checkerboard(m: torch.Tensor) -> torch.Tensor` — clone with every element where (i+j) is odd zeroed (two strided slice-assignments).
11. Probe script `week01/d2_views.py`: show `t[1:].data_ptr()` lies inside t's storage while `t[torch.tensor([1])].data_ptr()` does not; show mutating a basic slice changes `t` but mutating a fancy-indexed result does not. Print a one-line conclusion per case.
**Verify — done when:**
```python
m = torch.arange(9).reshape(3, 3)          # [[0,1,2],[3,4,5],[6,7,8]]
assert torch.equal(anti_diagonal(m), torch.tensor([2, 4, 6]))
assert torch.equal(select_per_row(m, torch.tensor([2, 0, 1])), torch.tensor([2, 3, 7]))
assert torch.equal(reverse_every_other_row(m)[0], torch.tensor([2, 1, 0]))
assert torch.equal(reverse_every_other_row(m)[1], torch.tensor([3, 4, 5]))
x = torch.tensor([-1.0, 2.0, -3.0]); relu_(x)
assert torch.equal(x, torch.tensor([0.0, 2.0, 0.0]))           # mutated in place
assert last_col(torch.zeros(2, 3, 4)).shape == (2, 3)
assert as_column(torch.arange(5)).shape == (5, 1)
```
plus your probe script's two memory-sharing demonstrations print the expected conclusions.
**If stuck:** "Advanced indexing" in the NumPy doc above; `torch.Tensor.data_ptr` docs for the storage probe.

## Day 1.3 — Broadcasting (~2h)
- [x] done
**Goal:** compute the broadcast shape of any two tensors mentally, and exploit broadcasting to replace loops.
**Learn:**
- *The two rules* — align shapes at the RIGHT edge; (1) a missing dim is treated as size 1; (2) two dims are compatible iff equal or either is 1. The result dim is the max of the pair. Anything else raises.
- *Size-1 stretching is virtual* — a size-1 dim is read as if copied along that axis, but no memory is copied; broadcasting is free.
- *`None` insertion is the broadcasting power tool* — `a[:, None] - b[None, :]` turns two `(n,)` vectors into an `(n, n)` table of all pairwise differences. Inserting axes to force the alignment you want is 90% of practical broadcasting.
- *Common trap* — `(n,)` minus `(n, 1)` broadcasts to `(n, n)` instead of erroring. Shape bugs from accidental broadcasting are silent; assert shapes when unsure.
**Read (30–45 min):**
- Broadcasting semantics: https://docs.pytorch.org/docs/stable/notes/broadcasting.html (whole page — it is short).
- NumPy broadcasting guide for the pictures: https://numpy.org/doc/stable/user/basics.broadcasting.html
- d2l.ai broadcasting subsection: https://d2l.ai/chapter_preliminaries/ndarray.html (section "Broadcasting Mechanism").
**Build:** `week01/d3_broadcast.py` + tests.
1. `def broadcast_shape(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]` — implement the rules from scratch on plain tuples; `raise ValueError(f"incompatible: {a} vs {b}")` on conflict. No torch calls inside.
2. Randomized check in the test file: 200 iterations generating random shapes (rank 0–4, dims from {1, 2, 3, 5}); compare `broadcast_shape(a, b)` to `torch.broadcast_shapes(a, b)`, and assert `ValueError` is raised exactly when torch raises `RuntimeError`.
3. Katas (no loops): `de_mean_cols(m)` — subtract each column's mean, result columns sum to ~0; `outer(a, b)` — outer product via `None` indexing and `*`, no `torch.outer`; `pairwise_diff(a, b)` — `(n, d)`, `(m, d)` → `(n, m, d)` of `a[i] - b[j]`; `normalize_rows(m)` — each row divided by its L2 norm (`keepdim=True`); `dist_from(points, q)` — `(n, d)` points, `(d,)` query → `(n,)` Euclidean distances; `add_bias(x, b)` — `(batch, features) + (features,)`.
**Verify — done when:**
```python
assert broadcast_shape((5, 1, 3), (4, 3)) == (5, 4, 3)
assert broadcast_shape((), (3, 2)) == (3, 2)
import pytest
with pytest.raises(ValueError):
    broadcast_shape((3, 2), (2, 2))
a, b = torch.randn(4, 3), torch.randn(5, 3)
assert pairwise_diff(a, b).shape == (4, 5, 3)
assert torch.allclose(pairwise_diff(a, b)[2, 3], a[2] - b[3])
assert torch.allclose(outer(torch.arange(3.), torch.arange(4.)), torch.outer(torch.arange(3.), torch.arange(4.)))
assert torch.allclose(de_mean_cols(torch.randn(6, 4)).mean(dim=0), torch.zeros(4), atol=1e-6)
assert torch.allclose(normalize_rows(torch.randn(5, 3)).norm(dim=1), torch.ones(5), atol=1e-6)
```
and the 200-iteration randomized agreement test passes.
**If stuck:** re-read the worked examples at the bottom of the PyTorch broadcasting note; for `pairwise_diff`, write the target indices `out[i, j, k] = a[i, k] - b[j, k]` and add `None` where an operand lacks an index.

## Day 1.4 — Reductions & numerical stability (~2.5h)
- [x] done
**Goal:** reduce along chosen dims fluently and implement softmax/cross-entropy that survive extreme logits.
**Learn:**
- *`dim` and `keepdim`* — `x.sum(dim=1)` collapses dim 1 away; `keepdim=True` leaves it as size 1 so the result broadcasts back against `x` (essential for "subtract the row max" patterns).
- *max vs amax vs argmax* — `x.max(dim)` returns a `(values, indices)` named tuple; `x.amax(dim)` returns values only; `x.argmax(dim)` indices only.
- *Why naive softmax fails* — `exp(1000.) = inf` in float32 (overflow starts near `exp(89)`), so `exp(x)/exp(x).sum()` yields `inf/inf = nan`. Subtracting `x.max()` first changes nothing mathematically (softmax is shift-invariant) but keeps every exponent ≤ 0.
- *The log-sum-exp trick* — `log Σ exp(xᵢ) = m + log Σ exp(xᵢ − m)` with `m = max(x)`. This is the stable core of log-softmax, cross-entropy, and (later) attention and mixture losses.
- *Cross-entropy from logits* — for target class y: `CE = −log_softmax(logits)[y] = logsumexp(logits) − logits[y]`. Never compute `log(softmax(x))` as two steps; `log(0.) = -inf`.
**Read (30–45 min):**
- Reduction ops table: https://docs.pytorch.org/docs/stable/torch.html#reduction-ops (read `sum`, `mean`, `max`, `amax`, `logsumexp` entries).
- `torch.logsumexp`: https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html
- Understanding Deep Learning, Chapter 5 (Loss functions), section on cross-entropy / categorical distribution: https://udlbook.github.io/udlbook/
**Build:** `week01/d4_stable.py` + tests. Use only primitives (`exp`, `log`, `max`, `sum`, indexing, broadcasting) — no `torch.softmax`, `F.*`, or `torch.logsumexp` inside the implementations.
1. `def naive_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor` — the textbook formula, kept to demonstrate the failure.
2. `def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor` — subtract `x.amax(dim, keepdim=True)` first.
3. `def stable_log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor` — `x - m - (x - m).exp().sum(dim, keepdim=True).log()` with `m` the keepdim max. Do NOT call log on softmax output.
4. `def logsumexp(x: torch.Tensor, dim: int = -1) -> torch.Tensor` — your own, via the trick.
5. `def cross_entropy_from_scratch(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor` — `(batch, K)` float logits, `(batch,)` int64 targets → scalar mean of `logsumexp(logits) − logits[i, targets[i]]` (use Day 1.2's per-row pick).
**Verify — done when:**
```python
import torch.nn.functional as F
x = torch.randn(8, 10)
assert torch.allclose(stable_softmax(x), F.softmax(x, dim=-1), atol=1e-6)
assert torch.allclose(stable_log_softmax(x), F.log_softmax(x, dim=-1), atol=1e-6)
assert torch.allclose(logsumexp(x), torch.logsumexp(x, dim=-1), atol=1e-6)
big = torch.tensor([[1e4, -1e4, 0.0]])
assert torch.isnan(naive_softmax(big)).any()                    # the failure, on display
assert torch.allclose(stable_softmax(big), F.softmax(big, dim=-1))
assert torch.isfinite(stable_log_softmax(big)).all()
t = torch.randint(0, 10, (8,))
assert torch.allclose(cross_entropy_from_scratch(x, t), F.cross_entropy(x, t), atol=1e-6)
# Mathematically guaranteed: uniform logits over K classes -> CE = ln(K)
K = 7
u = torch.zeros(4, K)
assert torch.allclose(cross_entropy_from_scratch(u, torch.randint(0, K, (4,))), torch.tensor(K).float().log())
```
You will reuse `stable_log_softmax` and `cross_entropy_from_scratch` verbatim in Week 4's loss functions — make them clean.
**If stuck:** the `torch.logsumexp` doc page states the identity; check `keepdim=True` on every intermediate reduction — a missing keepdim mis-broadcasts silently.

## Day 1.5 — Shape surgery (~2h)
- [x] done
**Goal:** move data between layouts (batch/channel/spatial) with reshape/permute and know which ops copy.
**Learn:**
- *`reshape` vs `view`* — both reinterpret the same elements in row-major order; `view` demands a compatible memory layout and errors otherwise, `reshape` silently copies when it must. `-1` infers one dimension.
- *`permute`/`transpose` reorder dims, not data* — they return views with rearranged strides; the elements don't move (full story on Day 2.1). `transpose(d0, d1)` swaps two dims; `permute` reorders all.
- *Reordering ≠ reshaping* — `x.reshape(b, c, h*w)` and `x.permute(...)` answer different questions; if your "transpose" could be done by reshape, you've probably scrambled the data. Verify with a small known tensor.
- *`squeeze`/`unsqueeze`/`flatten`* — drop/insert size-1 dims; `flatten(start_dim=1)` is the canonical "image → vector per sample".
- *`expand` vs `repeat`* — `expand` broadcasts size-1 dims as a zero-copy view (writing into it is dangerous); `repeat` tiles real copies. Prefer expand when downstream ops only read.
**Read (30–45 min):**
- Tensor views: https://docs.pytorch.org/docs/stable/tensor_view.html (the list of view ops).
- `Tensor.view` doc (the contiguity condition): https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html
- `Tensor.expand` doc: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.expand.html
**Build:** `week01/d5_shapes.py` + tests. Eight katas:
1. `def nhwc_to_nchw(x: torch.Tensor) -> torch.Tensor` — `(N, H, W, C)` → `(N, C, H, W)` via permute; and `nchw_to_nhwc` back.
2. `def flatten_batch(x: torch.Tensor) -> torch.Tensor` — `(N, ...)` → `(N, prod(...))` via `flatten(start_dim=1)`.
3. `def to_patches(img: torch.Tensor, p: int) -> torch.Tensor` — `(C, H, W)` with H, W divisible by p → `(num_patches, C * p * p)`: `reshape(C, H//p, p, W//p, p)` → `permute(1, 3, 0, 2, 4)` → flatten. (This is ViT patchification.)
4. `def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor` — `(B, T, D)` → `(B, n_heads, T, D // n_heads)` (reshape then transpose 1,2); and `merge_heads` inverting it (needs `.contiguous()` or `.reshape` — note which and why).
5. `def drop_unit_dims(x: torch.Tensor) -> torch.Tensor` — remove all size-1 dims via `squeeze`.
6. `def broadcast_row(row: torch.Tensor, n: int) -> torch.Tensor` — `(d,)` → `(n, d)` via `expand`, zero-copy.
7. `def tile_row(row: torch.Tensor, n: int) -> torch.Tensor` — same shape via `repeat`, real copy.
8. Probe: assert `broadcast_row(r, 4).stride(0) == 0` and `tile_row(r, 4).stride(0) != 0` — the stride-0 trick IS expand (Day 2.1 explains).
**Verify — done when:**
```python
x = torch.randn(2, 4, 6, 3)
assert nhwc_to_nchw(x).shape == (2, 3, 4, 6)
assert torch.equal(nchw_to_nhwc(nhwc_to_nchw(x)), x)             # round-trip
assert nhwc_to_nchw(x)[0, 1, 2, 3] == x[0, 2, 3, 1]              # data, not just shape
img = torch.arange(2 * 4 * 4.).reshape(2, 4, 4)
P = to_patches(img, 2)
assert P.shape == (4, 8)
assert torch.equal(P[0], torch.stack([img[0, :2, :2], img[1, :2, :2]]).reshape(-1))
h = split_heads(torch.randn(2, 5, 12), 3)
assert h.shape == (2, 3, 5, 4)
assert torch.equal(merge_heads(split_heads(torch.arange(48.).reshape(1, 4, 12), 4)), torch.arange(48.).reshape(1, 4, 12))
assert broadcast_row(torch.arange(3.), 4).stride(0) == 0
```
**If stuck:** print a `torch.arange(24).reshape(2, 3, 4)` before/after each op and find specific elements; the `view` doc's contiguity paragraph explains the `merge_heads` error you'll hit.

## Day 1.6 — Deep build: tensor toolbox (~3.5h)
- [x] done
**Goal:** combine the whole week into a tested toolbox, culminating in a loop-free kNN classifier.
**Learn:**
- *Vectorization discipline* — every function gets two implementations: a naive loop reference (allowed today only, in the test file) and the tensor version. Agreement via `allclose` is the proof.
- *The pairwise-distance expansion* — `‖a − b‖² = ‖a‖² + ‖b‖² − 2a·b` turns an O(n·m) loop over pairs into one matmul plus two broadcasts. Floating-point cancellation can produce tiny negatives — clamp at 0 before sqrt.
- *kNN as pure tensor ops* — distances → `topk(largest=False)` → gather neighbor labels → majority vote via `mode` (or one_hot-sum-argmax). No training; the "model" is the data.
**Read (20–30 min):** skim `torch.topk` (https://docs.pytorch.org/docs/stable/generated/torch.topk.html), `torch.cumsum`, `torch.mode` generated-doc pages.
**Build:** `week01/toolbox.py` + `week01/test_toolbox.py`.
1. `def one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor` — `(n,)` int64 → `(n, num_classes)` float32; no `F.one_hot`; use `zeros` + fancy-index assignment (`out[arange(n), labels] = 1.0`).
2. `def moving_average(x: torch.Tensor, w: int) -> torch.Tensor` — `(n,)` → `(n − w + 1,)` via cumsum: pad a zero in front, then `(c[w:] − c[:-w]) / w`. No loops, no conv.
3. `def pairwise_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor` — `(n, d)`, `(m, d)` → `(n, m)` Euclidean distances via the expansion; `clamp(min=0)` before `sqrt`.
4. `def batched_outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor` — `(B, n)`, `(B, m)` → `(B, n, m)` via `None` indexing.
5. `def make_blobs(n_per_class: int, centers: torch.Tensor, std: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]` — for `(K, d)` centers return `(K*n, d)` points (`center + std * randn`) and `(K*n,)` int64 labels.
6. `def knn_predict(train_x, train_y, test_x, k: int = 5) -> torch.Tensor` — `pairwise_dist` → `topk(k, largest=False)` → `train_y[indices]` → majority vote.
7. In the test file: loop references `pairwise_dist_loop`, `moving_average_loop`, and `knn_predict_loop` (triple-nested, brute force), plus an accuracy report.
8. Promote for reuse: create `tlib/ops.py` exporting `one_hot`, `pairwise_dist`, `stable_softmax`, `stable_log_softmax`, `cross_entropy_from_scratch` (import or copy from Day 1.4). Add them to `tlib/__init__.py`. Week 4's nn-from-scratch imports these.
9. Stretch (only if under 3h so far): `def knn_accuracy_sweep(tr_x, tr_y, te_x, te_y, ks: list[int]) -> torch.Tensor` — accuracy for each k, reusing one `pairwise_dist` call across all ks (compute distances once, slice `topk` results). Plot accuracy vs k with matplotlib and eyeball where it degrades.
**Verify — done when:**
```python
a, b = torch.randn(50, 8), torch.randn(30, 8)
assert torch.allclose(pairwise_dist(a, b), torch.cdist(a, b), atol=1e-4)
assert torch.allclose(pairwise_dist(a, b), pairwise_dist_loop(a, b), atol=1e-4)
x = torch.randn(100)
assert torch.allclose(moving_average(x, 7), moving_average_loop(x, 7), atol=1e-5)
lab = torch.tensor([0, 2, 1])
assert torch.equal(one_hot(lab, 3), torch.tensor([[1., 0, 0], [0, 0, 1], [0, 1, 0]]))
centers = torch.tensor([[0., 0.], [6., 6.], [0., 6.]])
X, y = make_blobs(60, centers, std=1.0, seed=0)
tr_x, tr_y, te_x, te_y = X[::2], y[::2], X[1::2], y[1::2]
pred = knn_predict(tr_x, tr_y, te_x, k=5)
assert torch.equal(pred, knn_predict_loop(tr_x, tr_y, te_x, 5))  # exact agreement with loop version
acc = (pred == te_y).float().mean().item()
assert acc > 0.9   # blobs 6 std-devs apart: expect approximately 0.95-1.0
```
**If stuck:** for cancellation-driven `nan`s in `pairwise_dist`, inspect `(d2 < 0).sum()` before the clamp; `torch.mode` docs for tie behavior in the vote.

## Day 1.7 — Review, quiz, redo-cold (~2h)
- [ ] done
**Goal:** consolidate; find and patch the gaps before Week 2.
**Learn:** nothing new — re-skim your own `week01/` code and the broadcasting note. Write a half-page summary in `week01/notes.md` of the three facts you found least obvious this week.
**Self-quiz (closed book, write answers down, then check the Answers section at the bottom of this file):**
1. What dtype results from `torch.ones(3, dtype=torch.int64) * torch.tensor(2.0)`? And from `torch.ones(3, dtype=torch.int64) * 2.0`?
2. Which of these share memory with the source: `t[2:5]`, `t[t > 0]`, `t[torch.tensor([0, 1])]`, `t.permute(1, 0)`, `t.reshape(-1)` on a contiguous t?
3. Broadcast shapes of `(7, 1, 5)` with `(3, 1)`? Of `(2, 3)` with `(3, 2)`?
4. Why is softmax shift-invariant, and why does that make it numerically rescuable?
5. Write the cross-entropy of one sample purely in terms of `logsumexp` and one logit.
6. What is the exact mean cross-entropy when all K logits are equal, and why?
7. `x.max(dim=1)` vs `x.amax(dim=1)` — what does each return?
8. When does `.view()` throw where `.reshape()` succeeds, and what does reshape do in that case?
9. `expand` vs `repeat`: which allocates, and why is writing into the other one dangerous?
10. `a` has shape `(n,)`, `b` has shape `(n, 1)`. What shape is `a - b`, and why is this a classic silent bug?
11. You need `out[i] = m[i, idx[i]]`. Write the one-line indexing expression.
12. `torch.from_numpy(arr)` then `arr += 1` — what happens to the tensor?
**Redo cold (no docs, no peeking at your old code; fresh file `week01/d7_cold.py`):**
1. `broadcast_shape` from scratch, verified against `torch.broadcast_shapes` on 5 hand-picked pairs including one failure.
2. `stable_log_softmax` + the `±1e4` extreme-logits check.
3. `pairwise_dist` via the expansion, verified against `torch.cdist`.
4. `to_patches` for a `(3, 8, 8)` image with p=4, with the data-correctness assert (not just shape).
5. `select_per_row` and `one_hot` (both are one or two lines — they should take under two minutes each).
Score yourself: any drill needing the docs goes on a note to redo in three days.

---

## Answers (Day 1.7 quiz)
1. Both `float32`. Tensor×tensor promotes int64+float32→float32; the Python scalar `2.0` also promotes to the default float dtype, float32.
2. Share: `t[2:5]`, `t.permute(1, 0)`, and `t.reshape(-1)` (reshape on a contiguous tensor returns a view). Copies: boolean mask result, integer (fancy) indexing result.
3. `(7, 3, 5)` — right-align, pad `(3, 1)` to `(1, 3, 1)`, take maxes. `(2, 3)` vs `(3, 2)`: error — 3 vs 2 in the last dim, neither is 1.
4. `softmax(x + c) = softmax(x)` because `e^{x+c} = e^c e^x` and the `e^c` cancels in the ratio. So you may subtract `max(x)`, making the largest exponent 0 — no overflow possible, and the result is bit-for-bit mathematically identical.
5. `CE = logsumexp(logits) − logits[y]`.
6. `ln(K)`. Softmax of equal logits is uniform `1/K`, and `−log(1/K) = ln K`, independent of the target class — guaranteed, not approximate.
7. `max(dim=1)` returns a named tuple `(values, indices)`; `amax(dim=1)` returns only the values tensor.
8. `view` throws when the requested shape can't be expressed as strides over the existing memory layout (e.g. after `transpose`); `reshape` falls back to copying the data into a fresh contiguous tensor.
9. `repeat` allocates real copies. `expand` returns a stride-0 view where many output elements alias one memory cell, so a single write appears at every "copy".
10. `(n, n)` — `(n,)` is padded to `(1, n)`, broadcasting against `(n, 1)`. Silent because no error is raised; a downstream `.mean()` still returns a scalar and the bug hides.
11. `m[torch.arange(m.shape[0]), idx]`.
12. The tensor changes too — `from_numpy` shares the buffer, and `+=` mutates it in place. (`arr = arr + 1` would rebind the name and NOT affect the tensor.)
