# Week 2 — Memory Model & Advanced Ops

Week 1 treated tensors as logical grids; this week opens the hood: a tensor is a (storage, offset, shape, strides) tuple, and every view trick from last week falls out of stride arithmetic. On top of that mental model you learn the high-leverage ops — `as_strided`, `einsum`, `gather`/`scatter` — and finish by rebuilding matmul and conv1d from raw strides, which is the moment "PyTorch is magic" becomes "PyTorch is bookkeeping plus fast kernels". It precedes Week 3 (autograd) because debugging gradients requires knowing exactly when ops alias memory versus copy it.

**Week outcome:** `week02/` with a layout inspector, stride-built sliding windows, ten einsum reimplementations, scatter/gather utilities, plus `tlib/utils.py` (seeding + device helpers) used for the rest of the program; the skill of predicting strides on paper and choosing the right advanced op without searching.

Setup: `mkdir -p week02`. Run tests with `pytest week02/ -q`. Week 1's conventions still apply (test naming, allclose tolerances, loops only in test-file references). New rule for this week: before running any expression involving views, say its expected stride out loud (or write it down) — the predict-then-check habit from Day 2.1 is the actual skill being trained, and it only forms if you predict every time.

## Day 2.1 — Storage & strides (~2.5h)
- [ ] done
**Goal:** given any tensor expression, predict shape, strides, and storage offset before running it.
**Learn:**
- *The four-field model* — a tensor is a pointer into a flat 1-D storage, plus `storage_offset`, `shape`, and `strides`. Element `t[i, j]` lives at storage index `offset + i*stride[0] + j*stride[1]`. That single formula explains views, transposes, and slicing.
- *Strides in elements, contiguity defined* — a fresh `(R, C)` tensor has strides `(C, 1)`: step 1 storage slot to move along a row, C slots to move down a column. "Contiguous" means strides are exactly this row-major pattern, so the logical order matches memory order.
- *Views are stride edits* — `t.transpose(0, 1)` just swaps the stride tuple to `(1, C)`; `t[2:, 1:]` bumps the offset and shrinks the shape. Zero data movement, which is why mutations propagate between a view and its base.
- *Why transpose breaks `view`* — after transposing, memory order no longer matches logical row-major order, and no stride tuple over the same storage can express the flattened sequence. `view` errors; `.contiguous()` materializes a fresh row-major copy, after which `view` works; `reshape` does contiguous-then-view for you when needed.
- *`expand` is stride 0* — broadcasting a size-1 dim sets its stride to 0, so every logical index along it reads the same storage cell (the Day 1.5 probe, explained).
**Read (30–45 min):**
- Tensor views: https://docs.pytorch.org/docs/stable/tensor_view.html (full page).
- `torch.Tensor.stride`, `torch.Tensor.storage_offset`, `torch.Tensor.is_contiguous` generated-doc pages under https://docs.pytorch.org/docs/stable/tensors.html
- d2l.ai "Saving Memory" + indexing subsections for the aliasing angle: https://d2l.ai/chapter_preliminaries/ndarray.html
**Build:** `week02/d1_layout.py` + `week02/test_d1_layout.py`.
1. `def explain_layout(t: torch.Tensor) -> dict` — returns `{"shape": tuple(t.shape), "stride": t.stride(), "offset": t.storage_offset(), "contiguous": t.is_contiguous(), "numel": t.numel(), "storage_size": len(t.untyped_storage()) // t.element_size()}`. Docstring: a worked example for `torch.arange(12).reshape(3, 4)[1:, ::2]`.
2. `def index_to_storage(t: torch.Tensor, idx: tuple[int, ...]) -> int` — compute `offset + Σ idx[k] * stride[k]` yourself (don't call torch inside). Verify by reading the storage directly at that position: `t.as_strided((1,), (1,), index_to_storage(t, idx)).item() == t[idx].item()`.
3. Predict-then-check suite (the core exercise): in the test file, for each case below FIRST write the expected stride/offset as a literal in the assert, THEN run. Cases for `base = torch.arange(24.).reshape(2, 3, 4)`:
   - `base` itself → stride `(12, 4, 1)`, offset 0.
   - `base.transpose(0, 2)`, `base.permute(2, 0, 1)`, `base[1]`, `base[:, 1:, ::2]`, `base[..., 3]`, `base.unsqueeze(1)`, `torch.arange(5.).expand(4, 5)`.
   Wrong prediction = delete the literal, re-derive on paper, do not just paste the printed value.
4. `def shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool` — compare `a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()`. Demonstrate: `shares_storage(t, t.t())` is True, `shares_storage(t, t.contiguous())` is False for a transposed t, `shares_storage(t, t.clone())` is False.
5. Demonstrate the view/reshape/clone triangle: `t.t().view(-1)` raises RuntimeError (assert with `pytest.raises`); `t.t().reshape(-1)` succeeds but no longer shares storage; `t.t().contiguous().view(-1)` succeeds.
**Verify — done when:** all predict-then-check asserts pass with your hand-written literals, plus:
```python
t = torch.arange(12.).reshape(3, 4)
v = t[1:, ::2]
assert explain_layout(v) == {"shape": (2, 2), "stride": (4, 2), "offset": 4,
                             "contiguous": False, "numel": 4, "storage_size": 12}
assert index_to_storage(v, (1, 1)) == 4 + 1*4 + 1*2 == 10
v[0, 0] = 99.
assert t[1, 0] == 99.                       # views write through
import pytest
with pytest.raises(RuntimeError):
    t.t().view(-1)
```
**If stuck:** the tensor_view doc lists exactly which ops return views; re-derive the storage-index formula for a 2×3 example on paper — every stride question reduces to it.

## Day 2.2 — as_strided & sliding windows (~2h)
- [ ] done
**Goal:** construct overlapping-window views by writing strides directly, and know why that's sharp-edged.
**Learn:**
- *`as_strided(size, stride, offset)`* — build an arbitrary view by specifying the layout tuple yourself. It bypasses all safety checks: sizes/strides that walk past the storage read garbage or crash, so always reason from the formula `offset + Σ idx·stride < storage_size`.
- *Overlapping windows for free* — a sliding window over a 1-D tensor is the view `size=(num_windows, window)`, `stride=(step*s, s)` where `s = x.stride(0)`. The same element appears in many windows without being copied — O(1) memory.
- *`unfold` is the safe wrapper* — `x.unfold(dim, size, step)` produces exactly these windowed views with bounds checking. Use it in real code; build it manually today to understand it.
- *The danger: aliased writes* — in-place ops on overlapping views (e.g. `windows += 1`) hit shared cells multiple times or in undefined order. Rule: treat `as_strided`/`unfold` results as read-only; `.clone()` before mutating.
**Read (30–45 min):**
- `torch.as_strided`: https://docs.pytorch.org/docs/stable/generated/torch.as_strided.html (note the warning paragraph).
- `torch.Tensor.unfold`: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.unfold.html (study the doc's worked example).
**Build:** `week02/d2_strided.py` + tests.
1. `def sliding_window_1d(x: torch.Tensor, window: int, step: int = 1) -> torch.Tensor` — `(n,)` → `(1 + (n − window)//step, window)` via `x.as_strided(...)` using `x.stride(0)` (do not assume it is 1 — test on a sliced input like `y[::2]`). Raise `ValueError` if `window > n`.
2. `def rolling_mean(x: torch.Tensor, window: int) -> torch.Tensor` — `sliding_window_1d(x, window).mean(dim=1)`. Cross-check against Day 1.6's cumsum `moving_average`.
3. `def toeplitz(c: torch.Tensor, r: torch.Tensor) -> torch.Tensor` — first column `c` (len n), first row `r` (len m), `c[0] == r[0]`; `T[i, j] = c[i−j] if i ≥ j else r[j−i]`. Construction: `v = torch.cat([r.flip(0), c[1:]])`, then `T[i, j] = v[m−1+i−j]`; since as_strided forbids negative strides, build the Hankel view `v.as_strided((n, m), (1, 1))` — whose `[i, j']` is `v[i+j']` — and `flip(1)` it. Also write `toeplitz_loop` in the test file as reference.
4. Danger demo `week02/d2_danger.py`: take `x = torch.zeros(6)`, `w = sliding_window_1d(x, 3, 1)`, run `w.add_(1)`, print x — interior elements got incremented once per window covering them (values like `[1, 2, 3, 3, 2, 1]`). End with a printed warning line: "never write through overlapping strided views".
**Verify — done when:**
```python
x = torch.arange(10.)
assert torch.equal(sliding_window_1d(x, 4, 2), x.unfold(0, 4, 2))
y = torch.arange(20.)[::2]                  # non-unit base stride
assert torch.equal(sliding_window_1d(y, 3, 1), y.unfold(0, 3, 1))
assert torch.allclose(rolling_mean(x, 3), moving_average(x, 3))   # Day 1.6 import
c, r = torch.tensor([1., 2., 3.]), torch.tensor([1., 9., 8., 7.])
T = toeplitz(c, r)
assert T.shape == (3, 4) and torch.equal(T, toeplitz_loop(c, r))
assert torch.equal(T[:, 0], c) and torch.equal(T[0], r)
assert torch.equal(T.diagonal(), torch.tensor([1., 1., 1.]))      # constant diagonals
assert sliding_window_1d(x, 4, 2).data_ptr() == x.data_ptr()      # zero copy
```
**If stuck:** the `unfold` doc example maps directly onto your stride tuple; for toeplitz, write out `v` and the Hankel `[i, j']` indices for the 3×4 case by hand.

## Day 2.3 — einsum (~2h)
- [ ] done
**Goal:** read and write any einsum expression by reducing it to three mechanical rules.
**Learn:**
- *The notation* — `"ij,jk->ik"`: each operand gets one letter per dim; the output spec names the dims that survive. One mental rule generates everything: multiply operands over all index combinations, then SUM OVER every letter missing from the output.
- *Repeated letter within one operand = diagonal* — `"ii->i"` walks the diagonal; `"ii->"` sums it (trace). Repeated across operands = contraction (matmul's `j`).
- *No summation, just reordering* — `"ij->ji"` is transpose; einsum with all letters kept is pure permute/elementwise.
- *Batch dims ride along* — letters present in all operands and the output (`b` in `"bij,bjk->bik"`) are vectorized over, not summed. This is why einsum scales to attention shapes without comment.
- *Why bother* — einsum statements carry their own shape documentation, eliminate manual `transpose`/`unsqueeze` gymnastics, and dispatch to the same optimized kernels (matmuls) underneath.
**Read (30–45 min):**
- `torch.einsum`: https://docs.pytorch.org/docs/stable/generated/torch.einsum.html
- Tim Rocktäschel, "Einsum is all you need": https://rockt.github.io/2018/04/30/einsum (a text article with runnable code; read the rules + the first batch of examples).
**Build:** `week02/d3_einsum.py`: ten one-line functions, each `def ein_<name>(...) -> torch.Tensor` containing ONLY a `torch.einsum` call, plus a test comparing each to the reference torch op on random inputs:
1. `ein_matmul(a, b)` ≙ `a @ b` — `"ij,jk->ik"`.
2. `ein_bmm(a, b)` ≙ `torch.bmm` — `"bij,bjk->bik"`.
3. `ein_trace(a)` ≙ `torch.trace` — `"ii->"`.
4. `ein_transpose(a)` ≙ `a.T` — `"ij->ji"`.
5. `ein_outer(a, b)` ≙ `torch.outer` — `"i,j->ij"`.
6. `ein_row_sums(a)` ≙ `a.sum(dim=1)` — `"ij->i"`.
7. `ein_col_sums(a)` ≙ `a.sum(dim=0)` — `"ij->j"`.
8. `ein_bilinear(x, A, y)` ≙ `x @ A @ y` (scalar, `xᵀAy`) — `"i,ij,j->"`.
9. `ein_attn_scores(q, k)` — `(B, S, D)`, `(B, T, D)` → `(B, S, T)` of dot products ≙ `q @ k.transpose(1, 2)` — `"bsd,btd->bst"`. (This exact line reappears in Week 4's attention.)
10. `ein_batch_diag(a)` — `(B, n, n)` → `(B, n)` per-batch diagonals ≙ `torch.diagonal(a, dim1=1, dim2=2)` — `"bii->bi"`.
Write the einsum string for each BEFORE consulting anything, from the rules alone.
**Verify — done when:** every pair agrees:
```python
a, b = torch.randn(3, 4), torch.randn(4, 5)
assert torch.allclose(ein_matmul(a, b), a @ b, atol=1e-6)
q, k = torch.randn(2, 5, 8), torch.randn(2, 7, 8)
assert torch.allclose(ein_attn_scores(q, k), q @ k.transpose(1, 2), atol=1e-6)
m = torch.randn(4, 4)
assert torch.allclose(ein_trace(m), torch.trace(m), atol=1e-6)
x, A, y = torch.randn(3), torch.randn(3, 4), torch.randn(4)
assert torch.allclose(ein_bilinear(x, A, y), x @ A @ y, atol=1e-6)
B = torch.randn(6, 3, 3)
assert torch.allclose(ein_batch_diag(B), torch.diagonal(B, dim1=1, dim2=2), atol=1e-6)
```
(plus analogous asserts for the remaining five).
**If stuck:** for each expression ask only two questions — which letters appear in the output, and which letters got dropped (those are summed). The Rocktäschel post's table of examples covers eight of your ten.

## Day 2.4 — gather, scatter & friends (~2.5h)
- [ ] done
**Goal:** select and write tensor elements by index tensors along arbitrary dims — the address-book of embedding tables, top-k, and sampling code.
**Learn:**
- *`gather(dim, index)`* — `out[i][j] = input[index[i][j]][j]` for dim=0 (the indexed coordinate is replaced by the index tensor's value; all others stay). Index tensor's shape = output shape. It's "select_per_row" generalized to any dim and multiple picks.
- *`scatter_(dim, index, src)` is gather's inverse* — writes `src` values to indexed positions. `scatter_add_` accumulates instead of overwriting, making it the vectorized histogram/`bincount` primitive. (Trailing underscore = in-place, mutates the receiver.)
- *`index_select(dim, index)`* — pick whole rows/columns by a 1-D index tensor; simpler than gather when you want entire slices. `weight.index_select(0, ids)` IS embedding lookup.
- *`masked_select` / `take_along_dim`* — boolean cousin (returns flat copy) and the gather variant aligned with `argsort`/`topk` outputs (`take_along_dim(x, indices, dim)` pairs perfectly with `x.argsort(dim)`).
- *`topk`, `sort`, `argsort`* — `topk(k, dim, largest=)` returns `(values, indices)`; combined with scatter you can build sparse masks (today) and top-k / top-p sampling (later weeks).
**Read (30–45 min):**
- `torch.gather`: https://docs.pytorch.org/docs/stable/generated/torch.gather.html
- `torch.Tensor.scatter_`: https://docs.pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html (the one_hot pattern is in the examples).
- Skim `torch.take_along_dim`, `torch.topk` generated-doc pages.
**Build:** `week02/d4_gather.py` + tests.
1. `def embedding_lookup_select(weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor` — `(V, D)` table, `(n,)` int64 ids → `(n, D)` via `index_select`.
2. `def embedding_lookup_gather(weight, ids)` — same result via `gather(0, ids[:, None].expand(-1, weight.shape[1]))`. Same answer, two address modes — instructive.
3. `def one_hot_scatter(labels: torch.Tensor, num_classes: int) -> torch.Tensor` — `zeros(n, K).scatter_(1, labels[:, None], 1.0)`.
4. `def bincount_scatter(x: torch.Tensor, num_bins: int) -> torch.Tensor` — `(n,)` int64 → `(num_bins,)` counts via `zeros(num_bins).scatter_add_(0, x, torch.ones_like(x, dtype=torch.float32))`.
5. `def topk_mask(x: torch.Tensor, k: int) -> torch.Tensor` — `(B, n)` → same shape with everything except each row's k largest zeroed. Build a zeros tensor, scatter the topk values back by their indices. (This is the core of top-k sampling and MoE routing.)
6. `def sort_rows_by_key(m: torch.Tensor, key: torch.Tensor) -> torch.Tensor` — reorder rows of `(n, d)` by ascending `(n,)` key using `argsort` + `index_select`.
7. `def sort_each_row_take_along(m: torch.Tensor) -> torch.Tensor` — sort every row of `(n, d)` independently using `argsort(dim=1)` + `take_along_dim` (NOT `torch.sort` — the point is pairing index tensors with `take_along_dim`).
8. `def gather_roundtrip(x: torch.Tensor) -> torch.Tensor` — shuffle each row by a random permutation via gather, then invert the shuffle via gather with `argsort` of the permutation; returns the reconstruction. Property: output equals input — gather composed with the argsort of its index is the identity.
**Verify — done when:**
```python
W, ids = torch.randn(10, 4), torch.tensor([3, 3, 7, 0])
assert torch.equal(embedding_lookup_select(W, ids), W[ids])
assert torch.equal(embedding_lookup_gather(W, ids), W[ids])
lab = torch.tensor([0, 2, 1])
import torch.nn.functional as F
assert torch.equal(one_hot_scatter(lab, 3), F.one_hot(lab, 3).float())
x = torch.randint(0, 6, (1000,))
assert torch.equal(bincount_scatter(x, 6).long(), torch.bincount(x, minlength=6))
t = torch.tensor([[3., 1., 4., 1.], [5., 9., 2., 6.]])
mk = topk_mask(t, 2)
assert torch.equal(mk, torch.tensor([[3., 0., 4., 0.], [0., 9., 0., 6.]]))
assert (mk != 0).sum(dim=1).eq(2).all()
key = torch.tensor([3., 1., 2.])
m = torch.eye(3)
assert torch.equal(sort_rows_by_key(m, key), torch.stack([m[1], m[2], m[0]]))
r = torch.randn(4, 6)
assert torch.equal(sort_each_row_take_along(r), torch.sort(r, dim=1).values)
assert torch.equal(gather_roundtrip(r), r)                  # shuffle then unshuffle = identity
```
**If stuck:** the gather doc's three-line formula — substitute your shapes into it literally; for `topk_mask`, print `topk` values and indices for the 2×4 example and hand-trace the scatter.

## Day 2.5 — Devices, RNG & reproducibility (~2h)
- [ ] done
**Goal:** write device-agnostic, exactly-reproducible code; start `tlib/utils.py` for the rest of the program.
**Learn:**
- *Device-agnostic pattern* — pick `device` once at the top (`cuda` if available, else `cpu`), pass it down, and create tensors with `device=device` or `torch.zeros_like(x)` (inherits device+dtype). `.to(device)` returns a NEW tensor (copy across devices, no-op if already there) — it does not mutate.
- *Global vs local RNG* — `torch.manual_seed(s)` seeds the global generator that `rand/randn/randint/randperm` consume. A `torch.Generator().manual_seed(s)` passed via `generator=` gives an isolated stream: your data shuffling stops perturbing your weight init.
- *Same seed ⇒ same sequence, same machine* — bitwise reproducibility holds for a fixed PyTorch version/device; across GPUs or versions only statistically. Some CUDA kernels are nondeterministic; `torch.use_deterministic_algorithms(True)` makes those error instead of silently varying (CPU is mostly deterministic already).
- *`multinomial` & `randperm`* — `torch.multinomial(weights, n, replacement=)` samples indices proportional to non-negative weights (THE sampling primitive for LLM decoding); `randperm(n)` is a random permutation — the standard dataset shuffle.
**Read (30–45 min):**
- Reproducibility note: https://docs.pytorch.org/docs/stable/notes/randomness.html (full page).
- `torch.Generator`: https://docs.pytorch.org/docs/stable/generated/torch.Generator.html
- `torch.multinomial`: https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html
**Build:** `tlib/utils.py` (yes, in tlib — permanent kit) + `week02/test_d5_utils.py`.
1. `def seed_all(seed: int) -> None` — seeds `random.seed`, `np.random.seed`, `torch.manual_seed`, and (guarded by `torch.cuda.is_available()`) `torch.cuda.manual_seed_all`.
2. `def get_device(prefer: str | None = None) -> torch.device` — returns `torch.device(prefer)` if given, else cuda if available, else cpu. (GPU variant: this one function is your whole migration story.)
3. `def make_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator` — isolated seeded stream.
4. `def weighted_sample(weights: torch.Tensor, n: int, replacement: bool = True, seed: int | None = None) -> torch.Tensor` — `multinomial` with a fresh `make_generator(seed)` when seed is given, global RNG otherwise. Raise `ValueError` if `not replacement and n > len(weights)`.
5. `def shuffle_split(x: torch.Tensor, y: torch.Tensor, frac: float, seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]` — shuffle rows with a seeded `randperm(generator=...)`, split into train/test at `frac`; returns `(tr_x, tr_y, te_x, te_y)`. You will call this in every data-loading script from Week 4 on.
6. Export all five from `tlib/__init__.py`.
**Verify — done when:**
```python
from tlib import seed_all, get_device, make_generator, weighted_sample
seed_all(0); a = torch.randn(5)
seed_all(0); b = torch.randn(5)
assert torch.equal(a, b)                                   # same seed, identical draws
assert torch.equal(weighted_sample(torch.ones(10), 5, seed=1),
                   weighted_sample(torch.ones(10), 5, seed=1))
g1, g2 = make_generator(7), make_generator(7)
assert torch.equal(torch.randn(3, generator=g1), torch.randn(3, generator=g2))
w = torch.tensor([0.0, 0.0, 1.0])
assert (weighted_sample(w, 100, seed=0) == 2).all()        # zero-weight never drawn
s = weighted_sample(torch.tensor([1., 3.]), 10_000, seed=0)
frac = (s == 1).float().mean().item()
assert 0.72 < frac < 0.78                                   # approximately 0.75, stochastic
assert isinstance(get_device(), torch.device)
X, Y = torch.randn(10, 3), torch.arange(10)
tr_x, tr_y, te_x, te_y = shuffle_split(X, Y, 0.8, seed=0)
assert tr_x.shape == (8, 3) and te_x.shape == (2, 3)
assert torch.equal(torch.cat([tr_y, te_y]).sort().values, Y)   # a permutation, nothing lost
assert torch.equal(shuffle_split(X, Y, 0.8, seed=0)[0], tr_x)  # seeded => reproducible split
import pytest
with pytest.raises(ValueError):
    weighted_sample(torch.ones(3), 5, replacement=False)
```
**If stuck:** the randomness note's "Controlling sources of randomness" section is a literal checklist; remember `multinomial` takes weights, not probabilities — no normalization needed.
If `shuffle_split` reproducibility fails, check that you passed `generator=` to `randperm` itself — seeding a Generator and then not passing it changes nothing.

## Day 2.6 — Deep build: from strides to ops (~3.5h)
- [ ] done
**Goal:** rebuild matmul from raw strides and conv1d from unfold, then measure why you'd never ship them.
**Learn:**
- *Matmul = broadcast + multiply + reduce* — `C[i, j] = Σ_k A[i, k] B[k, j]`: expand A to `(n, m, k)` and B to `(n, m, k)` as stride-0 views (zero copies), multiply elementwise, sum the last dim. Every contraction decomposes this way.
- *Convolution = sliding windows + contraction* — unfold the padded input into `(B, C_in, L_out, K)` windows, contract with the `(C_out, C_in, K)` weight. Conv "is" matmul over windowed views — the insight behind im2col in real frameworks.
- *Why torch wins anyway* — your version materializes an `n·m·k` product tensor then reduces; BLAS kernels tile the loops to keep operands in cache, vectorize (SIMD), multithread, and never materialize intermediates. Same FLOPs, wildly different memory traffic.
**Read (20–30 min):** `torch.nn.functional.conv1d` (https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.conv1d.html) for the exact shape/padding/stride contract and `L_out` formula; `torch.nn.functional.pad` page.
**Build:** `week02/strided_ops.py` + `week02/test_strided_ops.py`.
1. `def matmul_strided(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor` — `(n, k) @ (k, m)` using ONLY `as_strided` (or `expand`), elementwise `*`, and `sum`. Forbidden: `@`, `mm`, `matmul`, `einsum`, `bmm`. Layout: `Ae = A.as_strided((n, m, k), (A.stride(0), 0, A.stride(1)))`, `Be = B.as_strided((n, m, k), (0, B.stride(1), B.stride(0)))`, return `(Ae * Be).sum(-1)`.
2. `def conv1d_unfold(x, w, bias=None, stride=1, padding=0)` — x `(B, C_in, L)`, w `(C_out, C_in, K)`; `F.pad(x, (padding, padding))`, `.unfold(2, K, stride)` → `(B, C_in, L_out, K)`, contract with `torch.einsum("bclk,ock->bol", windows, w)` (einsum allowed here — yesterday's tool), add `bias[:, None]` if given.
3. Randomized tests: 20 trials each. Matmul: shapes with n, k, m drawn from 1–32, also non-contiguous inputs (`A.t().contiguous().t()` and sliced tensors). Conv: B ≤ 4, C_in/C_out ≤ 8, K ∈ {1, 3, 5}, stride ∈ {1, 2, 3}, padding ∈ {0, 1, 2}, L 8–32, bias on/off.
4. `week02/d6_timing.py` — `time.perf_counter` over ~10 reps after 2 warmup reps: `matmul_strided` vs `@` at n=k=m ∈ {64, 256}; `conv1d_unfold` vs `F.conv1d` at one mid-size config. Print a table plus a 3-line comment on WHERE the slowdown comes from (materialized `(n, m, k)` intermediate = memory traffic; no cache tiling; single fused kernel vs several). Report whatever ratios you measure — do not expect specific numbers, only that torch is faster and the gap grows with size. (GPU variant: on CUDA you must call `torch.cuda.synchronize()` before each timestamp, or you time kernel launches instead of kernels.)
5. Stretch (only if under 3h so far): `def bmm_strided(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor` — batched `(B, n, k) @ (B, k, m)` by the same expand-multiply-reduce scheme (one extra stride-0 dim on each operand), verified against `torch.bmm`.
**Verify — done when:**
```python
for _ in range(20):
    n, k, m = (int(x) for x in torch.randint(1, 33, (3,)))
    A, B = torch.randn(n, k), torch.randn(k, m)
    assert torch.allclose(matmul_strided(A, B), A @ B, atol=1e-5)
At, Bt = torch.randn(8, 8).t(), torch.randn(8, 4)           # non-contiguous left operand
assert torch.allclose(matmul_strided(At, Bt), At @ Bt, atol=1e-5)
import torch.nn.functional as F
x, w, b = torch.randn(2, 3, 16), torch.randn(5, 3, 3), torch.randn(5)
assert torch.allclose(conv1d_unfold(x, w, b, stride=2, padding=1),
                      F.conv1d(x, w, b, stride=2, padding=1), atol=1e-5)
assert conv1d_unfold(x, w, None, 2, 1).shape == F.conv1d(x, w, None, 2, 1).shape
```
plus the timing script runs and prints its table. Zero-copy check: assert inside `matmul_strided` is reachable only via views — `Ae` requires no allocation (you can verify `Ae.data_ptr() == A.data_ptr()`).
**If stuck:** for `matmul_strided` on non-contiguous inputs, use `A.stride(0)/A.stride(1)` symbolically — never assume `(k, 1)`; for conv, check `L_out = (L + 2·padding − K)//stride + 1` against your unfold result's dim 2 before debugging values.

## Day 2.7 — Review, quiz, redo-cold (~2h)
- [ ] done
**Goal:** lock in the memory model before autograd builds on it.
**Read (45 min):** Edward Yang, "PyTorch internals": http://blog.ezyang.com/2019/05/pytorch-internals/ — ONLY the tensor fundamentals/strides material and the TensorImpl section (stop at autograd; that's Week 3). As you read, map each stride diagram onto your `explain_layout` output from Day 2.1 — the post is the authoritative version of the model you built by hand. Then write `week02/notes.md` with the three least obvious facts of the week and one sentence on what `TensorImpl` stores that your `explain_layout` dict also reports.
**Self-quiz (closed book; Answers at the bottom of this file):**
1. For `t = torch.arange(24.).reshape(2, 3, 4)`, what are `t.stride()`, `t.transpose(0, 2).stride()`, and `t[1, :, ::2]`'s stride and storage offset?
2. State the storage-index formula and use it to locate `t[1, 2, 3]` of the tensor above.
3. Why exactly does `t.t().view(-1)` fail while `t.t().reshape(-1)` succeeds? What does the latter cost?
4. What stride value implements `expand`, and what goes wrong if you write through such a view?
5. Give the as_strided `(size, stride)` for non-overlapping windows of length 4, step 4, over a contiguous `(16,)` tensor. Now overlapping with step 2.
6. Write einsum strings for: trace, batched matmul, `xᵀAy`, per-batch diagonal of `(B, n, n)`.
7. In `out = x.gather(1, idx)`, state the formula for `out[i][j]`. What shape must `idx` have?
8. How do you build `bincount` from `scatter_add_`? Why must the destination start at zeros?
9. Difference between `torch.manual_seed(0)` and passing `generator=torch.Generator().manual_seed(0)` — when does the distinction matter in a training script?
10. Your `matmul_strided` and `A @ B` do the same FLOPs. Name two concrete reasons torch is faster anyway.
11. `x.unfold(0, 3, 1)` then `result.add_(1)` — what's wrong?
12. `conv1d` with L=16, K=3, stride=2, padding=1 — compute L_out.
**Redo cold (no docs, fresh file `week02/d7_cold.py`):**
1. Predict strides/offset on paper for five fresh expressions over `torch.arange(36.).reshape(3, 3, 4)` (a permute, two slices, an expand, an unsqueeze), then assert your predictions.
2. `sliding_window_1d` via as_strided, verified against `unfold` — including a non-contiguous input.
3. `topk_mask` and `bincount_scatter` from memory, verified against torch.
4. einsum for attention scores `(B, S, D), (B, T, D) → (B, S, T)` verified against `q @ k.transpose(1, 2)`.
5. `matmul_strided` from memory for square 8×8, verified with `allclose`.
Anything that required docs: schedule a redo in three days. Week 3 (autograd) starts by asking what `.backward()` does to views — your Day 2.1 answers are the prerequisite.

---

## Answers (Day 2.7 quiz)
1. `t.stride() == (12, 4, 1)`. Transpose(0, 2) permutes it: `(1, 4, 12)`. `t[1, :, ::2]`: shape `(3, 2)`, stride `(4, 2)`, storage offset 12.
2. `storage_index = offset + Σ idx[k]·stride[k]`. For `t[1, 2, 3]`: `0 + 1·12 + 2·4 + 3·1 = 23` (the last element).
3. After transpose, memory order disagrees with row-major logical order, and no single (offset, strides) over the same storage yields the flattened sequence — `view` refuses. `reshape` detects this and copies into fresh contiguous memory: it succeeds at the cost of an allocation + copy, and the result no longer aliases the original.
4. Stride 0 on the expanded dim: every index along it reads the same cell. Writing through it makes one physical write appear at all logical positions (and racy semantics for in-place accumulation) — which is why expanded tensors refuse many in-place ops.
5. Non-overlapping: `size=(4, 4), stride=(4, 1)`. Overlapping step 2: `size=(7, 4), stride=(2, 1)` — 7 because `1 + (16 − 4)//2 = 7`.
6. Trace `"ii->"`; batched matmul `"bij,bjk->bik"`; bilinear `"i,ij,j->"`; per-batch diagonal `"bii->bi"`.
7. For dim=1: `out[i][j] = x[i][idx[i][j]]`. `idx` has the output's shape: same rank as x, same size as x in all dims except dim 1, where it has as many entries as picks you want; int64.
8. `torch.zeros(num_bins).scatter_add_(0, values, torch.ones_like(values, dtype=torch.float32))` — each occurrence adds 1 at its bin. Zeros because scatter_add_ accumulates INTO existing contents; garbage start = garbage counts.
9. Both produce reproducible streams. The global seed couples every consumer: adding one extra `randn` call (say, a new data augmentation) shifts all subsequent global draws, silently changing your weight init. Per-purpose `Generator` objects isolate streams so components don't perturb each other.
10. (a) No materialized `(n, m, k)` intermediate — your version writes and re-reads n·m·k elements of memory, BLAS accumulates in registers; (b) cache-tiled, SIMD-vectorized, multithreaded kernels (plus a single fused kernel instead of several dispatched ops).
11. The windows overlap, so they alias storage; in-place `add_` writes each shared cell multiple times (interior cells get +1 per covering window) and the "windows" no longer hold their original values. Clone before mutating.
12. `L_out = (L + 2p − K)//s + 1 = (16 + 2 − 3)//2 + 1 = 7 + 1 = 8`.
