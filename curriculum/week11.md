# Week 11 — Performance Engineering

You have a correct, working LM stack (`tlib` + DecoderLM + BPE tokenizer + Trainer). This week you learn to make it *fast* — but more importantly, you learn the discipline of performance work: measure first, change one thing, verify correctness, measure again. Everything here is sequenced before Week 12 because the pretraining run is hours long; every optimization you bank now is wall-clock time saved then. The methodology (profiling, dataloading, compile) is fully CPU-native; AMP and attention days include CPU-bf16 paths with GPU variants clearly marked.

**Week outcome:** `tlib/bench.py` (timing + throughput utilities), `tlib/lmdata.py` (memmap token dataset), AMP + activation-checkpointing options in your Trainer, a `torch.compile`d training script, and — the centerpiece — a cumulative speed-run table in `LOG.md` showing measured before/after numbers for every change, including the ones that didn't help. Skill: you can profile any PyTorch workload, name its bottleneck with evidence, and never report a speedup you didn't measure.

## Day 11.1 — Measurement Methodology & Profiling (~2.5h)
- [ ] done
**Goal:** Build a benchmarking utility you trust, and produce a profiler trace that names the top-3 ops in one training step of your Week-10 model.
**Learn:**
- **Never trust a single timing.** First iterations pay one-time costs (allocator warmup, lazy init, cache misses); background load adds variance. Always: warmup iterations, then N timed iterations, report mean ± std.
- **CPU timing** is `time.perf_counter()` around the call. **GPU timing lies** without `torch.cuda.synchronize()`: CUDA kernels launch *asynchronously* — the Python call returns as soon as the kernel is queued, so naive timing measures launch overhead, not compute. Synchronize before starting and before stopping the clock.
- **Tokens/sec is the LM metric.** ms/step depends on batch and seq length; `tokens/sec = batch × seq_len / step_time` is comparable across configs and is what you'll quote all week.
- **torch.profiler** records every op with timing, shapes (`record_shapes=True`), and memory (`profile_memory=True`). `prof.key_averages().table(sort_by="self_cpu_time_total")` gives a top-ops table; **self time** is time in the op itself, excluding children — that's what tells you where compute actually goes.
- **Chrome traces.** `prof.export_chrome_trace("trace.json")` produces a timeline you open in `chrome://tracing` or https://ui.perfetto.dev — you literally see gaps (input-bound) vs solid compute.
**Read (30–45 min):**
- PyTorch Profiler recipe — all sections, esp. "Using profiler to analyze execution time" and "Examining stack traces": https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- `torch.profiler` API docs — `profile` args (`activities`, `record_shapes`, `profile_memory`, `schedule`): https://docs.pytorch.org/docs/stable/profiler.html
- Performance Tuning Guide — skim "General optimizations" to preview the week: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
**Build:**
1. Create `tlib/bench.py`:
   ```python
   def timed(fn: Callable[[], Any], warmup: int = 3, iters: int = 10,
             device: str = "cpu") -> dict[str, float]:
       """Run fn warmup times untimed, then iters times timed.
       Returns {"mean_ms": ..., "std_ms": ..., "iters": ...}.
       If device == "cuda", calls torch.cuda.synchronize() before each
       clock read. Uses time.perf_counter()."""

   def throughput(step_fn: Callable[[], Any], batch_size: int, seq_len: int,
                  warmup: int = 3, iters: int = 10, device: str = "cpu") -> dict[str, float]:
       """Wraps timed(); adds "tokens_per_sec" = batch_size*seq_len / (mean_ms/1000)."""
   ```
   Implementation notes: collect per-iteration times in a list and compute mean/std yourself
   (`statistics.mean/stdev`) — you want the raw samples available for debugging outliers.
   Add `if device == "cuda": torch.cuda.synchronize()` immediately before *each* clock read.
2. Write `week11/profile_step.py`: load your Week-10 DecoderLM config, build one real `(x, y)` batch, define `step()` = forward + loss + backward + optimizer step + `zero_grad`. Benchmark it with `throughput(...)` and print the dict.
3. In the same script, profile 5 steps:
   ```python
   from torch.profiler import profile, ProfilerActivity

   acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if torch.cuda.is_available() else [])
   with profile(activities=acts, record_shapes=True, profile_memory=True) as prof:
       for _ in range(5):
           step()
   print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15))
   prof.export_chrome_trace("week11/trace_baseline.json")
   ```
4. Open the trace in Perfetto. In `LOG.md`, record: baseline ms/step ± std, tokens/sec, and the top-3 ops by self time *with their percentages copied from the table* (expect matmul-family ops like `aten::addmm`/`aten::mm` to dominate — but write what YOU see).
**Verify — done when:**
- `timed(lambda: torch.randn(256, 256) @ torch.randn(256, 256))` returns a dict with a nonzero `std_ms` (add `assert out["std_ms"] >= 0 and "mean_ms" in out` in a quick pytest in `tests/test_bench.py`).
- The profiler table prints and `week11/trace_baseline.json` exists and opens in Perfetto.
- `LOG.md` has the baseline entry with numbers and the top-3 ops. This baseline is the denominator for the entire week — don't lose it.
**If stuck:** Profiler recipe (URL above); `torch.profiler.profile` docstring via `help(torch.profiler.profile)`; Perfetto: drag-drop the JSON onto https://ui.perfetto.dev.

## Day 11.2 — Dataloading & Step-Overhead Hygiene (~2h)
- [ ] done
**Goal:** Determine whether your training loop is compute-bound or input-bound, then eliminate input-pipeline cost with a memmapped pre-tokenized dataset.
**Learn:**
- **The constant-batch test.** Time N steps with real data loading, then time N steps reusing one pre-loaded batch. The difference *is* your input pipeline cost. If it's ~0, stop optimizing dataloading — you're compute-bound.
- **Pre-tokenize once, not per epoch.** Tokenizing text inside `__getitem__` re-does identical work every epoch. Encode the whole corpus once to a binary file of token IDs; training then just reads integer windows.
- **`np.memmap`** maps a file into virtual memory — the OS pages in only the bytes you touch. You can "open" a multi-GB token file instantly and slice windows out of it with zero parsing. **uint16** holds 0–65535, so it fits any vocab ≤ 65536 in 2 bytes/token — half the size of int32, and your vocab (≤ ~8k) fits easily.
- **`optimizer.zero_grad(set_to_none=True)`** (default in 2.x, but be explicit): sets `.grad = None` instead of filling with zeros — skips a memset per param per step, and the next backward writes grads directly instead of accumulating into zeros.
- **GPU-only hygiene:** `pin_memory=True` in the DataLoader + `tensor.to("cuda", non_blocking=True)` lets host→device copies overlap compute; and every `.item()`, `print(loss)`, or `tensor.cpu()` in the hot loop forces a sync that stalls the GPU pipeline — log every K steps, not every step.
**Read (30–45 min):**
- Performance Tuning Guide — "Enable asynchronous data loading and augmentation", "Set gradients to None", "Avoid unnecessary CPU-GPU synchronization": https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- `torch.utils.data` docs — Dataset/DataLoader, `num_workers`, `pin_memory`: https://docs.pytorch.org/docs/stable/data.html
- NumPy memmap docs: https://numpy.org/doc/stable/reference/generated/numpy.memmap.html
**Build:**
1. Run the constant-batch test on yesterday's `step()`:
   ```python
   # A: real pipeline — each iteration fetches the next batch from your dataset
   real = throughput(lambda: step(next(data_iter)), B, T)
   # B: cached batch — same compute, zero input cost
   xb, yb = next(data_iter)
   cached = throughput(lambda: step((xb, yb)), B, T)
   # input pipeline cost per step = real mean_ms - cached mean_ms
   ```
   Record both rows in your table; the difference is the most you can ever win from dataloading work.
2. Write `week11/pretokenize.py`: load your Gutenberg corpus, encode with your `BPETokenizer`, `assert max(ids) < 65536`, write `np.array(ids, dtype=np.uint16).tofile("data/corpus.bin")`. Print total token count.
3. Create `tlib/lmdata.py`:
   ```python
   class TokenFileDataset(torch.utils.data.Dataset):
       """LM dataset over a memmapped uint16 token file.
       __getitem__(i) returns (x, y): int64 tensors of shape (block_size,),
       where x = tokens[i:i+block_size], y = tokens[i+1:i+block_size+1]."""
       def __init__(self, path: str | Path, block_size: int) -> None: ...
       def __len__(self) -> int: ...  # n_tokens - block_size
       def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]: ...
   ```
   Implementation core:
   ```python
   def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
       chunk = self.tokens[i : i + self.block_size + 1].astype(np.int64)  # one read
       t = torch.from_numpy(chunk)
       return t[:-1], t[1:]
   ```
   Open the memmap in `__init__` with `np.memmap(path, dtype=np.uint16, mode="r")`. The `astype(np.int64)` copy is required — embeddings index with int64, and you must not hand autograd a tensor aliasing a read-only mmap.
4. Swap the Trainer's dataset for `TokenFileDataset`; ensure `zero_grad(set_to_none=True)` is what your Trainer calls. Re-measure → two more table rows (change | ms/step | tokens/sec | Δmemory if observable via `psutil.Process().memory_info().rss`).
5. **GPU variant (note for later, implement if you have a GPU today):** `DataLoader(..., pin_memory=True, num_workers=2)` and `x = x.to("cuda", non_blocking=True)`; move per-step `loss.item()` logging to every-K-steps. Each is one table row.
**Verify — done when:**
- Round-trip check: decode the first 200 tokens read from the memmap and confirm they reproduce the start of your corpus text.
- `tests/test_lmdata.py`: for a tiny synthetic token file, assert `x[1:] == y[:-1]` (shift-by-one alignment) and shapes are `(block_size,)`.
- The table has rows for: real-data vs constant-batch, after memmap, after `set_to_none`. Honest result expected: memmap helps a lot if you tokenized on the fly before, little if you already cached tensors. Write down which.
**If stuck:** the alignment test failing usually means an off-by-one in `__len__` or the slice; print `tokens[:10]` from both memmap and tokenizer output side by side; `np.memmap` docs (above) for dtype/mode pitfalls.

## Day 11.3 — Mixed Precision (~2h)
- [ ] done
**Goal:** Add an AMP option to your Trainer (CPU-bf16 path included), verify the loss curve still tracks fp32, and measure speed/memory honestly.
**Learn:**
- **Float anatomy.** fp32 = 1 sign / 8 exponent / 23 mantissa bits. **bf16** = 1/8/7 — *same exponent width as fp32*, so same dynamic range (~1e-38 to ~3e38) but only ~3 decimal digits of precision. **fp16** = 1/5/10 — more mantissa than bf16 but a tiny exponent: max ~65504, min normal ~6e-5, so small gradients *underflow to zero*.
- **That's why fp16 needs loss scaling and bf16 doesn't:** `GradScaler` multiplies the loss by a large factor so gradients land in fp16's representable range, then unscales before the optimizer step. bf16's range matches fp32 — no scaler needed.
- **`torch.amp.autocast(device_type=..., dtype=...)`** is a context manager that runs *eligible* ops in the low-precision dtype. It is per-op, not blanket casting.
- **What stays fp32 and why:** reductions, softmax, norms, and loss functions accumulate many values — error compounds — so autocast keeps them fp32. Matmuls and convs are compute-heavy and error-tolerant, so they get the low-precision dtype. (The op lists are in the AMP docs.)
- **CPU reality:** `autocast("cpu", dtype=torch.bfloat16)` is fully supported; whether it's *faster* depends on your CPU (AVX-512 BF16/AMX help; older cores may see no gain or a slowdown). Measuring this honestly is today's actual lesson.
**Read (30–45 min):**
- AMP docs — "Autocasting", "Gradient Scaling", and the "CPU Op-Specific Behavior" / "CUDA Op-Specific Behavior" sections: https://docs.pytorch.org/docs/stable/amp.html
- AMP recipe — the full worked training-loop example: https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
- AMP examples note — "Working with scaled gradients" (clipping interacts with the scaler!): https://docs.pytorch.org/docs/stable/notes/amp_examples.html
**Build:**
1. Add to your Trainer: `amp_dtype: torch.dtype | None = None`. When set, wrap forward+loss in `torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype)`. (Note the 2.x API: `torch.amp.autocast`, NOT the deprecated `torch.cuda.amp.autocast`.)
2. Scaler path (GPU fp16 only). The full step, with clipping in the right place:
   ```python
   scaler = torch.amp.GradScaler("cuda")          # only for fp16-on-cuda
   with torch.amp.autocast(device_type=dev, dtype=self.amp_dtype):
       loss = self.loss_fn(model(x), y)
   if scaler is not None:
       scaler.scale(loss).backward()
       scaler.unscale_(optimizer)                  # BEFORE clipping — grads are scaled!
       torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
       scaler.step(optimizer); scaler.update()
   else:                                           # fp32 or bf16 (CPU or GPU)
       loss.backward()
       torch.nn.utils.clip_grad_norm_(model.parameters(), self.clip)
       optimizer.step()
   ```
3. Correctness run: fixed seed, 200 steps on your LM, once fp32 and once bf16. Save both loss CSVs, plot together. They will NOT match step-for-step; they should have the same shape and end within a loose band — a reasonable check: `abs(final_bf16 - final_fp32) / final_fp32 < 0.05`, plus eyeball the plot. If bf16 diverges or NaNs, that's a bug, not noise.
4. Measure bf16 vs fp32 ms/step + tokens/sec + RSS with `tlib/bench.py` → table rows. Record the result even (especially) if bf16 is *slower* on your CPU. **GPU variant:** repeat with `device_type="cuda"`, bf16 (Ampere+) and fp16+scaler; expect real gains there.
**Verify — done when:**
- bf16 run completes 200 steps with finite loss; relative-final-loss check above passes; the overlay plot is saved to `week11/amp_loss_compare.png`.
- Table has fp32 vs bf16 rows with measured numbers and a one-line honest interpretation.
- Trainer with `amp_dtype=None` is bit-identical to the old path (same fixed-seed 20-step losses, `torch.allclose`).
**If stuck:** AMP examples note (gradient clipping section) if clipping+scaler ordering confuses you; CPU Op-Specific Behavior list in the AMP docs if some op errors under bf16 autocast.

## Day 11.4 — torch.compile (~2.5h)
- [ ] done
**Goal:** Compile your training step, separate warmup cost from steady-state speedup, find and fix a graph break, and prove compiled == eager numerically.
**Learn:**
- **Eager vs graphs.** Eager runs one op at a time from Python — flexible, but every op pays Python + dispatch + a round-trip through memory. A captured graph lets a compiler see the whole computation and optimize across op boundaries.
- **The 2.x stack:** **TorchDynamo** hooks CPython frame evaluation to capture your Python into FX graphs *while keeping Python semantics* (it falls back when it can't trace); **TorchInductor** lowers graphs to generated code (C++/OpenMP on CPU, Triton on GPU).
- **Fusion is the big win for memory-bound code.** An elementwise chain like `x.mul(a).add(b).relu()` in eager does 3 reads + 3 writes of the whole tensor; fused, it's 1 read + 1 write. Pointwise chains (bias+gelu, residual+norm pieces) fuse beautifully; matmuls mostly stay matmuls.
- **Graph breaks** happen when Dynamo hits something it can't trace: data-dependent Python control flow on tensor *values*, `.item()`, `print(tensor)`, unsupported builtins. A break splits the graph — you keep correctness but lose cross-boundary fusion. See them with `TORCH_LOGS=graph_breaks python train.py`.
- **Warmup cost is real:** the first call (and each new input-shape "recompile") takes seconds of compilation. Report first-step time and steady-state time as *separate* numbers, always.
- **`mode=`:** `"default"`, `"reduce-overhead"` (CUDA graphs — GPU), `"max-autotune"` (longer compile, autotuned kernels). On CPU, start with default.
**Read (30–45 min):**
- torch.compile tutorial — "Basic Usage", "Demonstrating Speedups", and the comparison-to-eager sections: https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
- ezyang, "Ways to use torch.compile" — realistic expectations, selective compilation: https://blog.ezyang.com/2024/11/ways-to-use-torch-compile/
- torch.compiler docs landing page — skim the Dynamo/Inductor overview: https://docs.pytorch.org/docs/stable/torch.compiler.html
**Build:**
1. In `week11/compile_bench.py`: build model, then `cmodel = torch.compile(model)`. Time step 1 alone (`timed(..., warmup=0, iters=1)`) → "first step (incl. compile)" row. Then `timed` with warmup=5 → "steady state" row. Compare against eager rows from your baseline. Expect the first step to take *seconds* — that's Dynamo tracing + Inductor codegen, paid once per shape.
2. Graph-break hunt:
   ```bash
   TORCH_LOGS=graph_breaks python week11/compile_bench.py 2> breaks.log
   ```
   If breaks appear in your code (a `print(loss.item())` per step or an `if loss < x:` is a classic), fix one (move logging out of the compiled region / use the periodic-K-steps pattern) and confirm the break disappears from the log. If your code is clean, *manufacture* one: add `if x.sum().item() > 0: pass` inside the model's forward, observe the logged break (it will name the line and a reason like `Tensor.item`), then remove it — write both log excerpts into `LOG.md`. Programmatic alternative:
   ```python
   explanation = torch._dynamo.explain(model)(x)
   print(explanation.graph_break_count, explanation.break_reasons)
   ```
3. Correctness: fixed weights, fixed batch; `out_eager = model(x)`, `out_compiled = cmodel(x)`; assert `torch.allclose(out_eager, out_compiled, atol=1e-4, rtol=1e-4)` — loose tolerance because fused kernels legally reassociate floating-point math.
4. Add `--compile` flag to your training script; measure compiled steady-state training step → table row.
**Verify — done when:**
- The allclose check passes; first-step vs steady-state timings recorded separately in the table.
- `LOG.md` shows a graph-break log line (real or manufactured) and the after-fix run without it.
- Loss over 50 fixed-seed steps compiled vs eager tracks closely (overlay plot or final-loss check as on Day 11.3).
**If stuck:** torch.compile tutorial troubleshooting section; `torch._dynamo.explain(model)(x)` prints break reasons programmatically; ezyang post for "is this expected?" calibration.

## Day 11.5 — Memory: Checkpointing & the Batch-Size Search (~2h)
- [ ] done
**Goal:** Estimate where your activation memory goes, add activation checkpointing to your transformer blocks, and find max batch size empirically before/after.
**Learn:**
- **Activations dominate training memory.** Weights+grads+Adam state are fixed (~16 bytes/param fp32: 4 weight + 4 grad + 8 Adam moments); activations scale with `batch × seq × d_model × n_layers × c` (c ≈ 10–20 depending on what's saved per block: attention internals, MLP hidden at 4×d, norms). Compute the estimate for YOUR config — it's a Fermi estimate, expect ±2×.
- **Activation (gradient) checkpointing:** `torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)` doesn't store the block's internal activations in forward; it *recomputes* the block during backward. Trade: ~+33% forward compute for dropping per-block activation storage to just block boundaries.
- **It's exact, not approximate** — recomputation is the same math, so loss/grads match the uncheckpointed run bit-for-bit-ish (allclose-level; same kernels rerun).
- **Gradient accumulation is the other lever** (your Trainer already has it): K micro-batches of size B behave like batch K×B for memory of a single B — the levers compose: checkpointing raises max micro-batch, accumulation raises effective batch.
- **Measuring memory:** GPU — `torch.cuda.max_memory_allocated()` / `memory_summary()` are precise. CPU — `psutil.Process().memory_info().rss` is coarse (allocator caching, Python overhead included); treat CPU numbers as trend, not truth.
**Read (30–45 min):**
- `torch.utils.checkpoint` docs — esp. the `use_reentrant=False` recommendation and its caveats: https://docs.pytorch.org/docs/stable/checkpoint.html
- Performance Tuning Guide — GPU memory sections (for the GPU variant): https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
**Build:**
1. In `LOG.md`, write your model's memory budget. Template:
   ```
   params P = ...            # sum(p.numel() for p in model.parameters())
   fixed   = 16 * P bytes    # fp32: 4 weight + 4 grad + 8 Adam (m, v)
   activations ≈ batch * seq * d_model * n_layers * c * 4 bytes   # c ≈ 10–20
   # e.g. B=8, T=256, d=256, L=6, c=14 → 8*256*256*6*14*4 ≈ 176 MB
   ```
   Compute it for YOUR config and compare against observed RSS growth during a step — within ±2× counts as understanding.
2. Add to `DecoderLM`: `self.use_checkpoint: bool = False`; in forward:
   ```python
   from torch.utils.checkpoint import checkpoint
   for block in self.blocks:
       if self.use_checkpoint and self.training:
           x = checkpoint(block, x, use_reentrant=False)
       else:
           x = block(x)
   ```
3. Write `week11/max_batch.py`:
   ```python
   def find_max_batch(make_step: Callable[[int], Callable[[], None]],
                      start: int = 1, limit: int = 4096) -> int:
       """Doubling search: try batch sizes 1,2,4,... until the step raises
       (RuntimeError/MemoryError) or exceeds limit; return last success."""
   ```
   On GPU catch CUDA OOM and `torch.cuda.empty_cache()` between tries; on CPU set a sane `limit` so you don't trigger the WSL2 OOM-killer (watch RSS, stop near ~70% of RAM).
4. Measure with/without checkpointing: max batch, ms/step at a *common* batch size, tokens/sec at each config's *own* max batch → table rows. The interesting question: does (bigger batch × slower step) win on tokens/sec?
**Verify — done when:**
- Exactness check passes: fixed batch + fixed weights, loss with `use_checkpoint=True` vs `False` — `torch.allclose(loss_a, loss_b)`, and grads of 2–3 named params allclose too.
- `find_max_batch` returns and prints a number in both configs; table rows recorded.
- Memory-budget paragraph exists in `LOG.md`.
**If stuck:** checkpoint docs (above) for the `use_reentrant` semantics; if grads differ, check you're not checkpointing through dropout without a fixed seed state — run the check in `model.eval()` mode or with dropout=0 first.

## Day 11.6 — Deep Build: The Speed-Run (~3.5h)
- [ ] done
**Goal:** Starting from the re-measured Week-10 baseline, apply the full optimization menu one change at a time, with a correctness check after each, producing the cumulative table.
**Learn:**
- **One variable at a time.** Stacked changes can mask regressions (one +30%, one −10% looks like +20%). The cumulative table with one row per change is the professional artifact.
- **Re-measure the baseline first.** Your machine today ≠ your machine last week (thermals, background load, library versions). Never compare against stale numbers.
- **Shape "niceness":** kernels are written for tiles (multiples of 8/16/64). Padding vocab to a multiple of 64 can make the giant lm_head matmul measurably faster, especially with tensor cores on GPU — the extra rows are dead weight the optimizer happily ignores. On CPU the effect is often small. Measure; report honestly.
- **Negative results are results.** On CPU, bf16 or compile may not help, or may hurt. A table row saying "bf16: 0.97× — not adopted" is exactly as valuable as a win.
**Read (15 min, reference as needed):** Your own Week-11 notes + Performance Tuning Guide as lookup: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
**Build:**
1. `week11/speedrun.py` (or flags on your training script). Protocol, in order, each step = (apply change → correctness check → `throughput` measurement → table row):
   - **Row 0:** baseline re-measured (Week-10 config, on-the-fly data, fp32, eager).
   - **Row 1:** memmap `TokenFileDataset`.
   - **Row 2:** `zero_grad(set_to_none=True)` (confirm/force).
   - **Row 3:** bf16 autocast (`torch.amp.autocast("cpu", dtype=torch.bfloat16)`).
   - **Row 4:** `torch.compile` (steady-state; note compile time separately).
   - **Row 5:** activation checkpointing — adopt only if it raises tokens/sec via a bigger batch; otherwise record "tried, not adopted" with the number.
   - **Row 6:** pad vocab to a multiple of 64 (re-init lm_head/embedding with padded vocab; correctness = loss on real tokens unaffected since padded IDs never occur).
2. Correctness check after each row: 30 fixed-seed steps, loss curve overlaid on baseline (exact-match expected for rows 1–2 and 5–6; loose-band for 3–4). Keep a `week11/speedrun_losses/` dir of CSVs.
3. Table format in `LOG.md` (every cell measured, no projections):
   ```
   | change                  | ms/step | tokens/sec | × vs baseline | adopted? |
   |-------------------------|---------|------------|---------------|----------|
   | 0 baseline (re-measured)|         |            | 1.00          | —        |
   | 1 memmap dataset        |         |            |               |          |
   | 2 set_to_none           |         |            |               |          |
   | 3 bf16 autocast         |         |            |               |          |
   | 4 torch.compile         |         |            |               |          |
   | 5 act. checkpoint+bigB  |         |            |               |          |
   | 6 vocab pad to 64       |         |            |               |          |
   ```
4. Write the 10-line analysis: biggest win and *why* (tie it to Day-1 profile evidence — did the top-3 ops change?), what didn't help and your best supported explanation, what you'd try next on GPU.
5. Re-profile the final config (Day-1 script) and save `week11/trace_final.json`; note in `LOG.md` how the op mix shifted.
**GPU variant (if you have one today):** same protocol, plus: `torch.set_float32_matmul_precision("high")` (TF32) as its own row; confirm your attention goes through `F.scaled_dot_product_attention` and check which backend runs via `torch.nn.attention.sdpa_kernel` (https://docs.pytorch.org/docs/stable/generated/torch.nn.attention.sdpa_kernel.html) — flash vs efficient vs math; `pin_memory` + `non_blocking` row.
**Verify — done when:**
- The cumulative table exists with a measured number in every cell — no "expected ~2×" anywhere, only measured values.
- Every adopted row has a passing correctness artifact; final config's 30-step loss curve matches baseline within the stated tolerance.
- The 10-line analysis names at least one change that didn't help.
**If stuck:** any single row regressing badly → bisect by toggling that flag alone off the final config; profiler from Day 1 re-run on the final config tells you what the *new* bottleneck is.

## Day 11.7 — Review, Quiz & Redo-Cold (~1.5h)
- [ ] done
**Goal:** Consolidate the week; prove you can reproduce the core skills without notes.
**Learn (review):** Re-read your `LOG.md` week-11 entries end to end. For each table row, can you explain the mechanism behind the number? If not, re-read that day's docs section.
**Self-quiz (write answers, then check the Answers section at the bottom of this file):**
1. Why does `t0 = perf_counter(); model(x); dt = perf_counter() - t0` report a misleadingly small time on GPU, and what two lines fix it?
2. Why do we report tokens/sec rather than ms/step when comparing LM configs?
3. From memory: bf16 vs fp16 — bit layouts, and which one needs a GradScaler and why.
4. Which op families does autocast keep in fp32, and what's the numerical reason?
5. Give three distinct causes of a torch.compile graph break and the env var that reveals them.
6. Why do memory-bound elementwise chains benefit more from fusion than matmuls do?
7. Why does uint16 suffice for your token file, and what's the storage saving vs int64 tensors?
8. What exactly does `zero_grad(set_to_none=True)` skip?
9. Activation checkpointing trades ___ for ___; why is the loss still exactly equal (allclose) to the uncheckpointed run?
10. Your step time didn't change after switching to memmap data. What does that tell you, and which Day-2 measurement predicted it?
11. Why might padding vocab to a multiple of 64 speed up the lm_head matmul?
12. Why must the speed-run apply one change at a time?
**Redo-cold drills (no notes, ~15 min each):**
- Rewrite `timed()` from a blank file, including the GPU-sync branch, and make `tests/test_bench.py` pass.
- Write down the activation-memory estimate formula and compute it for your Week-12 candidate config (batch 16, seq 256, your d_model/n_layers).
- From a blank script: profile 3 steps of any small model and print the top-5 self-time table.
- State the three fp16/bf16/fp32 bit layouts and the autocast call for CPU bf16, from memory.

---

## Answers (Day 11.7)
1. CUDA kernels execute asynchronously — the Python call returns when the kernel is *enqueued*, not done, so you time launch overhead. Fix: `torch.cuda.synchronize()` immediately before reading the clock at both start and end.
2. ms/step changes whenever batch or seq length changes, so it can't compare configs; tokens/sec = batch×seq/step_time normalizes to useful work per second — the thing you actually care about for a pretraining run.
3. fp32 = 1/8/23 (sign/exp/mantissa), bf16 = 1/8/7, fp16 = 1/5/10. fp16 needs GradScaler: its 5-bit exponent gives min normal ~6e-5, so small gradients underflow to zero unless the loss is scaled up first. bf16 shares fp32's 8-bit exponent (same range), so no scaler.
4. Reductions/softmax/norms/losses stay fp32: they sum many values, and rounding error compounds across the accumulation; matmuls/convs are error-tolerant and compute-dominated, so they run low-precision.
5. (a) Data-dependent Python control flow on tensor values (`if loss < 1.0:`), (b) `.item()` / `tensor.tolist()` forcing a graph-exiting value read, (c) `print(tensor)` / unsupported builtins or C extensions. Reveal with `TORCH_LOGS=graph_breaks`.
6. Elementwise chains are memory-bound: each eager op reads and writes the full tensor, so N ops = N round-trips through memory; fusion collapses them to one read + one write. Matmuls are compute-bound — already limited by FLOPs, not memory traffic, so fusion saves little.
7. uint16 represents 0–65535, ≥ any vocab ≤ 65536 (yours is ≤ ~8k). 2 bytes/token vs 8 for int64 → 4× smaller on disk; you cast windows to int64 only at batch time.
8. The per-step memset of every gradient tensor to zeros (and the subsequent grad accumulation *into* those zeros): with `None` grads, the next backward allocates/writes grads directly.
9. Trades extra compute (re-running each block's forward during backward, ~+33% forward FLOPs) for not storing per-block internal activations. Exact because the recomputation performs the identical operations on identical inputs — it's the same math, just later.
10. Your loop is compute-bound: input-pipeline cost was already negligible. The Day-2 constant-batch test predicted it — real-data and cached-batch step times were ~equal.
11. Matmul kernels process tiles (multiples of 8/16/64 rows/cols); a vocab dimension that's an exact tile multiple avoids remainder/edge-case code paths and (on GPU) keeps tensor cores fully utilized. Effect is hardware-dependent — hence measure.
12. So each row's delta is attributable to exactly one mechanism; stacked changes can hide a regression inside a net win, and you can't decide what to keep (or debug a correctness failure) without isolation.
