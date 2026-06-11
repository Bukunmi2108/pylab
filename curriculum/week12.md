# Week 12 — Scaling Out, Extending PyTorch & the Pretraining Run

This week converts everything you've built into a real pretrained base model. First you learn the distributed substrate (process groups, collectives, DDP) from first principles on CPU with gloo — no GPU required to understand it deeply, and the parity proofs are *exact*. Mid-week you go under autograd's hood one more time at tensor scale (a fused cross-entropy `autograd.Function`), then you prep and launch THE RUN: your own DecoderLM, your own BPE tokenizer, your own Trainer, your Week-11 optimizations, hours of training, a loss curve, and sample generations that visibly improve. It's sequenced last because it consumes every prior week — and its output checkpoint is the base model for Weeks 13–14.

**Week outcome:** `week12/collectives_play.py` (verified collectives on 4 CPU procs), a manual data-parallel trainer proven param-identical to a single-process big-batch run and then re-proven with DDP, `FusedSoftmaxCrossEntropy` passing `gradcheck`, a pretokenized corpus in memmap shards, and `checkpoints/base_final.pt` + tokenizer + loss curve + `samples.txt` — **the Week 13–14 base model**. Skill: you can explain and verify data parallelism mathematically, drive `torchrun`, extend autograd, and execute a multi-hour training run with resumability.

## Day 12.1 — torch.distributed Fundamentals (~2h)
- [ ] done
**Goal:** Run a 4-process CPU job with torchrun and verify each core collective with asserts that pass on every rank.
**Learn:**
- **Process group:** N independent Python processes (not threads) that can exchange tensors. Each has a **rank** (0..N−1, its identity), shares a **world_size** (N), and a **local_rank** (rank within one machine — matters for picking a GPU; on one CPU box, rank == local_rank).
- **Backends:** **gloo** runs on CPU (and is fine for learning + CPU training); **nccl** is the GPU backend (GPU-direct, much faster) — same API, so everything you learn today transfers.
- **The collectives, in words:** `all_reduce(t, SUM)` — every rank contributes its tensor; afterwards *every* rank holds the elementwise sum (picture: all tensors flow to a meeting point, the sum flows back to everyone). `broadcast(t, src=0)` — rank 0's tensor overwrites everyone else's (one-to-all copy). `all_gather(list, t)` — everyone ends up with the full list of every rank's tensor, in rank order (all-to-all show-and-tell). `reduce_scatter` — like all_reduce, but the summed result is *split*, each rank keeping one shard (this is half of how ring all-reduce works internally).
- **torchrun spawning:** `torchrun --nproc_per_node=4 script.py` launches 4 copies of your script and sets env vars `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`; `init_process_group("gloo")` reads them. Your script *is* the per-rank program — there's no orchestrator file.
- **Collectives are synchronous rendezvous points:** every rank must call the same collective in the same order, or you deadlock. Mismatched calls are *the* classic distributed bug.
**Read (30–45 min):**
- Writing Distributed Applications with PyTorch — "Setup", "Point-to-Point Communication", "Collective Communication": https://docs.pytorch.org/tutorials/intermediate/dist_tuto.html
- torch.distributed API docs — "Backends" table and the collective functions: https://docs.pytorch.org/docs/stable/distributed.html
- torchrun docs — usage and the env vars it sets: https://docs.pytorch.org/docs/stable/elastic/run.html
**Build:**
1. Write `week12/collectives_play.py`:
   ```python
   def main() -> None:
       """Init gloo from torchrun env vars; run collective exercises; asserts
       must pass on every rank. Run: torchrun --nproc_per_node=4 week12/collectives_play.py"""
   ```
   Skeleton:
   ```python
   import torch
   import torch.distributed as dist

   def main() -> None:
       dist.init_process_group("gloo")            # reads RANK/WORLD_SIZE/... from env
       rank, world = dist.get_rank(), dist.get_world_size()
       ...
       dist.destroy_process_group()

   if __name__ == "__main__":
       main()
   ```
2. Exercise A — all_reduce:
   ```python
   t = torch.tensor([float(rank)])
   dist.all_reduce(t, op=dist.ReduceOp.SUM)
   assert t.item() == world * (world - 1) / 2   # sum 0..world-1 — guaranteed
   ```
3. Exercise B — broadcast: rank 0 makes `t = torch.arange(4.)`, others make `t = torch.zeros(4)`; `dist.broadcast(t, src=0)`; all ranks assert `torch.equal(t, torch.arange(4.))`.
4. Exercise C — all_gather:
   ```python
   mine = torch.tensor([rank * 10.0])
   out = [torch.zeros(1) for _ in range(world)]
   dist.all_gather(out, mine)
   assert [x.item() for x in out] == [10.0 * r for r in range(world)]
   ```
5. Print one line per rank (`f"rank {rank}/{world}: all checks passed"`), then `dist.destroy_process_group()`.
6. **Stretch (optional):** ring all-reduce by hand with `dist.send`/`dist.recv` — pass partial sums around the ring `rank → (rank+1) % world` for `world−1` hops, then circulate the total; assert it matches `all_reduce`.
**Verify — done when:** `torchrun --nproc_per_node=4 week12/collectives_play.py` exits 0 with 4 "all checks passed" lines; killing the assert values (e.g. wrong expected sum) makes it fail loudly — try it once to confirm the asserts are live.
**If stuck:** dist_tuto (URL above) has working init boilerplate; deadlock = ranks calling different collectives or different counts — add `print(f"rank {rank} before X", flush=True)` lines to find where ranks diverge; torchrun docs for env-var details.

## Day 12.2 — Data Parallelism From First Principles, Then DDP (~2.5h)
- [ ] done
**Goal:** Prove — by an exact parity test — that manual gradient-averaging data parallelism equals single-process big-batch training, then show DDP passes the same test.
**Learn:**
- **The math.** Keep identical model replicas on every rank. Split a batch of size B into `world` shards of size B/world. Each rank computes mean loss over its shard and backprops. Then average gradients across ranks. Because the gradient of a mean is the mean of per-example gradients, `mean_over_ranks(grad of shard-mean-loss) = grad of full-batch-mean-loss` — *exactly*, for equal shard sizes. Data parallelism isn't an approximation; it's an algebraic identity (up to float associativity).
- **Therefore:** identical init + identical data + averaged grads + identical optimizer ⇒ all ranks' params stay in lockstep forever, and match the single-process big-batch run step-for-step.
- **What breaks exactness in practice:** unequal shard sizes (use `drop_last`), dropout/randomness with different per-rank seeds, BatchNorm (per-shard batch stats). Today's parity test uses a deterministic model precisely to avoid these.
- **DDP = this, made fast.** `DistributedDataParallel` broadcasts params from rank 0 at construction, then *overlaps* gradient all_reduce with the backward pass by bucketing grads and reducing each bucket the moment it's ready — same math, hidden latency.
- **DistributedSampler** partitions dataset indices across ranks so shards don't overlap; `sampler.set_epoch(epoch)` reshuffles differently each epoch — forget it and every epoch sees the same order.
**Read (30–45 min):**
- DDP design notes — "Internal Design" (buckets, the backward hook, Reducer): https://docs.pytorch.org/docs/stable/notes/ddp.html
- Getting Started with DDP tutorial — "Basic Use Case", "Save and Load Checkpoints", torchrun section: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
**Build:**
1. Write `week12/manual_dp.py`. Spec the determinism precisely — this is the heart of the day:
   - Model: small MLP (e.g. `Linear(32,64) → tanh → Linear(64,10)`), **no dropout, no BatchNorm**; init under `torch.manual_seed(0)` on every rank (identical replicas — verify with a param checksum across ranks via all_gather before training).
   - Data: one fixed tensor dataset generated under `torch.manual_seed(1)`, full copy on every rank; total batch B=32, rank r takes rows `[r*B//world : (r+1)*B//world]` (equal shards, fixed order, no shuffling).
   - Loop (N=20 steps): forward shard → mean CE loss → backward → average grads → step:
     ```python
     loss = F.cross_entropy(model(x_shard), y_shard)   # mean over the shard
     loss.backward()
     for p in model.parameters():                      # the whole trick, 3 lines
         dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
         p.grad /= world
     opt.step(); opt.zero_grad(set_to_none=True)
     ```
     Plain SGD, lr=0.1 — Adam works too, but SGD keeps the parity argument transparent.
2. Reference run in the same file: when launched *without* torchrun (`WORLD_SIZE` unset), run single-process on the full B=32 batch, same seeds, same N steps, and save `params_ref.pt`.
3. Parity assert: after N steps, rank 0 compares every param to `params_ref.pt` with `torch.allclose(p, p_ref, atol=1e-6)`. This is THE verification — if it fails, your DP is wrong, not "approximately fine".
4. Write `week12/ddp_version.py`: same model/data, but `model = DDP(model)`, delete the manual all_reduce (DDP does it in backward), keep the loss/optimizer identical. Re-run the same parity assert vs the same `params_ref.pt`.
5. Add a `DistributedSampler` variant over a TensorDataset to see the idiom (`shuffle=True`, `sampler.set_epoch(e)`) — no parity claim here, just exercise the API and print which indices each rank gets for epoch 0 vs 1.
**Verify — done when:** `python week12/manual_dp.py` writes the reference; `torchrun --nproc_per_node=2 week12/manual_dp.py` passes the allclose; `torchrun --nproc_per_node=2 week12/ddp_version.py` passes it too; flipping a shard boundary off-by-one makes it fail (negative control — run once).
**If stuck:** parity failures: check identical init (print param checksums per rank), `p.grad /= world` (sum vs mean!), and that the reference uses mean loss over the *full* batch; DDP notes "Internal Design" if you wonder when DDP syncs.

## Day 12.3 — DDP Your LM Training (~2.5h)
- [ ] done
**Goal:** Make your real LM training script DDP-ready and validate a 2-process CPU smoke run against single-process training.
**Learn:**
- **Rank-0-only side effects:** logging, CSV writing, checkpoint saving, sample printing — all guarded by `if rank == 0:`. Otherwise N ranks write the same file concurrently (corruption) or N× log spam.
- **Barrier around save/load:** rank 0 saves; `dist.barrier()`; then (if other ranks need the file, e.g. resume) everyone loads. Without it, rank 1 may read a half-written checkpoint. Save `model.module.state_dict()` — DDP wraps your model, and you want clean keys for non-DDP loading later.
- **Data sharding for LM memmaps:** instead of DistributedSampler, the simplest correct scheme for your `TokenFileDataset` is rank-strided sampling — each rank draws windows from its own region or strides `rank::world` over shuffled indices. Equal counts per rank per step, no overlap.
- **LR scaling heuristic:** world ranks ⇒ effective batch ×world; the *linear scaling rule* says scale lr ×world (with warmup) to keep dynamics similar. It's a heuristic, not a law — at your scale, keeping per-step effective batch and lr identical to your tuned single-proc config is the safer default. Note which you chose.
- **CPU DDP is a real tool, not just a toy:** gloo all_reduce of your ~5M params per step has cost; on one box, 2 procs may or may not beat 1 proc ×2 batch. Measure it — Week-11 rules apply.
**Read (30–45 min):**
- DDP tutorial — "Save and Load Checkpoints" and the torchrun section: https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- torchrun docs (single-node usage, `--standalone`): https://docs.pytorch.org/docs/stable/elastic/run.html
**Build:**
1. Refactor your training entry point into `week12/train_lm.py` with a `setup_distributed() -> tuple[int, int]` helper:
   ```python
   def setup_distributed() -> tuple[int, int]:
       """If WORLD_SIZE env var set: init_process_group('gloo'), return (rank, world).
       Else return (0, 1). Training code is identical either way."""
   ```
2. Integrate: wrap model in DDP when `world > 1`; rank-strided batch sampling from the memmap; `if rank == 0` around your Trainer's logging/checkpoint callbacks; `dist.barrier()` after saving; divide the global batch so per-step *effective* tokens match the single-proc config. The save/load shape:
   ```python
   if rank == 0:
       torch.save({"model": model.module.state_dict() if world > 1 else model.state_dict(),
                   "optim": opt.state_dict(), "sched": sched.state_dict(),
                   "step": step}, ckpt_path)
   if world > 1:
       dist.barrier()        # unconditional on all ranks — never inside the rank-0 if
   ```
3. Smoke test: 300 steps single-proc at batch 2×B vs `torchrun --standalone --nproc_per_node=2` at per-rank batch B. Same seed for init; data order will differ — so the check is statistical, not exact: plot both loss curves; they should overlap in shape and final value (within normal run-to-run noise).
4. Measure tokens/sec (aggregate across ranks) for both → table row in `LOG.md`; honest verdict on whether CPU DDP helps on your box.
**Verify — done when:** 2-proc run completes; overlay plot saved (`week12/ddp_loss_compare.png`) and curves agree by eye + final losses within ~5%; after the run, `assert len(glob("checkpoints/smoke*")) == expected_count` — checkpoints written once, not twice; single-proc path (`python week12/train_lm.py`) still works unchanged.
**If stuck:** duplicate checkpoints ⇒ a callback not rank-guarded; hang at save ⇒ a barrier some ranks skip (barriers must be unconditional); DDP tutorial checkpoint section has the canonical save/load ordering.

## Day 12.4 — Extending Autograd: Fused Softmax Cross-Entropy (~2.5h)
- [ ] done
**Goal:** Implement cross-entropy as a custom `autograd.Function` with a hand-derived backward, verified by gradcheck and against `F.cross_entropy`.
**Learn:**
- **`autograd.Function` recap at tensor scale:** `forward(ctx, ...)` computes outputs and `ctx.save_for_backward(...)` stashes what backward needs; `backward(ctx, grad_out)` returns one gradient per forward input (`None` for non-differentiable args like integer targets).
- **The derivation — work through it on paper, it's the most famous gradient in deep learning.** For logits `z ∈ R^V`, target class `y`: `L = -log softmax(z)_y = -z_y + logsumexp(z)`. Then `∂L/∂z_j = softmax(z)_j - 1[j = y]`. In matrix form over a batch of N rows with mean reduction: `∂L/∂Z = (softmax(Z) - onehot(Y)) / N`. Intuition: push every logit's probability down toward 0 except the true class, which gets pushed up by `1 - p_y`.
- **Stability:** compute `log_softmax` as `z - z.max(dim=-1, keepdim=True).values - log(sum(exp(shifted)))` — subtracting the max prevents `exp` overflow; never materialize `softmax → log`.
- **Why "fused" matters:** `F.cross_entropy` exists precisely because composing `log_softmax` + `nll_loss` as separate autograd nodes stores intermediates and runs extra kernels; one Function = one saved tensor (the softmax, or the log-softmax) and one fused backward formula.
- **gradcheck** compares your analytical backward to finite differences — in float64, because fp32 finite differences are too noisy to trust.
- **When you'd go further:** custom CUDA/Triton kernels are the next level (fusing across ops autograd can't see, e.g. flash-attention) — know that `triton` exists and that today's Function is the *interface* such kernels plug into. Out of scope here.
**Read (30–45 min):**
- Extending PyTorch note — "Extending torch.autograd" (the custom Function how-to, `save_for_backward`, `needs_input_grad`): https://docs.pytorch.org/docs/stable/notes/extending.html
- `torch.autograd.gradcheck` docs: https://docs.pytorch.org/docs/stable/generated/torch.autograd.gradcheck.html
- Optional: Double Backward with Custom Functions tutorial (skim): https://docs.pytorch.org/tutorials/intermediate/custom_function_double_backward_tutorial.html
**Build:**
1. Create `tlib/fused_ce.py`:
   ```python
   class FusedSoftmaxCrossEntropy(torch.autograd.Function):
       """Mean-reduced cross-entropy from raw logits.
       forward(ctx, logits: Tensor[N, V], targets: Tensor[N] int64) -> scalar Tensor
         Computes stable log-softmax + NLL; saves softmax probs and targets.
       backward(ctx, grad_out) -> (Tensor[N, V], None)
         Returns grad_out * (probs - onehot(targets)) / N for logits; None for targets."""

   def fused_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
       """Convenience wrapper: FusedSoftmaxCrossEntropy.apply(logits, targets)."""
   ```
   Include the derivation as a comment block above `backward` — write it out, don't paste. Forward core:
   ```python
   @staticmethod
   def forward(ctx, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
       z = logits - logits.max(dim=-1, keepdim=True).values          # stability shift
       log_probs = z - z.exp().sum(dim=-1, keepdim=True).log()       # log-softmax
       loss = -log_probs[torch.arange(len(targets)), targets].mean()
       ctx.save_for_backward(log_probs.exp(), targets)               # probs, y
       return loss
   ```
   Backward returns `grad_out * (probs - onehot) / N` for logits and `None` for targets (`onehot` via `probs.scatter_`-style index subtraction: `probs[arange(N), targets] -= 1`, then `/ N` — but clone the saved tensor first; never mutate saved tensors in place).
2. `tests/test_fused_ce.py`:
   - Forward parity: random `(8, 50)` logits → `torch.allclose(fused_cross_entropy(z, y), F.cross_entropy(z, y), atol=1e-6)`.
   - Grad parity: same inputs through both, `.backward()`, compare `z.grad` with allclose.
   - `gradcheck(FusedSoftmaxCrossEntropy.apply, (z64.requires_grad_(), y), eps=1e-6, atol=1e-4)` with float64 logits, small shape like `(4, 7)`.
   - Stability: logits with a row containing `+1e4` — loss must be finite (the naive `softmax().log()` version NaNs/infs here; show that in a comment or a second assert on the naive formula).
3. Drop it into one short LM training run (50 steps) replacing `F.cross_entropy`; loss curve must match the F.cross_entropy run (same seed) step-for-step to ~1e-5.
**Verify — done when:** all four tests pass under `pytest tests/test_fused_ce.py`; the 50-step swap-in run matches; the derivation comment exists and ends with `(softmax - onehot)/N`.
**If stuck:** gradcheck failing with fp32 inputs ⇒ use float64 (it's a documented requirement); off-by-N ⇒ remember mean reduction divides the *gradient* by N too; Extending PyTorch note's worked Linear example mirrors the structure exactly.

## Day 12.5 — Pretraining Prep: Data, Config, Launch Script, Resume Proof (~2.5h)
- [ ] done
**Goal:** Choose and pretokenize the corpus, lock the model config and token budget, and prove resumability before committing hours to the run.
**Learn:**
- **Dataset choice.** **TinyStories** (`roneneldan/TinyStories`, ~2.1M synthetic short stories, small vocabulary) is ideal for small models — coherent English emerges at single-digit-millions of params. **wikitext-103** is real Wikipedia text — harder, needs more model. CPU path: TinyStories subset. GPU path: full TinyStories or wikitext-103.
- **Tokenizer sizing:** small vocab (4096–8192) suits a small corpus/model — fewer embedding params, more tokens per text (a tradeoff you can now articulate). Train your own BPE on a corpus *sample* (~50–100MB of text is plenty for merges).
- **Token budget math:** `total_tokens = tokens_per_sec × seconds_available`. Chinchilla's compute-optimal heuristic is ~20 tokens per parameter — at 5M params that's 100M tokens, which an overnight CPU run may or may not reach. Use it as *context*, not a promise: undertrained-but-real beats imaginary-optimal.
- **Resumability is non-negotiable** for an hours-long run: checkpoint must capture model, optimizer, scheduler, step count, and dataloader position (for memmap sampling: the RNG state or the step-derived offset). Prove resume works on a 5-minute run *before* the long one — discovering a resume bug at hour 6 is the avoidable disaster.
**Read (30–45 min):**
- HF datasets quickstart ("load a dataset" section): https://huggingface.co/docs/datasets/quickstart
- TinyStories dataset card: https://huggingface.co/datasets/roneneldan/TinyStories
- Your own Week-11 speed-run table — it dictates today's config defaults.
**Build:**
1. `uv add datasets`. Write `week12/prepare_data.py`:
   ```python
   def download_and_tokenize(dataset: str = "roneneldan/TinyStories",
                             vocab_size: int = 4096,
                             out_dir: Path = Path("data/pretrain"),
                             max_stories: int | None = None) -> None:
       """Load via datasets.load_dataset(dataset); train BPETokenizer on a
       sample; encode train/val splits to uint16 memmap shards
       (train.bin, val.bin); save tokenizer to out_dir/tokenizer.json.
       Prints progress every 10_000 docs and final token counts."""
   ```
   Load code:
   ```python
   from datasets import load_dataset
   ds = load_dataset("roneneldan/TinyStories")           # splits: train, validation
   texts = ds["train"]["text"]                            # list[str], one story each
   # wikitext variant: load_dataset("wikitext", "wikitext-103-raw-v1")
   ```
   Join stories with your EOS/EOT token between them (so the model learns document boundaries). CPU path: `max_stories≈200_000` keeps prep fast; print the resulting token count.
2. Pick a config (write both in `LOG.md`, choose one):
   ```
   CPU  (~5–8M params):  d_model=256, n_layers=6,  n_heads=8, block_size=256, vocab=4096
   GPU  (~30–60M params): d_model=512, n_layers=8–12, n_heads=8, block_size=512, vocab=8192
   ```
   Compute exact param count (`sum(p.numel())`), then the budget: measured tokens/sec (from Week 11 — re-measure with THIS config, not the Week-10 one!) × planned seconds → total tokens, and tokens/param ratio. Write all three numbers down before launching anything.
3. Write `week12/pretrain.py` — the launch script: TokenFileDataset on `train.bin`, all adopted Week-11 optimizations (memmap, set_to_none, bf16-if-it-helped, compile-if-it-helped), AdamW + warmup+cosine over the planned step count, grad clipping, val CE on `val.bin` every N steps, checkpoint every M steps to `checkpoints/pretrain_step{K}.pt` keeping the last 3, and a `--resume path.pt` flag. Checkpoint contents — all five or resume is broken:
   ```python
   {"model": ..., "optim": ..., "sched": ..., "step": step,
    "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}}
   ```
4. **Resume proof:** launch, kill (Ctrl-C) around step ~400, `--resume` from the last checkpoint, run to step 800. Plot the loss CSV: assert programmatically that the first post-resume loss is within, say, 3× the std of the 50 pre-kill losses (no cliff/spike), and that step numbering continues without gap or overlap.
**Verify — done when:** `train.bin`/`val.bin` exist with printed token counts; tokenizer round-trips a sample story; param count + token budget + tokens/param are in `LOG.md`; the kill-and-resume test passes its continuity assert. Do not start Day 6 until it does.
**If stuck:** datasets quickstart for load/split idioms; resume discontinuity ⇒ usually optimizer or scheduler state not restored (loss spikes when Adam moments reset) — diff the state_dict keys you save vs load.

## Day 12.6 — Deep Build: THE RUN (~3–4h active + hours of training; may span into Day 7)
- [ ] done
**Goal:** Launch the pretraining run, babysit it with periodic validation and sample generations, and land `checkpoints/base_final.pt` — the base model for Weeks 13–14.
**Learn:**
- **Babysitting is monitoring, not staring:** check val CE trend, watch for NaN/spike, read the samples. The samples are the payoff — at step ~0 the model emits token soup at loss ≈ ln(V); over hours it discovers words, then grammar, then short coherent story fragments. Save every sample; the evolution file is the artifact you'll actually show people.
- **What a healthy run looks like:** train CE falls fast then slows (roughly log-linear in steps); val CE tracks slightly above train; the cosine schedule's tail gives a late small dip. A flat curve from step 0 = bug (lr, data, or labels), not patience.
- **Generation cadence:** every N steps, run your `tlib/generate.py` sampler (temperature ~0.8, top-k 50) on 2–3 fixed prompts ("Once upon a time", empty/EOT) — fixed prompts make evolution comparable.
**Read (10 min):** Your Day-12.5 launch checklist. Nothing new today — execution day.
**GPU sidebar — renting for the run (Lambda / RunPod), the practical 10 lines:**
1. Pick a single mid-tier GPU (A10/A100-40GB class is overkill-fine; even an RTX 4090 instance works at this scale); check $/hr and pick on-demand, not spot, for a first run.
2. Launch with a PyTorch 2.x image; `ssh` in.
3. `rsync -av --exclude data/ your-repo/ ubuntu@IP:~/torch/` (or `git clone` + `scp data/pretrain/*.bin` — the .bin shards are the big part).
4. `uv venv && uv pip install torch numpy datasets` (match your local versions).
5. Re-run the resume proof ONCE on the instance (5 min) — new hardware, same rule.
6. Start inside `tmux` (`tmux new -s run`) so SSH drops don't kill training; detach with `Ctrl-b d`.
7. Watch `nvidia-smi -l 5` in a second pane — you want sustained high GPU-util; low util ⇒ input-bound, raise batch or check dataloading.
8. Use the GPU config + bf16 autocast + compile; re-verify tokens/sec and recompute your token budget before walking away.
9. Periodically `rsync` checkpoints + `samples.txt` *down* to your machine — the instance is disposable, your checkpoints aren't.
10. After `base_final.pt` is downloaded and verified loadable locally: terminate the instance. Billing stops at termination, not shutdown.
**Build:**
1. Launch: `python week12/pretrain.py --config week12/config_cpu.json 2>&1 | tee run.log` (CPU overnight) or the GPU variant per the sidebar. Record launch time, config hash, and planned step count in `LOG.md`.
2. The script should append generation blocks to `samples.txt` automatically, in this format so evolution is greppable:
   ```
   --- step 2000 | train_ce 4.81 | val_ce 4.95 ---
   prompt: "Once upon a time"
   Once upon a time ther was a littel dog nameed...
   ```
3. Babysit loop (every ~30–60 min while awake): note step, train CE, val CE; skim the newest `samples.txt` blocks; confirm checkpoints are rotating (`ls -lt checkpoints/ | head`). A quick health check you can run any time without touching the training process:
   ```bash
   tail -3 run.log && tail -20 samples.txt
   ```
3. On completion: copy the last checkpoint to `checkpoints/base_final.pt`; save the tokenizer alongside it; plot train+val CE → `week12/pretrain_loss.png`.
4. Post-run sanity: load `base_final.pt` fresh in a new process, generate 5 samples, compute val CE over 50 batches — matches the logged final val CE.
**Verify — done when:**
- Final val CE is meaningfully below the ln(V) starting point (ln 4096 ≈ 8.32; *any* real training passes this — it's a sanity bound, not a target) AND below your Week-10 Gutenberg model's CE is *not* required (different data — don't compare across corpora).
- `samples.txt` shows visible evolution: pick three blocks (early/middle/final) and paste them into `LOG.md` with one line of commentary each.
- `checkpoints/base_final.pt` + tokenizer load in a fresh process and generate. **This checkpoint is the base model for Week 13 (classification head, instruction SFT, LoRA) and Week 14 (DPO + eval harness). Guard it. Back it up.**
**If stuck:** loss flat at ln(V) ⇒ check labels are the shifted-by-one targets and lr isn't 0 during warmup misconfig; NaN ⇒ halve lr, confirm grad clipping active, check for bf16 issues by falling back to fp32 for 100 steps; run dies overnight ⇒ that's what the resume proof was for — `--resume` and continue.

## Day 12.7 — Review, Quiz, Redo-Cold & Postmortem (~2h, plus any run babysitting spillover)
- [ ] done
**Goal:** Consolidate distributed + extension knowledge and write an honest postmortem of the run.
**Learn (review):** Re-read the DDP notes "Internal Design" section once more *after* having used DDP — it reads differently now. Re-derive the cross-entropy gradient on paper without looking.
**Postmortem (half page, in `LOG.md`):** What was the final config, total tokens, tokens/param, wall-clock, and (if rented) cost? What surprised you (tokens/sec drift? sample quality timeline?)? What would you change for a rerun: data size vs model size? Which Week-11 optimization mattered most in practice? One paragraph each: "went well / went badly / do differently".
**Self-quiz (write answers, then check the Answers section below):**
1. Define rank, local_rank, world_size. Which one selects the GPU on a multi-GPU node?
2. Describe what each rank holds after: all_reduce(SUM); broadcast(src=0); all_gather; reduce_scatter.
3. Why is averaged-gradient data parallelism *exactly* equivalent to big-batch training (state the identity), and name two things that break the exactness.
4. What does DDP do at construction time, and what does it overlap with backward?
5. Why must `DistributedSampler.set_epoch(epoch)` be called, and what silently goes wrong without it?
6. Why is checkpoint saving rank-0-only, and why does a barrier belong next to it?
7. Derive ∂L/∂z for L = −log softmax(z)_y. What's the batched mean-reduction form?
8. Why does gradcheck require float64?
9. Why does the stable log-softmax subtract the row max, and what fails without it?
10. Your run starts at loss ≈ 8.3 with vocab 4096. Why that number, and what does a flat curve there indicate?
11. Why prove resumability with a kill-test *before* the long run, and which four pieces of state must a resume restore?
12. You're on a rented GPU and nvidia-smi shows 35% utilization. What's your first hypothesis and first measurement (Week-11 vocabulary)?
**Redo-cold drills (no notes):**
- Write the torchrun-launched gloo init + all_reduce-of-ranks assert from a blank file; run on 4 procs.
- Re-derive (softmax − onehot)/N on paper, including the logsumexp step.
- Write the manual gradient-averaging loop (the 3 lines per param after backward) and state the parity-test preconditions from memory.
- From a blank file: load TinyStories via `datasets`, print the first story and the train split size.
- State your final run's numbers from memory: params, tokens, tokens/param, final val CE.

**Looking ahead:** Week 13 starts from `checkpoints/base_final.pt` — you'll bolt a classification head onto it, run instruction SFT with loss masking, and implement LoRA. Week 14 adds DPO, an eval harness, and the capstone. Nothing more to do now; just make sure the checkpoint and tokenizer are backed up somewhere that isn't this machine.

---

## Answers (Day 12.7)
1. rank: global process index 0..N−1 across all nodes; local_rank: index within one node; world_size: total process count N. **local_rank** picks the GPU (`torch.device(f"cuda:{local_rank}")`) — global rank would be wrong on node 2+.
2. all_reduce(SUM): every rank holds the elementwise sum of all contributions. broadcast(src=0): every rank holds a copy of rank 0's tensor. all_gather: every rank holds the ordered list of all ranks' tensors. reduce_scatter: the sum is computed, then split — each rank holds only its shard of the reduced result.
3. Gradient of a mean is the mean of gradients: with equal shards, mean over ranks of ∇(shard-mean-loss) = ∇(full-batch-mean-loss), so updates are identical (up to float reassociation). Broken by: unequal shard sizes, per-rank randomness (dropout with different seeds), batch-statistics layers (BatchNorm computing per-shard stats).
4. At construction DDP broadcasts rank 0's params (and buffers) so all replicas start identical. During backward it buckets gradients and launches all_reduce per bucket as soon as that bucket's grads are ready — overlapping communication with the rest of backward's computation.
5. The sampler's shuffle is seeded by the epoch; without set_epoch every epoch uses the same permutation, so each rank sees the identical data order each epoch — quietly worse training, no error raised.
6. N ranks writing one file concurrently corrupts it (or wastes N× I/O); replicas are identical so one copy suffices. The barrier ensures no rank proceeds (e.g. to load, or to a collective the saving rank hasn't reached) while rank 0 is mid-write — preventing torn reads and deadlocks.
7. L = −z_y + logsumexp(z); ∂logsumexp/∂z_j = softmax(z)_j; ∂(−z_y)/∂z_j = −1[j=y]; so ∂L/∂z_j = softmax(z)_j − 1[j=y]. Batched with mean reduction over N rows: ∂L/∂Z = (softmax(Z) − onehot(Y))/N.
8. gradcheck compares analytic grads to finite differences (f(x+ε)−f(x−ε))/2ε; in fp32 the subtraction of nearly-equal values loses most significant bits, making the numerical reference itself too noisy at usable ε. float64 makes finite differences trustworthy.
9. exp(z) overflows to inf for z ≳ 88 in fp32; since softmax is shift-invariant (softmax(z) = softmax(z − c)), subtracting the row max makes the largest exponent argument 0, so all exp values are in (0, 1]. Without it, large logits give inf/inf → NaN.
10. A uniform distribution over V classes has CE = ln(V) = ln 4096 ≈ 8.32 — the loss of a model that knows nothing. Staying flat there means no learning is happening: lr is 0/wrong, labels aren't aligned with inputs (shift bug), or grads aren't reaching the params — a bug, never patience.
11. Because a resume bug discovered at hour 6 forfeits the run; the kill-test costs 5 minutes. Must restore: model state_dict, optimizer state (Adam moments!), scheduler state/step count, and RNG/data-position state. (Missing optimizer state shows up as a loss spike right after resume.)
12. Hypothesis: input-bound — the GPU is waiting on the data pipeline (or sync points), not compute-limited. First measurement: the Day-11.2 constant-batch test on the instance — time steps with a preloaded batch vs real loading; the gap is the pipeline cost. Then fix per Week 11 (memmap reads, pin_memory + non_blocking, no per-step .item()).
