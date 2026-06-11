# Week 7 — Convolutions & ResNet

This week you move from generic modules to the architecture family that made deep learning work on images. You already built conv1d via `unfold` in Week 2 and BatchNorm in Week 6; now you combine them: conv2d from first principles, output-shape and receptive-field arithmetic, a real CIFAR-10 training run on your own Trainer, and the single most important architectural idea of the 2010s — the residual connection. By the end you will have run your first genuine deep-learning experiment loop: baseline → ablation → tuned best run, with results recorded in a CSV.

**Week outcome:** `week07/conv_scratch.py` (conv2d + pooling verified vs torch), `tlib/resnet.py` (BasicBlock + configurable ResNet), `tlib/transforms.py` (your own augmentations), and `week07/experiments.csv` + `week07/LOG.md` documenting a CIFAR-10 best run that beats your own Day-3 baseline by ≥5 accuracy points. Skill: you can design a convnet's shapes and receptive field on paper, and explain why residual nets train where plain nets stall.

All CIFAR budgets below are CPU-feasible by design (10k-image subset, small nets); each training day carries a "GPU variant" note if you have CUDA — same code, bigger budget.

## Day 7.1 — Conv2d from first principles (~2.5h)
- [ ] done
**Goal:** Implement 2-D convolution yourself via unfold + matmul and match `F.conv2d` exactly.
**Learn:**
- *Cross-correlation vs convolution:* what deep learning calls "convolution" is cross-correlation — the kernel is not flipped. It doesn't matter because the kernel is learned; a flipped optimum is also an optimum.
- *Channels and feature maps:* a conv layer maps `(C_in, H, W)` → `(C_out, H', W')`. Each of the `C_out` output maps is produced by its own `(C_in, k, k)` kernel that sees *all* input channels and sums over them.
- *Output-shape arithmetic:* `out = floor((n + 2p − k) / s) + 1` per spatial dim. Internalize this; you will use it every day this week.
- *Weight sharing:* the same `k×k` kernel slides over every position. A `3×3, 64→64` conv has `64·64·9 + 64 ≈ 37k` params regardless of image size; a Linear on flattened `64×32×32` activations would need ~4 billion. Sharing encodes translation equivariance as an inductive bias.
- *Conv as matmul:* `F.unfold` extracts all `(C_in·k·k)` patches into columns; convolution is then a single matmul with the flattened weight — exactly your Week 2 conv1d trick, one dimension up.
**Read (30–45 min):**
- d2l.ai ch. 7, sections "Convolutions for Images", "Padding and Stride", "Multiple Input and Multiple Output Channels": https://d2l.ai/chapter_convolutional-neural-networks/index.html
- `torch.nn.Conv2d` docs (shape section, formula): https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
- `torch.nn.Unfold` docs (read the shape math carefully): https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html
**Build:** in `week07/conv_scratch.py` + `week07/test_conv_scratch.py`:
1. The shape helper:
   ```python
   def out_shape(n: int, k: int, s: int = 1, p: int = 0) -> int:
       """Output size of one spatial dim: floor((n + 2p - k)/s) + 1."""
   ```
2. The convolution:
   ```python
   def conv2d_unfold(x: Tensor, weight: Tensor, bias: Tensor | None = None,
                     stride: int = 1, padding: int = 0) -> Tensor:
       """x: (B, C_in, H, W); weight: (C_out, C_in, k, k). Returns (B, C_out, H', W').

       Implementation: F.unfold(x, k, stride=stride, padding=padding) -> (B, C_in*k*k, L);
       matmul with weight.view(C_out, -1); reshape L back to (H', W') via out_shape.
       Add bias as (1, C_out, 1, 1). No F.fold needed — a plain .view suffices.
       """
   ```
   Example: `x = torch.randn(2, 3, 8, 8)`, `weight = torch.randn(5, 3, 3, 3)`, `padding=1` → output shape `(2, 5, 8, 8)`.
   Worked shape trace for that example — write this as a comment above the function and check each line in a REPL:
   ```text
   F.unfold(x, 3, padding=1)            -> (2, 3*3*3, 64) = (B, C_in*k*k, L), L = 8*8
   weight.view(5, -1)                   -> (5, 27)
   (B, 27, 64) matmul'd with (5, 27)    -> (2, 5, 64)      # einsum or w @ cols
   .view(2, 5, 8, 8)                    -> done; + bias.view(1, 5, 1, 1)
   ```
3. Pytest `test_out_shape_table` — hand-computed cases asserted exactly:
   `(32,3,1,1)→32`, `(32,3,2,1)→16`, `(32,5,1,0)→28`, `(7,3,2,0)→3`, `(28,7,2,3)→14`.
4. Pytest `test_conv2d_vs_torch`: loop ≥20 randomized configs (`B∈{1,2,4}`, `C_in∈{1,3,8}`, `C_out∈{1,4,16}`, `H,W∈{5..16}`, `k∈{1,3,5}`, `s∈{1,2}`, `p∈{0,1,2}`, bias on/off, fixed seed) asserting `torch.allclose(conv2d_unfold(...), F.conv2d(...), atol=1e-5)`.
5. `test_param_count`: assert a `3×3, 64→64` conv (weight + bias) totals exactly `36928` params, and add a comment computing the Linear-equivalent count for a `64×32×32` input/output.
**Verify — done when:** `pytest week07/test_conv_scratch.py` passes; all 20+ randomized configs match `F.conv2d` to 1e-5.
**If stuck:** the `nn.Unfold` doc's output-shape formula; your own Week 2 conv1d-via-unfold solution (same structure, one fewer dim); the `F.conv2d` docs for argument semantics.

## Day 7.2 — Pooling, receptive fields & shape design (~2h)
- [ ] done
**Goal:** Implement MaxPool2d via unfold and build a receptive-field calculator you trust.
**Learn:**
- *Pooling's job:* summarize a neighborhood (max = "is the feature anywhere here?", avg = "how much on average?") and downsample. Modern nets often replace pooling with stride-2 convs — a learned downsample; ResNet uses both.
- *Receptive field:* the patch of input pixels that can influence one output unit. Stacked 3×3 convs grow it linearly; striding multiplies the growth rate of everything after it.
- *The recurrence:* with `jump_0 = 1, rf_0 = 1`, for layer `l` with kernel `k_l` and stride `s_l`: `rf_l = rf_{l-1} + (k_l − 1) · jump_{l-1}` and `jump_l = jump_{l-1} · s_l`. The jump is `prod(strides of earlier layers)` — the distance in input pixels between adjacent units of layer l's input.
- *Dilation:* spreads the kernel taps apart — effective kernel size `d·(k−1)+1` for receptive-field purposes, with no extra parameters.
- *Global average pooling (GAP):* mean over all spatial positions → `(B, C)`. Replaces giant flatten+Linear heads, makes the net input-size-agnostic, and is how your CIFAR nets will end.
**Read (30–45 min):**
- d2l.ai "Pooling": https://d2l.ai/chapter_convolutional-neural-networks/pooling.html
- `torch.nn.MaxPool2d` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
- Understanding Deep Learning, Chapter 10 "Convolutional networks" (receptive-field discussion): https://udlbook.github.io/udlbook/
**Build:** in `week07/conv_scratch.py` (+ tests):
1. Pooling via the same trick as conv:
   ```python
   def maxpool2d_unfold(x: Tensor, k: int, stride: int | None = None) -> Tensor:
       """stride defaults to k. F.unfold -> (B, C*k*k, L) -> view (B, C, k*k, L)
       -> amax over dim 2 -> view back to (B, C, H', W')."""
   ```
   Verify vs `nn.MaxPool2d(k, stride)` over ≥10 random configs with `torch.equal` (max of the same numbers is exact, not approximate).
2. The calculator:
   ```python
   def receptive_field(layers: list[tuple[int, int]]) -> int:
       """layers = [(k, s), ...] in forward order. Returns the receptive field of one
       unit in the final layer's output, using the rf/jump recurrence."""
   ```
   Asserted cases: `[(3,1)] → 3`; `[(3,1),(3,1)] → 5`; `[(3,2),(3,1)] → 7`; `[(7,2),(3,2),(3,1)] → 19`; and `[(3,1)]*n → 2n+1` for n in 1..5.
3. Paper exercise (write it as comments + asserts in the file): design a stack of 3×3 convs (some stride 2) whose rf ≥ 32 on a 32×32 input, using ≤8 layers. Assert it with `receptive_field`, and assert via `out_shape` that no layer's spatial size hits 0.
**Verify — done when:** pooling matches `nn.MaxPool2d` exactly; all rf asserts pass, including your own ≥32 design.
**If stuck:** trace `rf`/`jump` by hand for two layers on paper before debugging code; d2l pooling section; the Conv2d docs' dilation note for the effective-kernel formula.

## Day 7.3 — First CIFAR-10 convnet (~2.5h)
- [ ] done
**Goal:** Train a plain convnet on a fixed CIFAR-10 budget with your own Trainer and record a baseline.
**Learn:**
- *CIFAR-10:* 50k train / 10k test, 32×32 RGB, 10 classes. Small enough for CPU experiments, hard enough that architecture choices show up in the numbers.
- *Normalization with measured stats:* compute per-channel mean/std over your training subset yourself and normalize with them — never with stats copied off the internet, and never computed on test data (leakage).
- *Conv–BN–ReLU:* the canonical block. BN directly after conv (so use `bias=False` on the conv — BN's β makes the bias redundant), ReLU after BN.
- *Fixed-budget discipline:* every run this week uses the same data subset, epochs, and batch size unless the day says otherwise. Only then do comparisons between days mean anything.
**Read (30–45 min):**
- torchvision CIFAR10 docs: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html
- d2l.ai "Networks Using Blocks (VGG)" for the stack-of-blocks design idea: https://d2l.ai/chapter_convolutional-modern/vgg.html
- Skim your own `tlib/trainer.py` and `tlib/norms.py` — today they meet real data.
**Build:** `week07/cifar_baseline.py`:
1. Data:
   ```python
   def get_cifar(root: str = "data", n_train: int = 10_000) -> tuple[Dataset, Dataset]:
       """Download via torchvision. Train = first n_train images (fixed torch.arange
       subset, NOT random — reproducibility). Test = full 10k. Float tensors in [0,1]."""

   def channel_stats(ds) -> tuple[Tensor, Tensor]:
       """Per-channel mean and std over the train subset, each shape (3,)."""
   ```
   Normalize both splits with the *train* stats.
2. The model — exact spec, all convs 3×3 pad 1 `bias=False`, each followed by your BatchNorm + ReLU:
   ```python
   class PlainNet(nn.Module):
       """conv 3->32
          conv 32->64  (stride 2)   # 32x32 -> 16x16
          conv 64->64
          conv 64->128 (stride 2)   # 16x16 -> 8x8
          conv 128->128
          global average pool -> Linear(128, 10)"""
   ```
   Before coding, fill in this table by hand with `out_shape` and the conv parameter formula, then assert the total in code — `sum(p.numel() for p in model.parameters())` must land between 250_000 and 300_000 (it is ≈ 280k):
   ```text
   layer            out shape      conv params (C_out*C_in*9, no bias)
   conv 3->32       (32, 32, 32)   864
   conv 32->64 s2   (64, 16, 16)   18_432
   conv 64->64      (64, 16, 16)   36_864
   conv 64->128 s2  (128, 8, 8)    73_728
   conv 128->128    (128, 8, 8)    147_456
   GAP + fc         (10,)          128*10 + 10
   (+ your BatchNorm's 2*C per conv)
   ```
3. The week's standard budget — write these as module-level constants and reuse them all week:
   ```python
   N_TRAIN, EPOCHS, BATCH = 10_000, 8, 128
   ```
   Your AdamW (`lr=3e-3, weight_decay=5e-4`), your warmup+cosine schedule (5% warmup steps). Use `nn.Conv2d` (your scratch conv proved understanding; production uses torch's) but YOUR `tlib` norms, Trainer, schedules, optimizer, and losses.
4. Evaluate test accuracy each epoch. Append a row to `week07/experiments.csv` with header `run,arch,augment,epochs,final_train_loss,test_acc`.
5. GPU variant: if CUDA is available, use the full 50k train set and 20 epochs; everything else identical.
**Verify — done when:** the run finishes (≈10–20 min CPU); test accuracy is approximately 0.55–0.65 on this budget; `assert test_acc >= 0.55` passes; the CSV row exists. Record your exact number — Days 4–6 are measured against it.
**If stuck:** sanity-check on 256 images first (the net should overfit them to ~100% train accuracy — if not, the bug is in your pipeline, not the budget); your Week 5 LR finder; confirm normalization is applied to both splits using train stats.

## Day 7.4 — Residual connections (~2.5h)
- [ ] done
**Goal:** Build `tlib/resnet.py` and reproduce the degradation result: a 14-layer plain net trains worse than a 14-layer residual net.
**Learn:**
- *The degradation problem:* deeper plain nets get *higher training error* than shallower ones (He et al. §1, Fig. 1). Not overfitting — the optimizer fails. A deeper net could trivially represent the shallow one (extra layers = identity), but SGD can't find that solution in plain nets.
- *Identity shortcut:* `y = F(x) + x`. The block now learns a *residual*, and "do nothing" is the zero function — easy to learn. Gradients flow through the `+ x` path unattenuated: a highway through depth.
- *BasicBlock anatomy:* conv3×3–BN–ReLU–conv3×3–BN, add the skip, *then* ReLU. The second BN's output is added to the raw block input, not an activated one.
- *Projection shortcuts:* when the block changes channels or stride, the skip can't be identity; a 1×1 conv (with matching stride) + BN projects `x` to the right shape. Used only where shapes change.
**Read (30–45 min):**
- ResNet paper, §1, §3.1–3.3: https://arxiv.org/abs/1512.03385
- Understanding Deep Learning, Chapter 11 "Residual networks": https://udlbook.github.io/udlbook/
**Build:** `tlib/resnet.py` + `tlib/test_resnet.py`:
1. The block:
   ```python
   class BasicBlock(nn.Module):
       def __init__(self, c_in: int, c_out: int, stride: int = 1, residual: bool = True):
           """Main path: conv3x3(c_in, c_out, stride)-BN-ReLU-conv3x3(c_out, c_out)-BN.
           Skip: identity if stride == 1 and c_in == c_out, else conv1x1(stride) + BN.
           forward: relu(main + skip) if residual else relu(main)."""
   ```
   The `residual=False` flag gives you a plain control net with *identical* layer count and params (minus projections).
2. The builder:
   ```python
   def make_resnet(stage_cfg: list[tuple[int, int]], num_classes: int = 10,
                   residual: bool = True) -> nn.Module:
       """stage_cfg = [(channels, n_blocks), ...].
       Stem: conv3x3(3, stage_cfg[0][0], stride 1)-BN-ReLU.
       First block of every stage after the first uses stride 2.
       Head: global average pool + Linear(last_channels, num_classes).
       make_resnet([(16,2),(32,2),(64,2)]) has 14 weighted layers
       (1 stem + 6 blocks * 2 convs + 1 fc) — call it ResNet-14."""
   ```
   The ResNet-14 layout, spelled out (spatial sizes for a 32×32 input):
   ```text
   stem    conv3x3  3->16            32x32
   stage1  BasicBlock(16,16) x2      32x32   identity skips
   stage2  BasicBlock(16,32,s2), BasicBlock(32,32)   16x16   first skip = 1x1 proj
   stage3  BasicBlock(32,64,s2), BasicBlock(64,64)    8x8    first skip = 1x1 proj
   head    GAP -> Linear(64, 10)
   ```
3. Tests: `make_resnet([(16,2),(32,2),(64,2)])` on input `(2,3,32,32)` returns `(2,10)`; count conv+linear layers programmatically (skip the 1×1 projections, as the paper does) and assert 14; assert `residual=True/False` models have equal param counts when projections are excluded.
4. The experiment, `week07/plain_vs_residual.py`: train ResNet-14 with `residual=True` and `residual=False`, same Day-3 budget, same seed, identical everything else. Save per-step train losses; plot both curves with matplotlib to `week07/plain_vs_residual.png`; append both rows to `experiments.csv`.
   GPU variant: also run the 20-layer pair `[(16,3),(32,3),(64,3)]` and put all four curves on one plot — the residual advantage growing with depth is the paper's Fig. 1 story reproduced by you.
**Verify — done when:** tests pass; the plot shows the residual curve below the plain curve for most of training (approximately — early steps may interleave); `assert final_train_loss_residual < final_train_loss_plain` holds. At 14 layers the gap is modest; if you have patience or a GPU, repeat with `[(16,3),(32,3),(64,3)]` (20 layers) — the gap widens with depth.
**If stuck:** paper §3.3 and the CIFAR experiment description in §4.2 for exact block/stage layout; check ReLU placement (after the add — activating before adding is the classic bug); confirm the projection path also has BN.

## Day 7.5 — Augmentation & schedule tuning (~2h)
- [ ] done
**Goal:** Write your own tensor-space augmentations and measure what they do to the train/test gap.
**Learn:**
- *Augmentation = data-space regularization:* label-preserving transforms make the model see "more" data; it can't memorize pixel arrangements that keep changing. Helps most when data is scarce relative to model capacity — exactly your 10k-subset regime.
- *The CIFAR pair:* pad 4 then random 32×32 crop (translation invariance), random horizontal flip (mirror invariance). Vertical flips would change semantics — augmentations must respect the label distribution.
- *Train-only:* augment training data only. Augmenting evaluation is a different technique (test-time augmentation); mixing it in silently corrupts comparisons.
- *Schedules and smoothing recap:* warmup+cosine (Week 6) and label smoothing (Week 4 losses) are the standard CIFAR companions to augmentation — and augmented runs often want a longer budget, since each epoch is "harder".
**Read (30–45 min):**
- "Bag of Tricks for Image Classification with CNNs", sections on augmentation, cosine decay, label smoothing: https://arxiv.org/abs/1812.01187
- `torchvision.transforms.v2` docs for reference semantics (you implement your own): https://docs.pytorch.org/vision/stable/transforms.html
**Build:** `tlib/transforms.py` + tests:
1. ```python
   class RandomCrop:
       def __init__(self, size: int, padding: int = 4): ...
       def __call__(self, x: Tensor) -> Tensor:
           """x: (C, H, W). F.pad with zeros by `padding` on each spatial side, then
           crop a random size x size window using two torch.randint offsets."""

   class RandomHorizontalFlip:
       def __init__(self, p: float = 0.5): ...
       def __call__(self, x: Tensor) -> Tensor:
           """Flip dim -1 with probability p."""
   ```
2. Tests: output shape equals input shape; `RandomCrop(32, padding=0)` is the identity; `RandomHorizontalFlip(p=1.0)` applied twice is the identity; every pixel value of a cropped output appears in the padded input (spot-check using a `torch.arange` image where every pixel is unique).
3. `week07/augment_ablation.py`: ResNet-14, Day-3 budget but `EPOCHS=10` for both arms, augmented vs not, identical otherwise. Record final train loss, train accuracy, and test accuracy for both arms to `experiments.csv`.
**Verify — done when:** transform tests pass; in the CSV, the augmented run shows a smaller `(train_acc − test_acc)` gap and a higher train loss than the un-augmented run (approximately — the test-accuracy winner can go either way at this tiny budget; the shrinking gap is the robust signal).
**If stuck:** `plt.imshow` a 4×4 grid of augmented copies of one image — augmentation bugs are visually obvious; check you augment per-sample (inside the Dataset/collate path), not once per batch.

## Day 7.6 — Deep build: the budgeted best run (~3.5–4h)
- [ ] done
**Goal:** Combine everything into one tuned run that beats your Day-3 baseline by ≥5 accuracy points under an explicit compute budget.
**Learn:**
- *Experiment discipline:* one variable per run, every run in the CSV, conclusions only from recorded numbers. This is the day's real lesson; the accuracy is a by-product.
- *Grid search, tiny and honest:* with a small budget you can afford ~4 short probe runs; pick the winner, then spend the full budget exactly once. No re-rolling the final run until it passes.
- *Where the points come from:* on this setup, expect (approximately) augmentation and the longer schedule to matter most, optimizer choice and label smoothing less — but verify against your own CSV, not this sentence.
**Read (15 min):** re-skim Bag of Tricks (https://arxiv.org/abs/1812.01187) for the pieces you'll use; your own Week 5 LR-finder notes.
**Build:** `week07/best_run.py`:
1. The budget — hard caps, written as constants at the top of the file:
   - Probes: 4 runs × 3 epochs each, on the 10k subset.
   - Final: ONE run, ≤25 epochs on the 10k subset (≈≤30 min CPU).
   - GPU variant: probes on 10k, final on the full 50k, ≤30 epochs.
2. Fixed for all runs: ResNet-14 `[(16,2),(32,2),(64,2)]` (or up to `[(32,2),(64,2),(128,2)]` if your Day-4 CPU timings allow), your RandomCrop+Flip, warmup+cosine.
3. Probe grid — pick the winner by 3-epoch test accuracy:
   - optimizer ∈ { your AdamW(lr=3e-3, wd=5e-4), your SGD(momentum=0.9, lr=0.1, wd=5e-4) }
   - label smoothing ∈ { 0.0, 0.1 }
4. Final run with the winning combo; checkpoint best-by-test-acc via your Trainer callbacks; append every probe and the final to `experiments.csv`.
5. `week07/LOG.md` — about 10 lines: budget, probe results table, what you picked and why, final number vs Day-3 baseline, and the one change that mattered most *according to your CSV*.
**Verify — done when:** `assert best_acc >= day3_baseline_acc + 0.05` — self-relative: read the baseline back from your own CSV, never a hard-coded absolute. Approximate expectation: the longer-schedule augmented ResNet lands well above the 8-epoch plain baseline. If you fall short, the first suspects are LR (re-run the finder for the winning optimizer) and a missing/wrong normalization.
**If stuck:** compare per-epoch test-acc curves of final vs baseline — diverging early points at LR/schedule, diverging late points at regularization; your Week 6 diagnostics hooks for dead ReLUs / BN running stats.

## Day 7.7 — Review, quiz & reading torchvision (~2h)
- [ ] done
**Goal:** Consolidate the week; read production ResNet code and diff it against yours.
**Learn:**
- *Reading production source:* torchvision's `resnet.py` is short and canonical — reading it *after* building your own is the highest-yield code reading you can do this week.
- *Self-testing beats re-reading:* do the quiz cold, then check the Answers section at the bottom of this file.
**Read (45 min):** torchvision ResNet source: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py — read `BasicBlock`, `Bottleneck`, `_make_layer`, and the stem in `ResNet.__init__`.
**Build:**
1. `week07/RESNET_DIFF.md` — exactly 5 bullets diffing torchvision's `BasicBlock`/stem vs yours. Look for: the ImageNet stem (7×7 stride-2 conv + maxpool) vs your CIFAR stem; the `Bottleneck` 1×1–3×3–1×1 block you didn't need; where `downsample` is constructed (`_make_layer`, passed in); `inplace=True` ReLUs; `zero_init_residual` zero-initializing the last BN γ per block.
2. Redo-cold drills (no peeking at your own code; ~15 min each):
   - Re-implement `out_shape` and `receptive_field` from memory; run the Day-1/2 tests against them.
   - Write `BasicBlock.forward` from memory on paper, including the projection-shortcut condition and ReLU placement; diff against `tlib/resnet.py`.
   - Hand-compute the parameter count of `Conv2d(64, 128, 3, bias=False)` plus its BatchNorm; verify in a REPL (expect 73728 + 256).
   - From memory, the three reasons conv beats Linear on images (sharing, locality, translation equivariance) — one sentence each.
3. Self-quiz — answer all 12 cold, then check the Answers section:
   1. Output spatial size of a 5×5 kernel, stride 2, padding 2 on a 64×64 input?
   2. Why does a conv layer's parameter count not depend on input H×W?
   3. rf after three 3×3 stride-1 convs? After a 3×3 stride-2 then two 3×3 stride-1 convs?
   4. In a BasicBlock, why is the ReLU after the add rather than before?
   5. When is a 1×1 projection shortcut required, and what two hyperparameters must it match?
   6. Why is degradation (He §1) not explained by overfitting?
   7. Why set `bias=False` on a conv followed by BatchNorm?
   8. Why must normalization stats come from the train split only?
   9. What does global average pooling buy over flatten+Linear? (two things)
   10. Why is horizontal flip label-preserving on CIFAR-10 while vertical flip is dubious?
   11. Your augmented run had higher train loss but a smaller train/test gap. Explain.
   12. What does `zero_init_residual` in torchvision do, and why might it help early training?
**Verify — done when:** all redo-cold drills match your tested code; quiz self-scored ≥10/12 (revisit the relevant day for anything missed).
**If stuck:** the ResNet paper §3.3; your own Day-4 tests are the ground truth for block anatomy.

---

## Answers (Day 7.7 quiz)
1. floor((64 + 4 − 5)/2) + 1 = floor(63/2) + 1 = 31 + 1 = **32**.
2. Weight sharing: the parameters are the kernel `(C_out, C_in, k, k)` (+ bias) that slides over all positions, so the count depends only on channels and kernel size, never on spatial extent.
3. Stride-1 ×3: rf 1→3→5→**7**. Stride-2 first: rf 1→3 with jump now 2, then +2·2 = 7, then +2·2 = **11**.
4. So the block computes `relu(F(x) + x)`: the skip carries the *raw* input and the residual branch's BN output adds to it before the shared nonlinearity. With F(x) = 0 the block is exactly identity (post-ReLU on a previously-ReLU'd signal), which is the easy-to-learn default the design wants; this is the paper's §3.3 layout.
5. When the block changes channel count and/or uses stride > 1. The 1×1 conv must match the main path's output channels and its stride.
6. Because the deeper plain net has higher *training* error — overfitting would show lower train error with worse test error. Degradation is an optimization failure, not a generalization failure.
7. BN immediately subtracts the per-channel mean, erasing any constant offset the bias adds; BN's own learnable β provides the offset instead, so the conv bias is dead weight.
8. Test-derived stats leak information about the evaluation distribution into preprocessing; accuracy estimates and run-to-run comparisons are no longer honest.
9. (a) Far fewer parameters than flatten+Linear (no H·W factor); (b) the head becomes independent of input spatial size (with some translation robustness as a bonus).
10. CIFAR classes (planes, dogs, ships…) are left/right symmetric in the world, so mirrored images are valid class samples; upside-down examples essentially never occur in the data distribution, so vertical flips push the model off-distribution.
11. Augmentation makes the train set harder to fit — every epoch sees perturbed images, so train loss stays higher — while punishing memorization; test performance therefore holds up relative to train and the gap closes. It is regularization implemented in data space.
12. It initializes the *last* BN's γ in each residual branch to 0, so every block starts as the identity (`relu(0 + x)`); the network begins as an effectively shallow, well-conditioned model and deepens gradually as the γs grow.
