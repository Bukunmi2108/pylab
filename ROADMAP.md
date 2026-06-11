# PyTorch Expertise: 14-Week Curriculum — Master Index

A self-contained, day-by-day program. Beginner → expert: tensors → autograd → your own
nn/optim → training engineering → transformers → pretraining → performance → finetuning →
preference optimization. Every day's full instructions live in `curriculum/weekNN.md`.

## How this works

- **Days 1–5 each week:** one focused session (1.5–2.5 h). **Day 6:** deep build (3–4 h).
  **Day 7:** review + self-quiz (answers included in each week file) + redo-cold drills.
- Every day has: **Learn** (concepts explained), **Read** (exact doc sections/papers),
  **Build** (exact files, signatures, behavior specs), **Verify — done when** (runnable
  checks: pytest, `torch.allclose` against PyTorch's own implementation, finite
  differences, guaranteed numeric values), **If stuck** (doc/source pointers — never AI).
- You accumulate **your own library, `tlib/`** — modules, optimizers, trainer, data
  utilities, transformer, tokenizer, generation. By week 14 it's a small framework you
  wrote entirely yourself, and the expertise is the point.

## Rules

1. **No generative models.** Not for code, not for hints. Each day's *If stuck* section
   gives you doc/source pointers instead. (This file is the last AI-written thing you need.)
2. **No videos.** Reading material is official PyTorch docs, PyTorch source, original
   papers, *Understanding Deep Learning* (free, udlbook.github.io), d2l.ai.
3. **Type every line.** Never paste.
4. **A day is done when its Verify block passes** — not when the code "looks right".
5. **Keep `LOG.md`:** one line per day — what you built, what surprised you.
6. **Slip policy:** compress *Read*, never *Build*. Never skip a Verify to stay on
   schedule. Day numbers are suggestions; Verify blocks are gates.

## Setup (Day 0)

```bash
cd ~/code/torch
uv init && uv add torch numpy pytest jupyter matplotlib
mkdir -p tlib && touch tlib/__init__.py LOG.md
```
Extras added when a week calls for them: `torchvision` (wk 4, 7), `datasets` (wk 12–14).
Everything runs on CPU; GPU-variant notes appear where a GPU helps (wk 11–12 especially).

## The 14 weeks

| Wk | File | Theme | You can, by the end |
|----|------|-------|---------------------|
| 1 | `curriculum/week01.md` | Tensor fundamentals | Predict any broadcast/reduction result without running it; stable softmax/CE from primitives |
| 2 | `curriculum/week02.md` | Memory model & advanced ops | Explain view-vs-copy via strides; matmul from `as_strided`; einsum fluency; gather/scatter |
| 3 | `curriculum/week03.md` | Autograd | Build a scalar autograd engine whose grads match `torch.autograd` to 1e-6; own gradcheck |
| 4 | `curriculum/week04.md` | nn & optimizers from scratch | Your `tlib` Linear/init/losses/SGD/Adam train FashionMNIST, verified vs `torch.nn`/`optim` |
| 5 | `curriculum/week05.md` | Data pipelines & the training loop | Custom Datasets/samplers/collate; a reusable `tlib.Trainer` with clipping, accumulation, checkpointing |
| 6 | `curriculum/week06.md` | Training dynamics & normalization | Hook-based diagnostics; BatchNorm/LayerNorm/RMSNorm from scratch; LR schedules; fix sick runs |
| 7 | `curriculum/week07.md` | Convolutions & ResNet | conv2d via unfold verified vs `nn.Conv2d`; residual nets on CIFAR-10 with your Trainer |
| 8 | `curriculum/week08.md` | Embeddings & LM foundations | n-gram and MLP language models; LSTM cell from its equations; perplexity you can explain |
| 9 | `curriculum/week09.md` | Attention & the transformer | Decoder-only transformer fully from scratch; ablations (heads, pos-emb, residuals) measured |
| 10 | `curriculum/week10.md` | Tokenization & generation | BPE trained from scratch with round-trip guarantees; temperature/top-k/top-p; KV cache w/ correctness proof |
| 11 | `curriculum/week11.md` | Performance engineering | Profile-first optimization: AMP, `torch.compile`, checkpointing — each change measured in tokens/sec |
| 12 | `curriculum/week12.md` | Scaling out & extending | DDP training; `torch.distributed` primitives; custom `autograd.Function` passing gradcheck; **pretrain your base LM** |
| 13 | `curriculum/week13.md` | Finetuning | Classification FT, instruction SFT with loss masking, LoRA implemented from the paper |
| 14 | `curriculum/week14.md` | Preference optimization & capstone | DPO loss from the paper; eval harness; a chat model you built end-to-end + final report |

## Continuity (artifacts that carry forward)

- `tlib/modules.py`, `tlib/optim.py` (wk 4) → used everywhere after.
- `tlib/trainer.py`, `tlib/data.py` (wk 5) → all later training.
- Diagnostics/hooks toolkit (wk 6) → debugging in wks 7–14.
- Char corpus + LM baselines (wk 8) → transformer comparison (wk 9).
- `tlib/transformer.py` (wk 9) + `tlib/tokenizer.py`, `tlib/generate.py` (wk 10) → the model.
- Speed work (wk 11) + DDP (wk 12) → **base checkpoint pretrained at end of wk 12** →
  finetuned (wk 13) → DPO + capstone (wk 14).

## Progress

Mark days off inside each week file (`- [x]`). Weekly gate = Day 6 Verify + Day 7 quiz.
