# Week 9 — Attention & the Transformer

This week you build the architecture behind every model you call at work, from the single equation up. You have already built every supporting piece: Linear/Dropout/Sequential (W1–2), stable cross-entropy (W3), AdamW + warmup-cosine (W4, W6), Trainer with clipping (W5), LayerNorm (W6), and an LSTM language model whose validation cross-entropy is your recorded baseline (W8). The transformer is sequenced here because it is *mostly assembly* of those parts — the only genuinely new math is scaled dot-product attention, and you will derive, implement, and property-test it before stacking it. By Friday a decoder-only transformer you wrote beats your LSTM on the same book; by Saturday you know *which* of its components earn their keep, because you ablated them.

**Week outcome:** `tlib/transformer.py` containing `MultiHeadAttention`, `Block`, and `DecoderLM` (all causality-tested and verified against `F.scaled_dot_product_attention` / `nn.MultiheadAttention`); a trained char-level `DecoderLM` checkpoint whose val CE beats your Week-8 LSTM baseline in `experiments.csv`; an ablation table + analysis proving which components matter. Skill: you can write attention from a blank file in under 15 minutes, with the shapes right.

---

## Day 9.1 — Scaled dot-product attention from the equation (~2h)
- [ ] done

**Goal:** Implement `softmax(QK^T/√d_k)V` from the paper's equation and prove it is causal when masked.

**Learn:**
- **Attention as a soft dictionary lookup.** A dict lookup matches a query against keys and returns one value. Attention computes a *similarity score* between one query and *all* keys, softmaxes the scores into weights, and returns a *weighted average of all values*. Differentiable lookup — that's the whole idea.
- **Q, K, V are three views of the same tokens** (in self-attention): each token emits a query ("what am I looking for?"), a key ("what do I contain?"), and a value ("what do I hand over if matched?"). Today they're just three input tensors; Day 2 adds the learned projections that produce them.
- **Why divide by √d_k:** if q and k have i.i.d. entries with mean 0, variance 1, then q·k = Σᵢ qᵢkᵢ is a sum of d_k terms each with variance 1, so Var(q·k) = d_k. At d_k=256 raw scores have std 16; softmax of values that spread is one-hot, gradients through it vanish. Dividing by √d_k restores variance ≈ 1. You will confirm this numerically today.
- **Masking happens *before* softmax, with -inf.** Setting a score to -inf makes its softmax weight exactly 0 (e^{-inf}=0) while the remaining weights still sum to 1. A *causal* mask zeroes attention to future positions (lower-triangular keep-matrix); a *padding* mask zeroes attention to pad tokens. Same mechanism, different pattern.
- **Causality is what makes next-token training valid.** If position t could see t+1, predicting token t+1 from position t would be copying, not modeling. Hence today's most important test: the output at position t must be bit-for-bit independent of inputs after t.

**Read (30–45 min):**
- Attention Is All You Need, §3.2.1 "Scaled Dot-Product Attention" (incl. footnote 4, which gives the variance argument): https://arxiv.org/abs/1706.03762
- Understanding Deep Learning, ch.12 §12.1–12.2 (self-attention as routing, matrix form): free PDF at https://udlbook.github.io/udlbook/
- Skim d2l.ai §11.3 "Attention Scoring Functions" for an alternate derivation: https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html

**Build:** all in `week09/attention.py` + `week09/test_attention.py`.

1. Implement the function:
   ```python
   def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
            mask: torch.Tensor | None = None) -> torch.Tensor:
       """Scaled dot-product attention.
       q: (..., T_q, d_k)   k: (..., T_k, d_k)   v: (..., T_k, d_v)
       mask: bool, broadcastable to (..., T_q, T_k); True = MAY attend,
             False = may NOT (set score to -inf before softmax).
       Returns (..., T_q, d_v).
       """
   ```
   Steps with shapes for the (B, T, d) case: `scores = q @ k.transpose(-2, -1)` → (B, T, T); `scores = scores / math.sqrt(q.shape[-1])`; if mask is not None: `scores = scores.masked_fill(~mask, float("-inf"))`; `w = scores.softmax(dim=-1)` → (B, T, T), rows sum to 1; `return w @ v` → (B, T, d). Use `...` dims so it also works for (B, H, T, hs) on Day 2 unchanged.
2. Causal mask helper: `def causal_mask(T: int) -> torch.Tensor:` returns `torch.tril(torch.ones(T, T, dtype=torch.bool))` — shape (T, T), True on and below the diagonal. Row t (query t) may attend to columns 0..t.
3. Variance experiment, `def variance_demo() -> None:` for `d_k in (4, 16, 64, 256)`: sample `q, k = torch.randn(10_000, d_k)`, compute `dots = (q * k).sum(-1)`; print `dots.var()` (≈ d_k) and `(dots / math.sqrt(d_k)).var()` (≈ 1). Also print, for d_k=256, the mean max softmax weight over rows of a (T=32) score matrix with and without scaling — watch unscaled collapse toward 1.0.
4. Tiny worked example in a docstring or `__main__`: T=2, d=1, q=k=v=[[1.],[2.]] — compute by hand that row 0 of softmax([[1,2],[2,4]]/1) weights value 2 more, and check your function agrees.

**Verify — done when** (`pytest week09/test_attention.py` passes):
- vs torch, no mask: `q,k,v = torch.randn(3, 2, 8, 16)` per tensor (B=2,T=8,d=16); `torch.allclose(sdpa(q,k,v), F.scaled_dot_product_attention(q,k,v), atol=1e-6)`.
- vs torch, causal: `torch.allclose(sdpa(q,k,v, mask=causal_mask(8)), F.scaled_dot_product_attention(q,k,v, is_causal=True), atol=1e-6)`; also pass your bool mask as `attn_mask=causal_mask(8)` (torch's bool `attn_mask` uses the same True=attend convention) and check all three agree.
- **The causality test** — write exactly this; it is the most important test of the week:
  ```python
  def test_causality():
      torch.manual_seed(0)
      B, T, d, t = 2, 8, 16, 3
      x1 = torch.randn(B, T, d); x2 = x1.clone()
      x2[:, t+1:, :] = torch.randn(B, T-t-1, d)   # corrupt the future
      m = causal_mask(T)
      y1, y2 = sdpa(x1, x1, x1, m), sdpa(x2, x2, x2, m)
      assert torch.allclose(y1[:, :t+1], y2[:, :t+1], atol=1e-6)   # past unchanged
      assert not torch.allclose(y1[:, t+1:], y2[:, t+1:])          # future did change
  ```
- Same test WITHOUT the mask must fail the first assert (prove the mask is doing the work).
- `variance_demo()` output matches the theory: unscaled variance ≈ d_k (within ~10%), scaled ≈ 1.

**If stuck:** the paper's §3.2.1 footnote has the exact variance argument; `F.scaled_dot_product_attention` docs list the reference "math" implementation in pseudocode: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

---

## Day 9.2 — Multi-head attention and the reshape dance (~2.5h)
- [ ] done

**Goal:** Build `MultiHeadAttention` with a fused QKV projection and verify it weight-for-weight against `nn.MultiheadAttention`.

**Learn:**
- **Why heads:** one attention head computes ONE weighted average per position — one "relationship type". H heads run attention in H independent d_model/H-dim subspaces in parallel (one can track syntax-distance, another copy recent tokens), then concatenate. Cost is the same as one full-width head.
- **The reshape dance** — memorize this, you'll redo it cold on Day 7. With C = d_model, hs = C // H:
  (B, T, C) —`view(B, T, H, hs)`→ (B, T, H, hs) —`transpose(1, 2)`→ (B, H, T, hs). Now H is a batch-like dim and your Day-1 `sdpa` runs per-head for free. Coming back: (B, H, T, hs) —`transpose(1, 2)`→ (B, T, H, hs) —`.contiguous().view(B, T, C)`→ (B, T, C). The `.contiguous()` is required because transpose returns a non-contiguous view that `view` can't reshape.
- **Fused QKV:** instead of three Linears, one `Linear(C, 3C)` then split — fewer kernel launches, and it matches how `nn.MultiheadAttention` stores its weights (`in_proj_weight`), which you exploit in verification.
- **Output projection:** after concatenating heads, a final `Linear(C, C)` lets heads mix. Without it, head outputs stay in disjoint channel blocks forever.

**Read (30–45 min):**
- Attention Is All You Need §3.2.2 "Multi-Head Attention" (the h=8, d_k=d_v=d_model/h convention): https://arxiv.org/abs/1706.03762
- d2l.ai §11.5 "Multi-Head Attention" — their `transpose_qkv` is the same dance: https://d2l.ai/chapter_attention-mechanisms-and-transformers/index.html
- `torch.nn.MultiheadAttention` docs — read the `in_proj_weight`, `out_proj`, `attn_mask` parameter descriptions: https://docs.pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html

**Build:** create `tlib/transformer.py` (this file grows all week); tests in `week09/test_mha.py`.

1. ```python
   class MultiHeadAttention(Module):   # your tlib Module base, as in Week 7
       def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
           """Causal multi-head self-attention.
           qkv:  Linear(d_model, 3*d_model)  — weight shape (3C, C), rows [0:C]=Q, [C:2C]=K, [2C:3C]=V
           proj: Linear(d_model, d_model)
           drop: Dropout(dropout) applied to the projected output.
           Requires d_model % n_heads == 0 (assert it).
           """
       def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
           """x: (B, T, C) -> (B, T, C). mask broadcastable to (B, H, T, T), True=attend."""
   ```
   Forward with shapes: `qkv(x)` → (B, T, 3C); `q, k, v = out.split(C, dim=-1)` → 3×(B, T, C); dance each to (B, H, T, hs); `y = sdpa(q, k, v, mask)` → (B, H, T, hs) (your Day-1 function — note mask (T,T) broadcasts over B and H); dance back to (B, T, C); `return drop(proj(y))`.
2. Weight-mapping verification script `week09/verify_vs_torch.py`. Build both modules with C=64, H=4, then copy YOUR weights INTO torch's (tlib Linear stores weight (out, in), same as torch — you verified this in Week 2):
   ```python
   ref = nn.MultiheadAttention(64, 4, batch_first=True)
   with torch.no_grad():
       ref.in_proj_weight.copy_(mine.qkv.weight)    # (3C, C): Q rows, then K, then V
       ref.in_proj_bias.copy_(mine.qkv.bias)        # (3C,)
       ref.out_proj.weight.copy_(mine.proj.weight)  # (C, C)
       ref.out_proj.bias.copy_(mine.proj.bias)      # (C,)
   ```
   Compare on `x = torch.randn(2, 8, 64)`, both in eval mode, dropout=0. **Trap:** `nn.MultiheadAttention`'s bool `attn_mask` is INVERTED vs yours and vs `F.sdpa` — there True means *blocked*. Pass `attn_mask=~causal_mask(8)` to torch, `mask=causal_mask(8)` to yours. Call as `ref_out, _ = ref(x, x, x, attn_mask=~causal_mask(8), need_weights=False)`.

**Verify — done when:**
- Shape: input (2, 8, 64) → output (2, 8, 64); `d_model % n_heads != 0` raises.
- Causality test from Day 1, verbatim but through `MultiHeadAttention` (eval mode, causal mask): corrupting tokens after t leaves outputs ≤ t unchanged. The dance has many ways to silently break this — e.g. `view(B, H, T, hs)` instead of view-then-transpose mixes time into heads; this test catches it.
- `torch.allclose(mine(x, causal_mask(8)), ref_out, atol=1e-5)` with the weight copy above, both with and without the causal mask.
- Single-head equivalence: with H=1, MHA(x) equals `proj(sdpa(Wq(x), Wk(x), Wv(x)))` where Wq/Wk/Wv are Linears built from slices `qkv.weight[0:C]`, `[C:2C]`, `[2C:3C]` (and matching bias slices) — allclose, atol 1e-6.

**If stuck:** d2l §11.5 implements exactly this dance with prints; the `nn.MultiheadAttention` source (`torch/nn/modules/activation.py`, then `F.multi_head_attention_forward`) shows the in_proj split order is Q,K,V.

---

## Day 9.3 — The transformer block: norms, MLP, residuals, positions (~2.5h)
- [ ] done

**Goal:** Build the pre-norm transformer `Block` and both positional-encoding variants, and verify the sinusoidal linearity property from the paper.

**Learn:**
- **Residual stream as highway:** each block computes `x = x + f(x)` twice (attention, then MLP). Information flows layer-to-layer through the identity path; blocks write *updates* into it. This is why 100+-layer transformers train at all — gradients reach layer 0 through the skip connections (your Week-7 ResNet lesson, reapplied).
- **Pre-norm vs post-norm:** the 2017 paper normalized *after* the residual add (`LN(x + f(x))`, post-norm); GPT-2 moved LN *inside* the branch (`x + f(LN(x))`, pre-norm). Pre-norm keeps the identity path clean of any nonlinearity, so gradients at init are well-scaled and training is stable without warmup tricks; post-norm typically needs careful warmup. Xiong et al. 2020 proved the gradient-scale argument. You ablate this on Day 6.
- **The MLP:** `Linear(C, 4C) → GELU → Linear(4C, C)`. The 4× expansion is the paper's d_ff = 4·d_model convention; it is where most parameters live (8C² of the block's 12C²).
- **GELU, exactly:** GELU(x) = x·Φ(x) = 0.5·x·(1 + erf(x/√2)). The tanh approximation (used by GPT-2): 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))). Unlike ReLU it is smooth and slightly weights inputs by magnitude.
- **Why positions at all:** attention is a weighted average over a *set* — permute the input tokens and (unmasked) outputs permute identically. Without positional information the model literally cannot represent word order. Two fixes: learned absolute embeddings (a (block_size, C) table, GPT-2 style) or the paper's fixed sinusoids PE(pos, 2i) = sin(pos/10000^{2i/d}), PE(pos, 2i+1) = cos(pos/10000^{2i/d}).

**Read (30–45 min):**
- Attention Is All You Need §3.3 (FFN) and §3.5 (positional encoding, incl. the "linear function of PE_pos" claim): https://arxiv.org/abs/1706.03762
- GPT-2 paper §2.3 for pre-norm placement ("layer normalization was moved to the input of each sub-block"): https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
- Skim Xiong et al. 2020, "On Layer Normalization in the Transformer Architecture", §1 + Fig.1: https://arxiv.org/abs/2002.04745
- GELU: Hendrycks & Gimpel, §2 (both forms): https://arxiv.org/abs/1606.08415

**Build:** in `tlib/transformer.py`; tests in `week09/test_block.py`; plot script `week09/pe_plot.py`.

1. `class GELU(Module):` with `__init__(self, approximate: str = "none")` ("none" = erf form via `torch.erf`, "tanh" = approximation above). Verify both vs `F.gelu(x)` / `F.gelu(x, approximate="tanh")`, allclose atol 1e-6.
2. ```python
   class MLP(Module):
       def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0) -> None: ...
       def forward(self, x: torch.Tensor) -> torch.Tensor:
           """(B, T, C) -> (B, T, C): drop(fc2(gelu(fc1(x)))). fc1: C->4C, fc2: 4C->C."""
   ```
3. ```python
   class Block(Module):
       def __init__(self, d_model: int, n_heads: int, mlp_ratio: int = 4,
                    dropout: float = 0.0, prenorm: bool = True) -> None:
           """ln1, ln2: YOUR tlib LayerNorm(d_model). attn: MultiHeadAttention. mlp: MLP."""
       def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
           # prenorm:  x = x + attn(ln1(x), mask);  x = x + mlp(ln2(x))
           # postnorm: x = ln1(x + attn(x, mask));  x = ln2(x + mlp(x))   (Day 6 ablation)
   ```
   Every intermediate is (B, T, C); the block is shape-preserving by construction.
4. ```python
   def sinusoidal_pe(block_size: int, d_model: int) -> torch.Tensor:
       """(block_size, d_model) fixed table. pe[pos, 2i] = sin(pos/10000**(2i/d_model)),
       pe[pos, 2i+1] = cos(same argument). Build with one outer product, no loops."""
   ```
   `pe_plot.py`: `plt.imshow(sinusoidal_pe(128, 128).T, aspect="auto")`, save `week09/pe.png` — you should see frequency decreasing down the channel axis.

**Verify — done when:**
- GELU both variants allclose vs `F.gelu` (and the two variants differ from *each other* by more than 1e-4 somewhere — proves you implemented two things).
- `Block` keeps shape (2, 16, 64) → (2, 16, 64).
- Causality through depth: stack 4 Blocks (`Sequential` or a loop, causal mask), run the Day-1 corruption test on the stack — outputs at ≤ t unchanged (atol 1e-5; eval mode).
- Sinusoidal linearity (paper §3.5: PE(pos+k) is a linear function of PE(pos), same matrix for all pos): with `pe = sinusoidal_pe(128, 64)` and k=7, solve `M = torch.linalg.lstsq(pe[:-7], pe[7:]).solution` and assert `(pe[:-7] @ M - pe[7:]).abs().max() < 1e-4`. Sanity-contrast: the same residual for a *learned* (randn) table is large.

**If stuck:** paper §3.5 for the PE formula (note the exponent is 2i/d_model, off-by-one here ruins the linearity test); Xiong et al. §3 for why pre-norm gradients are tame; your Week-6 LayerNorm tests if norm shapes fight you.

---

## Day 9.4 — DecoderLM: the full model (~2.5h)
- [ ] done

**Goal:** Assemble token + positional embeddings, N blocks, final norm, and a weight-tied LM head into `DecoderLM`, with exact parameter accounting and an init-loss sanity check.

**Learn:**
- **The decoder-only pipeline:** idx (B, T) ints → token emb (B, T, C) + pos emb (T, C) broadcast → dropout → N causal Blocks → final LayerNorm → LM head Linear(C, V, no bias) → logits (B, T, V). Position t's logits predict token t+1; loss is your Week-3 cross-entropy on shifted targets, exactly like Week 8.
- **Weight tying:** the LM head matrix is the *same tensor* as the token embedding (head computes h·E^T). Both map between token-space and C-space, so sharing is semantically coherent (Press & Wolf 2016), and it deletes V·C parameters — for small models the single biggest matrix.
- **GPT-2 init convention:** all weights ~ N(0, 0.02), biases 0, BUT the two residual-branch output projections per block (`attn.proj`, `mlp.fc2`) get std 0.02/√(2N) for N layers. Why: the residual stream receives 2N additive branch writes; independent writes add in variance, so the stream's variance grows like 2N·σ². Scaling each write by 1/√(2N) keeps the stream O(1) at any depth.
- **Loss at init ≈ ln(V):** with tiny init the logits are near 0, softmax is near uniform, so CE ≈ −ln(1/V) = ln V. This is a *guaranteed* number — a free, sharp sanity check before any training. (V=80 chars → ln 80 ≈ 4.382.)
- **Param accounting as a habit:** you should never be surprised by `count_params()`. Closed form below; deriving it forces you to know where every weight lives.

**Read (30–45 min):**
- Press & Wolf, "Using the Output Embedding to Improve Language Models", §1–2: https://arxiv.org/abs/1608.05859
- UDL ch.12, decoder/transformer-LM sections (12.5–12.7 area): https://udlbook.github.io/udlbook/
- `nn.Embedding` docs (you may use your own from Week 8 if you built one, else `torch.nn.Embedding` is allowed here): https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html

**Build:** in `tlib/transformer.py`; tests in `week09/test_decoderlm.py`.

1. ```python
   class DecoderLM(Module):
       def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int,
                    block_size: int, dropout: float = 0.0, tie_weights: bool = True,
                    pos_type: str = "learned") -> None:
           """pos_type: 'learned' | 'sinusoidal' | 'none' (the last for Day 6's ablation).
           Registers a (block_size, block_size) causal mask buffer once (torch.tril)."""
       def forward(self, idx: torch.Tensor) -> torch.Tensor:
           """idx: (B, T) int64, T <= block_size (assert) -> logits (B, T, vocab_size)."""
       def count_params(self) -> int:
           """Number of trainable parameters (tied weight counted once)."""
       @torch.no_grad()
       def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
           """Greedy: loop max_new_tokens times: crop idx to last block_size tokens,
           forward, take logits[:, -1, :] (B, V), argmax -> (B, 1), cat to idx.
           idx: (B, T0) -> (B, T0 + max_new_tokens). Week 10 replaces this with real sampling."""
   ```
   Tying: after building the head, set `head.weight = tok_emb.weight` (same tensor object, not a copy — `assert model.head.weight is model.tok_emb.weight`). Apply the init scheme in a `_init_weights` pass; collect the residual projections by name.
2. Closed-form param formula (module-level function):
   ```python
   def expected_param_count(vocab_size: int, d_model: int, n_layers: int,
                            block_size: int, tie_weights: bool = True,
                            pos_type: str = "learned") -> int:
       """V*C (tok emb) + T*C (if learned pos) + n_layers*(12*C*C + 13*C)
       + 2*C (final LN) + (0 if tied else V*C).
       Per block: qkv 3C²+3C, proj C²+C, fc1 4C²+4C, fc2 4C²+C, ln1+ln2 4C."""
   ```
3. Smoke config for tests: `DecoderLM(80, 64, 4, 2, 32)`.

**Verify — done when:**
- `model.count_params() == expected_param_count(...)` **exactly**, for at least 3 configs (tied/untied, learned/sinusoidal — sinusoidal table is a buffer, not a parameter).
- Forward shape: (4, 32) int64 in → (4, 32, 80) out; T=16 < block_size also works; T=33 raises.
- Whole-model causality test (eval mode): build two (2, 32) index tensors identical up to position t=10, different after; logits at positions ≤ 10 allclose (atol 1e-5), different after. *This single test exercises mask wiring through embedding → blocks → head.*
- Init loss: batch (8, 32) of random indices, CE of logits vs random targets satisfies `abs(loss - math.log(80)) < 0.1` (eval mode). Assert it in pytest.
- Tying: `is`-identity assert above, and untied model has exactly `vocab_size*d_model` more params.
- Greedy `generate` from a (1, 4) prompt returns shape (1, 24) for max_new_tokens=20, and is deterministic across two calls.

**If stuck:** Press & Wolf §2 for tying; GPT-2 paper §2.3 lists the 1/√N residual scaling; recount params per block on paper — the formula bug is always a forgotten bias or a double-counted tied head.

---

## Day 9.5 — Train it: beat your LSTM (~2h hands-on + CPU training time)
- [ ] done

**Goal:** Train a ~0.8M-param char-level DecoderLM on the Week-8 book with your Trainer and beat the recorded LSTM baseline on val cross-entropy.

**Learn:**
- **Identical data protocol = comparable numbers.** Same book, same `CharVocab`, same train/val split, same per-character CE metric as Week 8 — otherwise "beats the baseline" is meaningless. Reuse `tlib/text.py` untouched.
- **Why the transformer should win:** the LSTM squeezes all history through one fixed-size recurrent state; attention gives every position direct, learned access to all 128 previous positions. On char-level text that means cleanly closing quotes, matching names, sustaining words.
- **The recipe is your Week 4–6 stack:** AdamW, linear warmup → cosine decay, grad-clip 1.0, dropout. Nothing new today — that's the point of the sequencing.

**Read (20–30 min):**
- UDL ch.12 training-of-transformers notes; skim Attention Is All You Need §5.3 (optimizer/warmup) and map each ingredient to a tlib component you built: https://arxiv.org/abs/1706.03762

**Build:** `week09/train_decoder.py`.

1. Config (CPU-sized; put it in a dict at the top): vocab from your `CharVocab` (~60–90 for a Gutenberg book), `d_model=128, n_heads=4, n_layers=4, block_size=128, dropout=0.1, tie_weights=True, pos_type="learned"` → ~0.81M params at V=80 (assert `count_params()` against your formula, printed at startup).
2. Data: `SequenceDataset` from Week 8 with block 128; batch_size 32 (drop to 16 if RAM-bound).
3. Training: AdamW(lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1 — apply decay only to ≥2-D params, as in your Week-4 verification), warmup 200 steps then cosine to 10% of peak over `max_steps=3000`, clip 1.0, your Trainer with checkpointing; eval val CE every 250 steps on a fixed batch set. Expect very roughly 20–60 min on CPU; run with `nohup`/background and reduce `max_steps` if needed (note any reduction in the log).
4. After training: print a 300-char greedy sample from a 10-char prompt; append a row to `experiments.csv` (same file as Week 8): `week,model,params,val_ce,notes`.

**Verify — done when:**
- A startup assert passes: init val CE ≈ ln(V) (the Day-4 check, on real data this time).
- Final val CE **< your Week-8 LSTM row in experiments.csv** — write the assert by reading the CSV, e.g. `assert transformer_val_ce < lstm_val_ce`, and print the margin. (Typical at this scale: a clear win; if you don't beat it, suspect masking/data bugs before tuning — the Day-4 causality test is your first stop.)
- Loss curve is sane: monotone-ish decline, no spike after warmup ends.
- Greedy sample is qualitatively better than Week 8's LSTM sample — real words, plausible punctuation (subjective; paste both into LOG.md and judge).

**If stuck:** your Week-5 Trainer tests; Week-6 schedule plots (a wrong warmup is the classic silent killer); re-run `week09/test_decoderlm.py` before blaming hyperparameters.

---

## Day 9.6 — Ablation lab: which components earn their keep? (~3.5–4h, deep build)
- [ ] done

**Goal:** Under a fixed compute budget, ablate heads, positional embeddings, pre-norm, and residuals, and rank what mattered with evidence.

**Learn:**
- **Ablation = controlled experiment:** change exactly one thing, hold budget/seed/data fixed, compare one metric. This is the skill that separates "I read the paper" from "I know why it's built this way".
- **No positions ⇒ permutation-equivariant attention:** without pos emb, self-attention (pre-mask) treats input as a set; the only order signal left is the causal mask's prefix structure, which is far weaker. Expect clear degradation.
- **No residuals ⇒ Week-7 déjà vu:** at depth 4+ with no skip path, signal and gradients must survive every block multiplicatively; pre-norm helps but training still degrades badly or stalls.
- **Post-norm may *train*, just worse/twitchier** at this scale without extra warmup — observe, don't prejudge; report what actually happened.

**Build:** `week09/ablate.py`.

1. Budget per run (fixed; state it in the script): 1500 steps, batch 32, block 128, same warmup-cosine (warmup 200), `torch.manual_seed(1337)` before model build AND before data shuffling, identical eval batches. Five runs:
   | run | change vs baseline |
   |---|---|
   | `baseline` | D9.5 config, 1500 steps |
   | `heads1` | `n_heads=1` (same d_model=128) |
   | `no_pos` | `pos_type="none"` |
   | `postnorm` | `prenorm=False` in every Block |
   | `no_residual` | Block computes `x = f(ln(x))` with no `x +` (add `use_residual: bool = True` flag to Block) |
2. Each run appends to `experiments.csv`: `run,steps,params,final_train_ce,final_val_ce,wallclock_s`.
3. `week09/LOG.md`: ~12 lines — table, then a ranking of components by damage-when-removed, with one mechanistic sentence each (use the Learn bullets as hypotheses to confirm or amend), plus anything surprising (e.g. how bad was 1 head *really* at this tiny scale? was post-norm fine?).

**Verify — done when:**
- All five rows exist with the same `steps` value.
- Directional asserts at the end of `ablate.py` (only the robust ones — heads and post-norm results are scale-dependent, so report those without asserting):
  ```python
  assert results["no_pos"] > results["baseline"]        # val CE, higher = worse
  assert results["no_residual"] > results["baseline"]
  ```
- LOG.md ranks components and every claim cites a number from the table.

**If stuck:** UDL ch.12 on positional encodings & permutation equivariance; Xiong et al. 2020 for post-norm behavior; your Week-7 plain-vs-residual CNN experiment for the no-residual prior.

---

## Day 9.7 — Review, quiz, redo-cold (~2h)
- [ ] done

**Goal:** Consolidate: rebuild the core from memory, take the quiz closed-book, and swap in torch's fused attention behind a flag.

**Redo-cold drills (closed book, from a blank file):**
1. **(≤15 min)** Rewrite `sdpa` + `causal_mask` + the causality test from memory in `week09/redo_sdpa.py`; run the test. Shapes annotated as comments on every line.
2. **(≤5 min)** On paper: the full reshape dance (B,T,C)→(B,H,T,hs)→back, with the exact `view`/`transpose`/`contiguous` calls. Check against Day 2.
3. **(≤5 min)** Write `expected_param_count` from memory; check it against `tlib/transformer.py` for the D9.5 config.
4. **(≤5 min)** Pre-norm block forward, both residual lines, from memory.

**Extra read + build (45 min):** `F.scaled_dot_product_attention` docs — backend selection (flash / memory-efficient / math) and the `torch.nn.attention.sdpa_kernel` context manager: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
Add `use_fused: bool = False` to `MultiHeadAttention.__init__`; when True, call `F.scaled_dot_product_attention(q, k, v, is_causal=True)` on your (B, H, T, hs) tensors instead of your `sdpa`. Verify: with the math backend forced (`with sdpa_kernel([SDPBackend.MATH]):`), fused vs your path allclose (atol 1e-5) on the trained model's logits. Then time 50 forward passes of the D9.5 model both ways on CPU (`time.perf_counter`, eval mode, no_grad) and record the measured ratio in LOG.md — whatever it is; CPU backends vary, promise nothing.

**Self-quiz** (write answers, then check against the bottom of this file):
1. Why √d_k and not d_k? State the variance argument precisely.
2. Why must masking use -inf *before* softmax instead of zeroing weights *after*?
3. What property does the causality test check, and what's the exact experimental setup?
4. In the dance, why `view` then `transpose` rather than `view(B, H, T, hs)` directly?
5. How do `in_proj_weight` rows map to Q/K/V, and what's inverted about `nn.MultiheadAttention`'s bool `attn_mask`?
6. Pre-norm vs post-norm: write both forward formulas and say why pre-norm is more stable.
7. Without positional embeddings, what symmetry does self-attention have, and what's the only remaining order signal in a causal LM?
8. Why is loss ≈ ln(V) at init guaranteed, and what bug classes does asserting it catch?
9. Why scale residual-projection init by 1/√(2N)?
10. Weight tying: what equals what, and how many params does it save?
11. Give the per-block param count in terms of C and identify which sub-module dominates.
12. Which ablation hurt most in YOUR run, and what's the mechanism?

**Foreshadow:** Week 10 gives `DecoderLM` a real tokenizer (your own BPE) and real sampling + a KV cache; Week 11 profiles and speeds up exactly this model and training loop; Week 12 pretrains a bigger one on a real dataset with the same tokenizer code. Nothing this week is throwaway.

---

## Answers (Day 9.7 quiz)

1. For q, k with i.i.d. mean-0 variance-1 entries, q·k = Σᵢqᵢkᵢ has variance d_k (sum of d_k independent variance-1 terms), so std grows as √d_k. Dividing by √d_k makes score variance 1 regardless of d_k, keeping softmax out of its saturated (near-one-hot) regime where gradients vanish. Dividing by d_k would over-shrink (variance 1/d_k), washing scores toward uniform.
2. -inf before softmax gives the masked position weight exactly e^{-inf}=0 *and* renormalizes remaining weights to sum to 1. Zeroing after softmax breaks the sum-to-1 property (output becomes a shrunken average), and the masked positions still influenced the normalizer — information leaks through the denominator.
3. Output at position t depends only on inputs 0..t. Setup: two inputs identical through position t, randomly different after; with the causal mask, outputs at 0..t must be allclose and outputs after t must differ. Without the mask the first comparison must fail.
4. The memory layout of (B, T, C) puts each token's C channels contiguous. `view(B, T, H, hs)` splits channels into heads correctly; a direct `view(B, H, T, hs)` would reinterpret memory so that "head" slices mix different *time steps* — silently destroying causality and meaning. The transpose then just reorders dims so H batches.
5. `in_proj_weight` is (3C, C): rows [0:C] are W_q, [C:2C] W_k, [2C:3C] W_v (`in_proj_bias` likewise). For `nn.MultiheadAttention`, bool `attn_mask=True` means *not allowed* to attend — the opposite of `F.scaled_dot_product_attention` and of our `sdpa`, so you pass `~causal_mask(T)`.
6. Pre-norm: `x = x + attn(ln1(x)); x = x + mlp(ln2(x))`. Post-norm: `x = ln1(x + attn(x)); x = ln2(x + mlp(x))`. Pre-norm leaves the identity path untouched, so gradient magnitude at init is roughly depth-independent and no warmup is needed; post-norm puts LN on the main path, producing large early gradients in deep stacks (Xiong et al. 2020) that demand warmup.
7. Permutation equivariance: permuting input tokens permutes (unmasked) outputs identically — the model sees a set. With a causal mask the only order signal is *how many* tokens are visible (prefix length), which can't encode relative order within the prefix.
8. Near-zero logits ⇒ near-uniform softmax ⇒ CE ≈ ln V, with deviation bounded by the tiny logit scale (hence ±0.1). Asserting it catches: oversized init, missing final norm, softmax/CE applied on the wrong dim, off-by-one target shifting, and accidentally pre-trained weights.
9. The residual stream accumulates 2N additive branch outputs over N blocks. Independent additions grow stream variance ∝ 2N·σ²; scaling each branch's output projection init by 1/√(2N) cancels the growth so activations stay O(1) at any depth (GPT-2 convention).
10. The LM head's weight matrix *is* the token-embedding matrix (same tensor; logits = h·E^T). Saves V·C parameters — e.g. 80×128 = 10,240 here; on real vocabs (50k×768) it's tens of millions.
11. 12C² + 13C: qkv 3C²+3C, attn proj C²+C, fc1 4C²+4C, fc2 4C²+C, two LayerNorms 4C. The MLP dominates with 8C² of the 12C².
12. Your numbers — but expect `no_residual` or `no_pos` at the top. Mechanisms: no residual ⇒ no identity path for signal/gradient at depth (optimization failure); no positions ⇒ order-blind model that can only exploit prefix-length cues (representation failure). Cite your table.
