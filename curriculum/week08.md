# Week 8 — Embeddings & language-modeling foundations

This week you pivot from images to text and build the entire pre-transformer history of language modeling in seven days: counting, logistic regression, MLPs over context windows, RNNs, and LSTMs — all on one corpus, all evaluated with one metric (cross-entropy / perplexity), all trained with your own `tlib` stack. The sequencing matters: each day's model is the previous day's model plus exactly one idea, and each day's validation CE must beat the last. The LSTM number you record on Day 6 is the baseline your Week 9 transformer must beat on this same corpus; in Week 10 you will replace today's char vocab with your own BPE.

**Week outcome:** `tlib/text.py` (vocab + sequence dataset), `week08/bigram_count.py`, `week08/bigram_sgd.py`, `week08/mlp_lm.py`, `week08/rnn_scratch.py`, `week08/lstm_lm.py`, and `week08/experiments.csv` with a strictly improving val-CE column ending at a trained LSTM. Skill: you can state and verify the week's equivalences (embedding = one-hot matmul; trained bigram = counted bigram; ppl = exp(CE)) and write the RNN/LSTM equations from memory.

## Day 8.1 — Text → tensors (~2.5h)
- [ ] done
**Goal:** Build the corpus pipeline: char vocab, positional split, and the shifted-by-one sequence dataset that everything this week trains on.
**Learn:**
- *Tokenization granularity:* chars (tiny vocab ~80, long sequences, no out-of-vocab problem) vs words (huge vocab, OOV pain) vs subwords (the modern compromise — you build BPE yourself in Week 10). This week: chars, so the modeling is the hard part, not the tokenizer.
- *Vocab = two dicts:* `stoi` maps char→int, `itos` maps back. Everything downstream is integer tensors; text exists only at the boundaries.
- *Split BY POSITION, never shuffle:* take the first 90% of the *character stream* as train, the last 10% as val. Shuffling windows before splitting leaks: a val window overlaps train windows shifted by one char — the model has effectively seen the answer. Positional splitting is the text equivalent of "no test stats in preprocessing".
- *nn.Embedding is a lookup, nothing more:* row `i` of a learnable matrix `W (V, D)`. Mathematically `one_hot(ids) @ W`; the lookup is just the efficient implementation, and gradient flows only into the rows that were actually used.
- *The LM training pair:* input `x = ids[i : i+T]`, target `y = ids[i+1 : i+T+1]`. One window yields `T` next-char prediction problems at once.
**Read (30–45 min):**
- d2l.ai ch. 9, "Converting Raw Text into Sequence Data": https://d2l.ai/chapter_recurrent-neural-networks/text-sequence.html
- `torch.nn.Embedding` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html
- `torch.nn.functional.one_hot` docs: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html
**Build:**
1. Download the corpus — Pride and Prejudice, plain text, from Project Gutenberg:
   ```bash
   mkdir -p data
   curl -L -o data/pride_raw.txt https://www.gutenberg.org/cache/epub/1342/pg1342.txt
   ```
2. `week08/corpus.py`:
   ```python
   def load_corpus(path: str = "data/pride_raw.txt") -> str:
       """Strip the Project Gutenberg header/footer: keep only the text strictly
       between the line containing
         '*** START OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***'
       and the line containing
         '*** END OF THE PROJECT GUTENBERG EBOOK PRIDE AND PREJUDICE ***'.
       Return the stripped string."""
   ```
   Assert the result is > 600_000 chars and contains neither "Project Gutenberg" boilerplate nor the START marker.
3. `tlib/text.py` — the vocab:
   ```python
   class CharVocab:
       def __init__(self, text: str): ...   # sorted unique chars -> stoi/itos
       def encode(self, s: str) -> Tensor: ...   # long tensor of ids
       def decode(self, ids: Tensor | list[int]) -> str: ...
       def __len__(self) -> int: ...
   ```
   Property test: for 50 random substrings `s` of the corpus, `vocab.decode(vocab.encode(s)) == s`.
4. `tlib/text.py` — the dataset:
   ```python
   class SequenceDataset(Dataset):
       def __init__(self, ids: Tensor, block_size: int): ...
       def __len__(self) -> int: ...        # len(ids) - block_size
       def __getitem__(self, i) -> tuple[Tensor, Tensor]:
           """(ids[i : i+block_size], ids[i+1 : i+1+block_size])"""
   ```
   Pytest with `ids = torch.arange(10), block_size=4`: item 0 is `([0,1,2,3], [1,2,3,4])`; the last item's target ends at `ids[-1]`; `len(ds) == 6`; index 6 raises IndexError.
5. Embedding-equivalence demo in `tlib/test_text.py`: with random `W = torch.randn(V, 16)` and random `ids` of shape `(8, 5)`, assert `torch.allclose(F.embedding(ids, W), F.one_hot(ids, V).float() @ W)`.
6. ```python
   def split_ids(ids: Tensor, frac: float = 0.9) -> tuple[Tensor, Tensor]:
       """Positional split: first frac as train, rest as val. No shuffling."""
   ```
   Assert `torch.equal(torch.cat([train, val]), ids)`.
**Verify — done when:** pytest passes; vocab size prints (approximately 60–90 chars for this book — record yours); round-trip and equivalence asserts hold exactly.
**If stuck:** print `repr(text[:200])` after stripping to catch a stray BOM or marker remnant; the Embedding doc's note on which rows receive gradient.

## Day 8.2 — Bigram counting LM (~2h)
- [ ] done
**Goal:** Build the simplest possible LM — a smoothed count table — and nail the evaluation metric you'll use all week.
**Learn:**
- *Language modeling = next-token distribution:* a model that outputs `P(next char | context)`. Everything from here to GPT differs only in how much context it uses and how it computes that distribution.
- *Bigram counting:* context = one char. `counts[a, b]` = how often `b` follows `a` in train; normalizing rows gives the conditional distribution. This is the maximum-likelihood estimate.
- *Additive (Laplace) smoothing:* `P = (counts + λ) / row_sum(counts + λ)`. Unseen pairs get nonzero mass — without it, one unseen val bigram makes CE infinite. As λ→∞ every row tends to uniform.
- *Cross-entropy and perplexity:* `CE = mean over positions of −log P(true next char)`, in nats. `ppl = exp(CE)`: "the model is as confused as a uniform choice among ppl options". The uniform model has CE = ln(V) exactly — a guaranteed yardstick for any V.
- *Sampling:* draw the next char from the current row with `torch.multinomial`, feed it back in, repeat. Generation is iterated sampling from conditionals.
**Read (30–45 min):**
- d2l.ai 9.3 "Language Models" (the perplexity section especially): https://d2l.ai/chapter_recurrent-neural-networks/language-model.html
- `torch.multinomial` docs: https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html
**Build:** `week08/bigram_count.py` + tests:
1. ```python
   def bigram_counts(ids: Tensor, V: int) -> Tensor:
       """(V, V) float tensor; counts[a, b] = number of occurrences of pair (a, b).
       Vectorize: torch.bincount(ids[:-1] * V + ids[1:], minlength=V*V).view(V, V).
       No Python loop over the corpus."""

   def bigram_probs(counts: Tensor, lam: float = 1.0) -> Tensor:
       """Rows sum to 1. assert torch.allclose(P.sum(1), torch.ones(V))."""

   def bigram_ce(P: Tensor, ids: Tensor) -> float:
       """Mean of -log P[ids[:-1], ids[1:]], in nats."""

   def sample(P: Tensor, start: int, n: int,
              generator: torch.Generator | None = None) -> Tensor:
       """Generate n ids by iterated torch.multinomial from rows of P."""
   ```
2. Report train and val CE and ppl for `lam=1.0`; generate and print 500 decoded chars.
3. Start `week08/experiments.csv` with header `model,context,params,train_ce,val_ce,val_ppl,notes` and append the `bigram_count` row.
4. Tests:
   - `test_uniform_yardstick`: a uniform P gives CE = `math.log(V)` to 1e-6.
   - `test_better_than_uniform`: `val_ce < math.log(len(vocab))` — guaranteed for any real text, since characters are far from uniformly distributed.
   - `test_smoothing_limit`: with `lam=1e9`, `abs(train_ce - math.log(V)) < 1e-3` — rows are essentially uniform, the λ→∞ limit property.
   - `test_counts_tiny`: on `encode("abab")`, counts[a→b] == 2 and counts[b→a] == 1, all else 0.
**Verify — done when:** all four tests pass; val CE is approximately 2.2–2.6 nats for this corpus (record YOUR number — Days 3 and 4 are measured against it); the 500-char sample shows English letter statistics but no real words (approximately — expect gibberish with plausible letter pairs).
**If stuck:** the pair-indexing is `P[ids[:-1], ids[1:]]` — Week 2 advanced indexing; d2l 9.3's perplexity definition; verify the tiny "abab" case by hand before trusting the corpus-scale numbers.

## Day 8.3 — Bigram as logistic regression (~1.5–2h)
- [ ] done
**Goal:** Train a `(V, V)` logit table with your AdamW and show gradient descent rediscovers the counting solution.
**Learn:**
- *Same model, different solver:* a one-hot input times a `(V, V)` weight matrix, softmaxed, is exactly a bigram model. Counting computes its maximum-likelihood solution in closed form; gradient descent finds it iteratively. Same model class ⇒ same optimum.
- *Why this matters:* it's the bridge from statistics to deep learning. Every LM after today is "the count table, except the table is computed by a network from a longer context".
- *Regularization ≈ smoothing:* weight decay pulls logits toward 0, i.e. rows toward uniform — the same direction λ-smoothing pushes probabilities. Two dialects of one idea.
**Read (20–30 min):**
- `torch.nn.CrossEntropyLoss` docs — it takes *logits* and fuses log-softmax + NLL: https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
- Your own `tlib/losses.py` and `tlib/optim.py` — you are about to rely on them for a real result.
**Build:** `week08/bigram_sgd.py`:
1. ```python
   class BigramLM(nn.Module):
       def __init__(self, V: int):
           """self.logits = nn.Parameter(torch.zeros(V, V))"""
       def forward(self, x: Tensor) -> Tensor:
           """Row lookup: self.logits[x], shape (*x.shape, V). Embedding-style —
           identical math to one_hot(x) @ logits (Day 1's equivalence)."""
   ```
2. Train on all consecutive pairs of the train split (large batches or full-batch; your AdamW with `weight_decay=0.0`, your cross-entropy; a few hundred steps — the objective is convex per row and converges fast).
3. The comparison baseline: the *unsmoothed* (`lam=0`) count model's TRAIN CE — the analytic minimum of this objective. Compare train CE to train CE. (Val CE needs smoothing to avoid infinities on unseen pairs; the clean mathematical statement is about the training optimum.)
4. Exercise (write conclusions as comments): rerun with `weight_decay ∈ {0, 0.1, 1.0}` and compare the trained rows' distance from uniform against count tables with `lam ∈ {0, 1, 10}`. Three sentences on the smoothing ↔ weight-decay correspondence.
5. Append the `bigram_sgd` row to `experiments.csv`.
**Verify — done when:** `assert abs(train_ce_sgd - train_ce_count_unsmoothed) < 0.01` — mathematically expected, not luck: identical model class, same objective, and counting is its closed-form MLE (state this in a comment); val CE approximately equals Day 2's smoothed val CE.
**If stuck:** if it won't converge, raise the LR (this problem tolerates ~0.1 with Adam) or run more steps; confirm `weight_decay=0` for the equivalence test — decay deliberately biases the optimum away from the MLE.

## Day 8.4 — MLP n-gram LM (Bengio 2003) (~2.5h)
- [ ] done
**Goal:** Replace the table with a network over a k-char context window and beat the bigram.
**Learn:**
- *The table doesn't scale:* a k-char context as a table needs V^k rows. Bengio's move: *embed* each context char into a small dense vector, concatenate, and let an MLP compute the distribution. Parameters grow linearly in k, not exponentially.
- *The architecture:* `ids (B, k) → Embedding (B, k, D) → flatten (B, k·D) → Linear + nonlinearity (B, H) → Linear (B, V)`. Today's context window is the direct ancestor of the transformer's.
- *Capacity/context tradeoffs:* bigger k = more information but more parameters and sparser effective data per pattern; D and H trade compute for capacity. You'll see this in val CE, not just prose.
- *Distributed representations:* chars used in similar contexts (digits, vowels, punctuation) end up with similar embedding vectors — the central claim of Bengio §1, and the reason the model generalizes to contexts it never saw verbatim.
**Read (30–45 min):**
- Bengio, Ducharme, Vincent, Jauvin (2003), "A Neural Probabilistic Language Model", §1–3: https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
- d2l.ai 9.3 again — the Markov / n-gram sections: https://d2l.ai/chapter_recurrent-neural-networks/language-model.html
**Build:** `week08/mlp_lm.py`:
1. ```python
   class MLPLM(nn.Module):
       def __init__(self, V: int, k: int = 8, d: int = 24, h: int = 256):
           """nn.Embedding(V, d) -> flatten to (B, k*d) -> your tlib Linear(k*d, h)
           -> tanh (or ReLU) -> your tlib Linear(h, V)."""
       def forward(self, x: Tensor) -> Tensor:
           """x: (B, k) ids -> (B, V) logits."""
   ```
   Print the parameter count (approximately 60–120k with these defaults and your V).
2. Data: targets are the single char after the window. Either reuse `SequenceDataset(ids, block_size=k)` and train on `(x, y[:, -1])`, or add a thin wrapper:
   ```python
   class NextCharDataset(Dataset):
       """__getitem__(i) -> (ids[i : i+k], ids[i+k])"""
   ```
3. Train with YOUR Trainer + AdamW + warmup-cosine. Budget: 2–3 epochs over the train split, batch 256 (≈10–15 min CPU). GPU variant: 5+ epochs, h=512.
4. Evaluate val CE the same way for every model this week: mean CE over all val positions. Sample 500 chars by sliding the k-window forward (seed with the corpus's first k chars).
5. Append the `mlp_k8` row to `experiments.csv`.
**Verify — done when:** `assert val_ce_mlp < val_ce_bigram - 0.15` — relative to YOUR recorded Day-2 number; the gap is approximately 0.3–0.6 nats at this budget, but assert only the modest floor. The sample is visibly more wordlike than Day 2's — short real words appear (subjective; note your impression in the CSV `notes` column).
**If stuck:** Bengio §3's figure maps one-to-one onto your forward — diff against it; first overfit 1000 windows to near-zero CE to validate the pipeline; if val CE barely beats bigram, LR is the first suspect (use your Week 5 LR finder).

## Day 8.5 — RNN from scratch (~2.5h)
- [ ] done
**Goal:** Implement the vanilla RNN cell from its equation, verify against `nn.RNNCell`, and measure the exploding/vanishing-gradient problem directly.
**Learn:**
- *The recurrence:* `h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)`. A fixed-size state `h` summarizes the entire prefix — context is no longer a fixed window.
- *BPTT is just autograd:* unroll the loop over T steps, sum the per-step losses, call `.backward()`. "Backpropagation through time" is a name, not a new algorithm — your Week 3 understanding already covers it.
- *Truncated BPTT:* you can't unroll a 700k-char book. Train on windows of length T; optionally carry `h` across windows but `.detach()` it, which cuts the gradient path and bounds both memory and gradient depth.
- *Exploding/vanishing gradients:* each backward step multiplies by `W_hh^T · diag(tanh′)`. Repeated multiplication by roughly the same matrix gives geometric growth or decay in T. Explosion has a cheap fix — clipping (already in your Week 5 Trainer); vanishing needs an architectural fix — tomorrow.
**Read (30–45 min):**
- d2l.ai 9.4–9.5, "Recurrent Neural Networks" (including the gradient discussion): https://d2l.ai/chapter_recurrent-neural-networks/rnn.html
- `nn.RNNCell` docs — the exact formula and parameter names; you must match them to copy weights: https://docs.pytorch.org/docs/stable/generated/torch.nn.RNNCell.html
- Pascanu, Mikolov, Bengio (2013), "On the difficulty of training recurrent neural networks", §1–2: https://arxiv.org/abs/1211.5063
**Build:** `week08/rnn_scratch.py` + tests:
1. ```python
   class MyRNNCell(nn.Module):
       def __init__(self, input_size: int, hidden_size: int):
           """Parameters named and shaped exactly like nn.RNNCell's:
           weight_ih (H, input_size), weight_hh (H, H), bias_ih (H,), bias_hh (H,)."""
       def forward(self, x: Tensor, h: Tensor) -> Tensor:
           """x: (B, input_size), h: (B, H) ->
           tanh(x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh)"""
   ```
2. `test_vs_torch_cell`: build `nn.RNNCell(input_size, H)`, copy its parameters into yours via `load_state_dict`, run 5 *chained* steps on random inputs; assert `torch.allclose(h_mine, h_torch, atol=1e-6)` at every step.
3. The unrolled LM loss:
   ```python
   def rnn_lm_loss(cell, emb, head, x: Tensor, y: Tensor) -> Tensor:
       """x, y: (B, T). h starts at zeros(B, H). For each t:
       h = cell(emb(x[:, t]), h); logits = head(h); accumulate CE vs y[:, t].
       Return the mean over B*T predictions."""
   ```
   Train briefly on the book (T=32, hidden 128, ~500 steps, your AdamW) — just enough to watch train CE drop below your bigram's train CE; the serious training run waits for the LSTM.
4. The gradient-norm experiment, `week08/grad_norms.py`: for `T in [8, 16, 32, 64, 128]`, with a fresh-initialized model each time, run one forward+backward on one batch and record the total grad norm (a) raw and (b) after `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` — recompute the norm yourself after clipping. Plot raw norm vs T (log y-axis) to `week08/grad_norms.png`.
5. Record in a comment what you observed: the raw norm trends roughly geometrically with T — whether it grows or shrinks depends on your init scale; record your direction rather than assuming one.
**Verify — done when:** your cell matches torch to 1e-6 over chained steps; post-clip norm satisfies `assert norm_after <= 1.0 + 1e-6` for every T; the raw-norm plot shows a clear monotone trend in T (approximately — direction depends on init).
**If stuck:** the `nn.RNNCell` doc states the formula with both biases — diff your forward against it term by term; if `load_state_dict` fails, print both `state_dict().keys()`; remember `clip_grad_norm_` *returns the pre-clip norm* — don't assert on its return value.

## Day 8.6 — Deep build: LSTM from its equations (~3.5–4h)
- [ ] done
**Goal:** Implement LSTMCell from the six equations, verify to 1e-6 vs torch, then train the week's best LM and record the baseline Week 9 must beat.
**Learn:**
- *The six equations* (σ = sigmoid, ⊙ = elementwise; `W_*` act on input `x_t`, `U_*` on hidden `h_{t-1}`; every gate has shape `(B, H)`):
  1. `i_t = σ(W_i x_t + U_i h_{t-1} + b_i)` — input gate: how much new candidate to write.
  2. `f_t = σ(W_f x_t + U_f h_{t-1} + b_f)` — forget gate: how much old cell state to keep.
  3. `g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)` — candidate values to write.
  4. `o_t = σ(W_o x_t + U_o h_{t-1} + b_o)` — output gate: how much cell to expose.
  5. `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t` — cell update: **additive**.
  6. `h_t = o_t ⊙ tanh(c_t)` — hidden state (what the next step and layer see).
- *Cell vs hidden:* `c` is the long-term conveyor belt, updated by + and never squashed on the carry path; `h` is the gated, bounded read-out of it.
- *Why gating fixes vanishing:* equation 5 is addition, not matrix multiplication. The gradient along the cell path is scaled per step by `f_t` — which the network can hold near 1 — instead of by powers of `W_hh^T diag(tanh′)`. Yesterday's forced geometric decay becomes a controllable, near-identity path.
- *torch's packing:* `nn.LSTMCell` stacks all four gates: `weight_ih (4H, input)`, `weight_hh (4H, H)`, biases `(4H,)`, in gate order **i, f, g, o** (the docs state it). Respect this order or weight copying silently produces garbage.
**Read (30–45 min):**
- `nn.LSTMCell` docs — equations and parameter layout: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
- d2l.ai "Long Short-Term Memory (LSTM)": https://d2l.ai/chapter_recurrent-modern/lstm.html
- Understanding Deep Learning — the memory/gating discussion in the sequence-model material: https://udlbook.github.io/udlbook/
**Build:** `week08/lstm_lm.py` + tests:
1. ```python
   class MyLSTMCell(nn.Module):
       def __init__(self, input_size: int, hidden_size: int):
           """Parameters exactly like nn.LSTMCell: weight_ih (4H, input_size),
           weight_hh (4H, H), bias_ih (4H,), bias_hh (4H,)."""
       def forward(self, x: Tensor, state: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
           """state = (h, c). Pre-activations:
           z = x @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh    # (B, 4H)
           i, f, g, o = z.chunk(4, dim=-1)  # torch's i,f,g,o order
           then equations 1-6; return (h_new, c_new)."""
   ```
2. `test_vs_torch_lstmcell`: copy `nn.LSTMCell(32, 64)` weights into yours; run 8 chained steps on random inputs; assert BOTH `h` and `c` allclose at every step, `atol=1e-6`.
3. The trained LM — after the cell is verified, use `nn.LSTM` for speed (you've earned the abstraction):
   ```python
   class LSTMLM(nn.Module):
       def __init__(self, V: int, d: int = 64, h: int = 256, layers: int = 1):
           """nn.Embedding(V, d) -> nn.LSTM(d, h, num_layers=layers, batch_first=True)
           -> your tlib Linear(h, V) applied at every timestep."""
       def forward(self, x: Tensor) -> Tensor:
           """x: (B, T) -> (B, T, V) logits."""
   ```
4. Training budget (CPU): `block_size=128`, *non-overlapping* windows (start indices `0, 128, 256, ...` — ≈5–6k samples/epoch on this corpus), batch 64, 4 epochs, your Trainer + AdamW + warmup-cosine + grad clip 1.0 (≈20–35 min CPU). GPU variant: 2 layers, h=512, overlapping windows, 8+ epochs.
5. Evaluate val CE per character (flatten logits to `(B·T, V)` against flattened targets). Sample 500 chars by carrying `(h, c)` forward one char at a time — note generation is O(1) per char, unlike the MLP's recomputed window. Append the `lstm_h256` row to `experiments.csv` with `notes=WEEK9_BASELINE`.
**Verify — done when:** the cell test passes at 1e-6 for h AND c over 8 chained steps; `assert val_ce_lstm < val_ce_mlp` (relative to YOUR Day-4 number; the margin is approximately 0.1–0.3 nats at this budget); the sample shows mostly real words and Austen-flavored fragments (subjective — paste a snippet into the notes); the CSV row is written.
**If stuck:** gate-order bugs dominate — debug with `hidden_size=1` and hand-computed numbers against the doc's formula block; if training plateaus at MLP-level CE, check the targets are shifted by exactly one and that you didn't `.detach()` the embedding output.

## Day 8.7 — Review, quiz & redo-cold (~2h)
- [ ] done
**Goal:** Consolidate; prove the week's core equations and equivalences live in your head, not just your files.
**Learn:**
- *The week in one line:* every model was `P(next | context)` — only the context encoder changed: nothing → table row → MLP(window) → RNN state → gated LSTM state. Week 9 swaps in attention; the corpus, split, vocab, and metric stay fixed so the numbers stay comparable.
- *Your CSV is the artifact:* a strictly improving val-CE column is evidence the ideas work — measured by you, on one corpus, under stated budgets.
**Read (20 min):** your `experiments.csv` and each day's code, top to bottom; then reread Bengio §1's distributed-representation argument now that you've watched it work.
**Build:**
1. Redo-cold drills (no peeking; ~15 min each):
   - Write all six LSTM equations from memory, with gate order and which state is additive; diff against Day 6.
   - Re-derive `ppl = exp(CE)` from "CE = mean −log P", and explain in two sentences why the uniform model's CE is ln(V); verify numerically for your V.
   - Re-implement `SequenceDataset.__getitem__` and `__len__` from memory; run the Day-1 tests against it.
   - Re-implement `MyRNNCell.forward` from memory; run the Day-5 allclose test.
   - In ≤5 sentences, explain why text splits must be positional — written for a reviewer who shuffled before splitting.
2. Sanity sweep: run the whole week in one shot — `pytest week08/ tlib/` — and confirm green.
3. Self-quiz — answer all 12 cold, then check the Answers section:
   1. Your vocab has V=81. What are the CE (nats) and perplexity of the uniform model?
   2. Why is `F.embedding(ids, W)` exactly `one_hot(ids) @ W`, and why is the lookup implementation preferred?
   3. Why does shuffling before the train/val split leak for LM data but not for CIFAR images?
   4. As λ→∞ in additive smoothing, what does each row of P converge to, and what is the CE limit?
   5. Why must the gradient-trained bigram (wd=0) match the count bigram's train CE to high precision?
   6. A k=8 context as a count table over V=81 needs how many rows (expression)? What is Bengio's fix?
   7. State the vanilla RNN update. Which factor in its backward pass causes geometric growth/decay over T?
   8. Gradient clipping fixes which of the two gradient pathologies — and why not the other?
   9. Which LSTM equation is the "gradient highway", and what per-step factor scales gradients along it?
   10. What is the difference in role between `c_t` and `h_t`?
   11. Why is sampling from the LSTM O(1) per character while the MLP-LM must recompute its full window?
   12. What number from this week must Week 9's transformer beat, and what conditions make that comparison fair?
**Verify — done when:** drills match your tested implementations; the full-week pytest is green; quiz self-scored ≥10/12 (each answer cites the day to revisit).
**If stuck:** your own test files are the ground truth; the cited day's Read links are the fallback.

---

## Answers (Day 8.7 quiz)
1. CE = ln(81) ≈ 4.394 nats; ppl = exp(CE) = 81. Uniform perplexity always equals V. (D8.2)
2. A one-hot row times a matrix selects exactly one row of the matrix; the lookup returns that row directly without materializing the (…, V) one-hot or doing V multiplications — identical math, O(D) instead of O(V·D) per token, and gradient flows only into the used rows. (D8.1)
3. Adjacent LM windows overlap in all but one character, so a shuffled "val" window's content also appears inside train windows — near-duplicate leakage that inflates val scores. CIFAR images are independent samples; shuffling them creates no overlap between train and test items. (D8.1)
4. Each row converges to uniform (1/V everywhere); CE converges to ln(V). Your `lam=1e9` test asserted exactly this. (D8.2)
5. They are the same model class — one probability row per context char — fit to the same objective, and counting IS the closed-form maximizer of train likelihood; a converged gradient fit of the same class must therefore approach the same train CE. (D8.3)
6. V^k = 81^8 rows (≈1.8×10^15) — exponential in context length. Bengio: embed each context char into d dimensions, concatenate (k·d inputs), and compute the distribution with an MLP — parameters linear in k. (D8.4)
7. `h_t = tanh(W_ih x_t + b_ih + W_hh h_{t-1} + b_hh)`. Backward multiplies by `W_hh^T · diag(tanh′)` once per unrolled step; repeated multiplication by roughly the same matrix yields geometric explosion or decay in T. (D8.5)
8. Explosion only: clipping rescales an oversized gradient to a bounded norm so a single step can't destroy the weights. A vanished gradient is already ≈0 — rescaling a tiny vector cannot recover information that was multiplied away; the fix must be architectural (gating). (D8.5/8.6)
9. Equation 5, `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`. Along the cell path the per-step gradient factor is the elementwise `f_t`, which the network can hold near 1 — no repeated matrix powers, no forced decay. (D8.6)
10. `c_t` is long-term memory: updated additively, never squashed on the carry path. `h_t = o_t ⊙ tanh(c_t)` is the gated, bounded read-out used as the step's output and as the next step's recurrent input. (D8.6)
11. The LSTM's `(h, c)` already summarizes the entire past, so generating one more char costs one cell step; the MLP is stateless — its only context is the explicit k-char window, re-fed in full for every generated char. (D8.6)
12. The LSTM's val CE / ppl recorded in `experiments.csv` (`WEEK9_BASELINE`). Fair only on the same corpus, same positional split, same char vocab, and the same per-character val-CE evaluation, with a comparable training budget noted alongside. (D8.6)
