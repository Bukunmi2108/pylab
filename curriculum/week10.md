# Week 10 — Tokenization & Generation

Week 9 left you with a transformer that reads characters and generates by greedy argmax — correct, but two big pieces of the real LLM stack are missing: a learned subword vocabulary, and decoding that doesn't loop the same phrase forever or recompute the whole prefix per token. This week you build both: a byte-level BPE tokenizer (`tlib/tokenizer.py`, class `BPETokenizer`) trained on your book, the full sampling toolkit (temperature / top-k / top-p / repetition penalty), and a KV cache inside `DecoderLM`. These come now because each is a pure software-engineering layer around the fixed Week-9 model — algorithms, invariants, and round-trip properties you can test exactly, no gradients required until Day 6's retrain. You use these APIs daily; by Sunday you'll have implemented every one of them.

**Week outcome:** `tlib/tokenizer.py` with a train/encode/decode/save/load-complete `BPETokenizer` (round-trip-verified on the whole book plus emoji and Arabic text); `tlib/generate.py` with a property-tested `sample()`; `DecoderLM.forward_with_cache` whose greedy output is provably identical to the uncached model and measurably faster; a BPE-trained mini-LM with a per-character-normalized comparison against the char model; and `week10/chat_cli.py` — a streaming REPL you can actually talk to. Skill: you can explain and reimplement BPE, nucleus sampling, and KV caching cold.

---

## Day 10.1 — Bytes, Unicode, and the tokenization problem (~2h)
- [ ] done

**Goal:** Train your DecoderLM on raw UTF-8 bytes and learn how to compare language models that use different units, via bits-per-byte and per-character normalization.

**Learn:**
- **Code points vs bytes:** a Unicode string is a sequence of code points (~150k defined); UTF-8 serializes each to 1–4 bytes (ASCII→1, Arabic ع→2, 🙂→4). `"é"` can even be one code point or two (e + combining accent). Bytes are the unambiguous ground floor.
- **Byte-level basis ⇒ zero OOV, ever:** any text any human can type is *some* byte sequence, and there are only 256 byte values. A byte-level model (or byte-level BPE) can never hit an unknown token. Char vocabs can't promise this — your Week-8 `CharVocab` would choke on the first emoji not in the book.
- **The tradeoff axis is sequence length vs vocab size:** bytes → tiny vocab (256), long sequences (1 token per byte); word vocab → short sequences, huge vocab + OOV problem. Subword (BPE) sits between: frequent strings become single tokens, rare strings fall back toward bytes. GPT-style models use *byte-level BPE*: BPE merges learned on top of the 256 byte alphabet — Days 2–3 build exactly this.
- **CE per *what*?** Your Week-8/9 numbers are nats *per character*. A byte model gives nats *per byte*, a BPE model nats *per token* — these are different units and NOT comparable directly. The bridge: total nats over a text is unit-independent, so convert: per-char nats = per-unit nats × (n_units / n_chars). Standard reporting unit: bits-per-byte = (nats per byte)/ln 2.
- **Why this matters practically:** "my BPE model has lower loss" is usually just a unit change — each BPE token carries several characters of information. Day 6's comparison hinges on today's conversion.

**Read (30–45 min):**
- Sennrich, Haddow & Birch 2015, "Neural Machine Translation of Rare Words with Subword Units" — §1–3.2 (the BPE algorithm and motivation): https://arxiv.org/abs/1508.07909
- HuggingFace LLM course, "Byte-Pair Encoding tokenization" (read the text + worked example; skip the `transformers` code): https://huggingface.co/learn/llm-course/chapter6/5
- Python docs, "Unicode HOWTO" — Encodings + the UTF-8 section: https://docs.python.org/3/howto/unicode.html

**Build:** `week10/bytes_explore.py`.

1. Exploration prints (top of the script): for `s = "Attention! مرحبا 🙂"`, print `len(s)` (code points), `len(s.encode("utf-8"))` (bytes), and the per-character byte counts `[(ch, len(ch.encode('utf-8'))) for ch in s]`. Compute and print the book's `bytes_per_char = len(text.encode('utf-8')) / len(text)` (≈1.0 for an English Gutenberg book — that's the punchline: today's distinction barely shows in English, which is exactly why people forget it exists).
2. Byte dataset: 
   ```python
   def byte_dataset(text: str, block_size: int) -> "SequenceDataset":
       """data = torch.tensor(list(text.encode('utf-8')), dtype=torch.long)  # values 0..255
       Reuse Week-8 SequenceDataset over this id tensor; vocab size is 256."""
   ```
3. Equal-compute mini-comparison: train two `DecoderLM(d_model=128, n_heads=4, n_layers=4, block_size=128, dropout=0.1)` for **the same 800 steps, batch 32** (same seed, your Week-9 train script as a function): (a) char-level (Week-8 `CharVocab`, V≈80), (b) byte-level (V=256). Record final val CE of each — note they will be numerically close here *because* bytes/char ≈ 1 for this corpus, but they are still different units in principle.
4. The conversion utilities (these go in `tlib/text.py` — Day 6 reuses them):
   ```python
   def per_char_ce(ce_per_unit: float, n_units: int, n_chars: int) -> float:
       """Convert mean CE in nats-per-unit (unit = byte or token) to nats-per-character.
       Total nats = ce_per_unit * n_units; spread over characters: * n_units / n_chars.
       n_units and n_chars MUST be measured on the SAME text (use the val split)."""
       return ce_per_unit * n_units / n_chars

   def bits_per_byte(ce_per_byte_nats: float) -> float:
       """nats/byte -> bits/byte: divide by ln(2)."""
       return ce_per_byte_nats / math.log(2)
   ```
   Apply them: for the val text, `n_chars = len(val_text)`, byte `n_units = len(val_text.encode('utf-8'))`. Print a 3-line table: char-model per-char CE; byte-model per-char CE (converted); byte-model bits-per-byte.

**Verify — done when:**
- Unit tests for the converters: `per_char_ce(2.0, 100, 100) == 2.0`; `per_char_ce(1.0, 150, 100) == 1.5`; `abs(bits_per_byte(math.log(2)) - 1.0) < 1e-12` (CE = ln 2 nats is exactly 1 bit by definition — mathematically guaranteed).
- The exploration prints show 🙂 = 4 bytes, each Arabic letter = 2 bytes, ASCII = 1.
- Both mini-models trained to completion and per-char-normalized numbers printed side by side (values are stochastic — record, don't assert; expect same ballpark for this near-ASCII corpus).
- One-paragraph note in `week10/LOG.md`: why raw per-unit CEs aren't comparable, in your own words.

**If stuck:** Sennrich §3.2 for the compression view; the Unicode HOWTO for byte/code-point confusion; your Week-8 `SequenceDataset` tests.

---

## Day 10.2 — BPE part 1: training merges (~2.5h)
- [ ] done

**Goal:** Implement BPE training — repeatedly merge the most frequent adjacent pair — and verify it exactly on a hand-worked micro-example.

**Learn:**
- **The algorithm is three lines:** start from the 256 byte ids; count all adjacent pairs in the corpus; replace every occurrence of the most frequent pair with a fresh id; repeat until vocab_size. Each merge is one learned "rule".
- **The merges list IS the tokenizer.** Training produces an ordered list of (pair → new_id) rules. Encoding (Day 3) just replays them in order on new text; the byte strings in `vocab` are derivable from merges. Ship the merges, you've shipped the tokenizer.
- **Order matters, so ties need a rule.** Two pairs can have equal counts; whichever you merge first changes everything downstream. Any deterministic rule works — ours: **lowest (id1, id2) tuple wins** (Python tuple comparison). State your tie-break or your tokenizer isn't reproducible.
- **Greedy, not optimal:** BPE never revisits a merge. It's a compression heuristic (it began life as a 1994 compression algorithm), not a probabilistic model — and it works embarrassingly well.

**Read (30–45 min):**
- Sennrich 2015 §3.2 — Algorithm 1 is the whole trainer in 10 lines of pseudocode (theirs is word-internal; ours runs on the raw byte stream, which is the GPT-2-style simplification): https://arxiv.org/abs/1508.07909
- GPT-2 paper §2.2 "Input Representation" — why byte-level BPE and what they had to work around: https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf

**Build:** create `tlib/tokenizer.py`; tests in `week10/test_bpe.py`.

1. ```python
   class BPETokenizer:
       """Byte-level BPE. After train():
       merges: dict[tuple[int, int], int]   # (id_a, id_b) -> merged id; insertion order = training order
       vocab:  dict[int, bytes]             # id -> the bytes it expands to; ids 0..255 preexist
       """
       def train(self, text: str, vocab_size: int) -> None:
           """Learn vocab_size - 256 merges from text. Requires vocab_size >= 256.
           ids = list(text.encode('utf-8'))
           for i in range(vocab_size - 256):
               counts = get_pair_counts(ids)
               if not counts: break                       # ran out of pairs
               best = max-count pair, ties -> smallest pair tuple
               new_id = 256 + i
               ids = merge_pair(ids, best, new_id)
               merges[best] = new_id
               vocab[new_id] = vocab[best[0]] + vocab[best[1]]"""
   ```
2. Helpers (module-level, pure functions — easiest to test):
   ```python
   def get_pair_counts(ids: list[int]) -> dict[tuple[int, int], int]:
       """Counts of each adjacent pair, one left-to-right pass.
       Overlaps count: get_pair_counts([97, 97, 97]) == {(97, 97): 2}."""

   def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
       """Replace non-overlapping occurrences of pair with new_id, scanning left to
       right: if ids[i:i+2] == pair, emit new_id and jump i += 2; else emit ids[i], i += 1.
       So [97, 97, 97] with pair (97,97) -> [new_id, 97]  (leftmost wins, no overlap)."""
   ```
   Deterministic best-pair selection in one line: `best = min(counts, key=lambda p: (-counts[p], p))` — max count, then smallest pair.
3. **Worked micro-example — put this in the module docstring and check every step by hand.** Train on `"aaabdaaabac"` (bytes: a=97, b=98, c=99, d=100) with `vocab_size=259` (3 merges):
   - Start: `[97,97,97,98,100,97,97,97,98,97,99]`. Pair counts: (97,97)→4 *(overlaps count: two in each "aaa")*, (97,98)→2, (98,100)→1, (100,97)→1, (98,97)→1, (97,99)→1. Best: (97,97) → **256** ("aa"). After merge (left-to-right, non-overlapping — "aaa" becomes [256, 97]): `[256,97,98,100,256,97,98,97,99]`.
   - Counts now: (256,97)→2, (97,98)→2, (98,100)→1, (100,256)→1, (98,97)→1, (97,99)→1. **Tie** between (256,97) and (97,98); smallest tuple is (97,98) → **257** ("ab"): `[256,257,100,256,257,97,99]`.
   - Counts: (256,257)→2, rest 1. Merge (256,257) → **258** ("aaab"): final `[258,100,258,97,99]`.

**Verify — done when** (`pytest week10/test_bpe.py`):
- Helper unit tests, including the overlap cases spelled out in the docstrings above, pass.
- Micro-example, exactly (deterministic given the tie-break — assert all of it):
  ```python
  tok = BPETokenizer(); tok.train("aaabdaaabac", 259)
  assert list(tok.merges.items()) == [((97,97),256), ((97,98),257), ((256,257),258)]
  assert tok.vocab[258] == b"aaab"
  ```
- `train(text, 256)` learns zero merges; `vocab_size=255` raises.
- Train on the full book with `vocab_size=512` (one-time cost ~a few minutes with this O(n) -per-merge implementation; print progress every 32 merges). Print the 20 longest vocab entries — expect common English fragments (" the", "ing", …). Save nothing yet; persistence is Day 3.

**If stuck:** Sennrich Algorithm 1; re-walk the micro-example with pencil — every bug shows up in step 1 or the tie.

---

## Day 10.3 — BPE part 2: encode, decode, persistence (~2.5h)
- [ ] done

**Goal:** Complete the tokenizer with encode/decode/save/load and prove the invariants: lossless round-trip on anything, idempotent persistence, real compression.

**Learn:**
- **Encoding replays training history:** on new text, repeatedly find which adjacent pair *present in the ids* was learned EARLIEST (lowest merge index) and apply it; stop when no present pair is in `merges`. Applying merges out of order produces different tokens than training saw — the classic subtle tokenizer bug.
- **Decoding is trivial by design:** concatenate `vocab[id]` byte strings, then `bytes.decode('utf-8', errors='replace')`. All the complexity lives in encode; decode is a lookup.
- **Why `errors='replace'`:** every id expands to valid *bytes*, but an arbitrary id *slice* (e.g. streamed generation, Day 6) can end mid-multibyte-character. `errors='replace'` yields U+FFFD (�) instead of raising. Full encode→decode round-trips are always whole, so they're exact anyway.
- **The three invariants worth testing:** (1) `decode(encode(s)) == s` for ALL valid Unicode `s` — guaranteed because merges only ever group bytes, never drop them; (2) save→load→encode is bit-identical; (3) tokens-per-char < 1 on in-domain text — otherwise why bother.

**Read (20–30 min):**
- HF LLM course BPE page, the "Tokenization" (applying merges) section: https://huggingface.co/learn/llm-course/chapter6/5
- Python `bytes.decode` / `bytes.hex` / `bytes.fromhex` docs (your persistence tools): https://docs.python.org/3/library/stdtypes.html#bytes.hex

**Build:** extend `tlib/tokenizer.py`; tests in `week10/test_bpe_roundtrip.py`.

1. ```python
   def encode(self, text: str) -> list[int]:
       """ids = list(text.encode('utf-8'))
       while len(ids) >= 2:
           counts = get_pair_counts(ids)
           pair = min(counts, key=lambda p: self.merges.get(p, float('inf')))
           if pair not in self.merges: break     # no learnable pair remains
           ids = merge_pair(ids, pair, self.merges[pair])
       return ids"""

   def decode(self, ids: list[int]) -> str:
       """return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')"""

   def save(self, path: str) -> None:
       """JSON: {"merges": [[a, b, new_id], ...]} in training order. bytes need no
       hex here — merges are pure ints; vocab is REBUILT from merges on load, which
       guarantees save/load can't drift. (If you also persist vocab for debugging,
       hex-encode the bytes: {"vocab": {str(id): vocab[id].hex()}}.)"""

   @classmethod
   def load(cls, path: str) -> "BPETokenizer":
       """Rebuild merges dict (order preserved) and vocab via
       vocab[new] = vocab[a] + vocab[b] replayed in order."""
   ```
2. Compression report `week10/compression.py`: with the Day-2 book tokenizer (vocab 512), print `len(encode(book)) / len(book)` (tokens per char) and the same for vocab 256 (no merges, = bytes_per_char ≥ 1.0).

**Verify — done when:**
- Round-trip on the **entire book**: `tok.decode(tok.encode(book)) == book` (exact string equality).
- Round-trip on adversarial strings — assert equality for each:
  ```python
  cases = ["", "🙂", "🙂🙂🙂", "مرحبا بالعالم",            # empty, emoji, Arabic
           "naïve café — ☕", "á", "ع🙂e\nmixed لغة text",
           "￿", "퟿"]                   # boundary code points around the surrogate range
  ```
  (Note: actual lone surrogates like `"\ud800"` cannot be UTF-8-encoded — `str.encode` raises `UnicodeEncodeError`. Add a test asserting `encode` raises on `"\ud800"` so you know the failure mode; the boundary cases above are the nearest *valid* neighbors.)
- Compression: `len(tok.encode(book)) / len(book) < 1.0` for vocab 512 — assert it (guaranteed in practice for 256 merges on megabytes of English; if it fails, your encode isn't applying merges).
- Persistence idempotence: `tok.save(p); tok2 = BPETokenizer.load(p); assert tok2.encode(sample) == tok.encode(sample)` for sample = first 10k chars of the book, AND `tok2.vocab == tok.vocab`.
- Order-matters regression test: encode a sentence with a deliberately *reversed* merge-priority (`max` instead of `min` over merge index) and assert the ids differ — proves the priority loop is load-bearing.

**If stuck:** HF course "Tokenization" section walks an encode by hand; print `(pair, merge_index)` per loop iteration on `"aaab"` and compare with Day 2's trace.

---

## Day 10.4 — Sampling: temperature, top-k, nucleus, repetition penalty (~2.5h)
- [ ] done

**Goal:** Replace greedy generation with a property-tested `sample()` implementing the full standard decoding toolkit.

**Learn:**
- **Greedy's pathology:** always taking argmax makes generation deterministic and self-repetitive — once a loop ("the the the", or a repeated phrase) starts, nothing breaks it. Holtzman et al. call this neural text degeneration; you've seen it in your Day-9.5 samples.
- **Temperature = logit scaling:** sample from `softmax(logits / T)`. T<1 sharpens (T→0 recovers argmax), T>1 flattens (T→∞ → uniform over the vocab). It rescales *log*-probabilities, so relative order never changes.
- **Top-k:** keep the k highest logits, set the rest to -inf, renormalize. Crude but effective — the tail of a 50k-token distribution is individually-unlikely junk that collectively holds real probability mass.
- **Top-p (nucleus):** keep the *smallest* set of tokens whose cumulative probability ≥ p. Adaptive where top-k is fixed: a confident distribution keeps 2 tokens, a flat one keeps 500. (Holtzman 2019's fix for both greedy degeneration and tail-sampling incoherence.)
- **Repetition penalty (CTRL-style):** before anything else, for every token id already in the context, divide its logit by penalty r>1 if positive, multiply by r if negative — both directions make the token *less* likely. It's a hack (no probabilistic story), but a ubiquitous one.
- **Order of operations matters and must be pinned:** penalty → temperature → top-k → top-p → softmax → sample. Different orders give different distributions; pick this one and test it.

**Read (30–45 min):**
- Holtzman et al. 2019, "The Curious Case of Neural Text Degeneration" — §1, §3 (degeneration evidence), §4 (nucleus definition): https://arxiv.org/abs/1904.09751
- Keskar et al. 2019, "CTRL", §4.1 for the repetition-penalty equation: https://arxiv.org/abs/1909.05858
- `torch.multinomial` and `torch.topk` docs: https://docs.pytorch.org/docs/stable/generated/torch.multinomial.html

**Build:** create `tlib/generate.py`; tests in `week10/test_sampling.py`.

1. The logit-transform helpers — pure functions on a (B, V) logits tensor, independently testable:
   ```python
   def apply_rep_penalty(logits: torch.Tensor, idx: torch.Tensor, penalty: float) -> torch.Tensor:
       """logits (B, V), idx (B, T) context ids. For each batch row, for each id present
       in that row: logit > 0 -> logit / penalty, logit <= 0 -> logit * penalty."""
   def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
       """All but the k largest logits per row -> -inf. k >= V is a no-op."""
   def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
       """Sort desc, softmax, cumsum. Keep the smallest prefix with cumsum >= p
       (i.e. drop tokens where the cumulative prob of *strictly better* tokens is
       already >= p); ALWAYS keep the top-1 token. Dropped -> -inf, unsorted order restored."""
   ```
2. The sampler:
   ```python
   @torch.no_grad()
   def sample(model, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
              top_k: int | None = None, top_p: float | None = None,
              rep_penalty: float = 1.0, greedy: bool = False,
              generator: torch.Generator | None = None) -> torch.Tensor:
       """idx (B, T0) -> (B, T0 + max_new_tokens). Per step: crop idx to block_size;
       logits = model(idx_cond)[:, -1, :]  # (B, V)
       1. if rep_penalty != 1.0: apply_rep_penalty
       2. if greedy or temperature == 0: next = argmax; skip 3-6
       3. logits = logits / temperature
       4. if top_k: apply_top_k     5. if top_p: apply_top_p
       6. probs = softmax(logits, -1); next = torch.multinomial(probs, 1, generator=generator)
       cat next (B, 1) onto idx."""
   ```
   Pass `generator` through for reproducible tests. Eval mode is the caller's job — assert `not model.training`.
3. Quick qualitative pass: load the Day-9.5 char checkpoint; print samples at T ∈ {0 (greedy), 0.5, 0.8, 1.0, 1.5} and (T=0.9, top_p=0.9). Paste the greedy one and your favorite into LOG.md.

**Verify — done when:**
- Temperature limits (numeric, on a fixed `logits = torch.tensor([[2.0, 1.0, 0.5, -1.0]])`):
  T = 0.01 → `softmax(logits/T)` has max prob > 0.999 (≈ argmax); T = 1e6 → `(probs - 0.25).abs().max() < 1e-4` (≈ uniform). Both asserted.
- Top-k support: for random (3, 50) logits and k=5, `(apply_top_k(l, 5).softmax(-1) > 0).sum(-1)` equals 5 in every row.
- Top-p support on a crafted distribution: `logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))`, p = 0.8 → surviving set is exactly {0, 1} (0.5 + 0.3 ≥ 0.8 and no smaller prefix reaches it); p = 0.79 → also {0, 1}; p = 0.5 → {0}; p = 1.0 → all 4. Assert each support set.
- Always-keep-one: a near-one-hot distribution with p = 0.01 still keeps the top token (no empty support / NaN).
- Rep penalty direction: token in context with positive logit gets *less* probable; with negative logit also less probable (assert both on hand-built logits).
- Greedy determinism: two `sample(..., greedy=True)` calls from the same prompt return `torch.equal` sequences, and match Day-9.4's `generate`.
- Seeded sampling reproducible: same `torch.Generator().manual_seed(7)` twice → identical output.

**If stuck:** Holtzman §4 for the exact nucleus set definition (your ≥ vs > bug lives there); CTRL §4.1 for the penalty's two-sided form; check `multinomial` rows sum to 1 after your -inf masking.

---

## Day 10.5 — KV cache: generation without recompute (~2.5h)
- [ ] done

**Goal:** Add `forward_with_cache` to `DecoderLM` and prove — before timing anything — that cached greedy generation is token-identical to uncached.

**Learn:**
- **The waste:** naive generation re-runs the full forward on all T tokens to produce one new token, then T+1, … Per-token cost grows linearly, total O(T²) attention work — and everything about positions 0..T−1 was already computed last step.
- **The fix:** in causal attention, K and V at past positions never change (they don't depend on the future). Cache each layer's K and V; per step, compute q/k/v for **only the new token**, append k,v to the cache, and attend with the single new query over the full cached keys/values.
- **Cache shape:** per layer, K and V of shape (B, H, T_cache, hs), grown along dim 2 each step. For your D9.5 model: 4 layers × 2 tensors × (1, 4, T, 32).
- **No mask needed for a 1-token query:** the new token at position T may attend to all of 0..T — the cache *is* exactly its allowed past. (Masks return if you batch-prefill multi-token chunks; we sidestep that by prefilling with cache built token-free via one ordinary forward — see Build.)
- **Positions must keep counting:** the new token's positional embedding index is T_cache, not 0. Forgetting this is the classic cache bug — outputs drift after the first generated token, which is precisely what the identity test catches.

**Read (30–45 min):**
- HF Transformers "Cache strategies" docs — the intro + Default cache section; map DynamicCache onto what you're about to build: https://huggingface.co/docs/transformers/kv_cache
- Attention Is All You Need §3.2.3 (decoder self-attention masking — why cached past is legal): https://arxiv.org/abs/1706.03762

**Build:** extend `tlib/transformer.py`; tests in `week10/test_kv_cache.py`; timing in `week10/bench_cache.py`.

1. Cache spec (pin it exactly): `cache: list[dict[str, torch.Tensor]]` — one dict per layer, keys `"k"` and `"v"`, each (B, H, T_cache, hs). Empty cache = `[{"k": empty, "v": empty} for _ in layers]` with `torch.empty(B, H, 0, hs)` so `torch.cat` works uniformly from step 0.
2. `MultiHeadAttention.forward(self, x, mask=None, kv_cache: dict | None = None)`: compute q, k_new, v_new from x (T_new tokens, the dance as usual → (B, H, T_new, hs)); if `kv_cache` is not None: `k = torch.cat([kv_cache["k"], k_new], dim=2)`, same for v, write both back into the dict (mutate in place); `y = sdpa(q, k, v, mask)` — note q has T_new rows, k/v have T_cache+T_new: scores (B, H, T_new, T_cache+T_new), output (B, H, T_new, hs). `Block.forward` gains and forwards the same `kv_cache` arg.
3. ```python
   def forward_with_cache(self, idx: torch.Tensor,
                          cache: list[dict[str, torch.Tensor]] | None = None,
                          ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
       """idx (B, T_new); cache as specced or None (None -> fresh empty cache).
       offset = cache[0]['k'].shape[2]; assert offset + T_new <= block_size.
       Pos-emb rows: arange(offset, offset + T_new).
       Mask: None when T_new == 1; for prefill (T_new > 1 with empty cache) use the
       ordinary causal mask. Returns (logits (B, T_new, V), cache)."""
   ```
4. `generate_cached(model, idx, max_new_tokens, **sample_kwargs)` in `tlib/generate.py`: one prefill call on the prompt (T_new = prompt length, empty cache), then a loop of 1-token calls feeding only the just-sampled token. Reuse the Day-4 logit pipeline so cached and uncached share sampling code.
5. `week10/bench_cache.py`: load the D9.5 checkpoint; greedy-generate 200 tokens from a 16-token prompt, cached and uncached; `time.perf_counter` around each (eval, no_grad, single run is fine but do one warmup pass); print both times and the ratio.

**Verify — done when (correctness FIRST, timing second):**
- **The canonical identity test:** from the same 16-token prompt, `generate_cached(..., greedy=True)` and Day-4 uncached greedy `sample(..., greedy=True)` produce sequences with `torch.equal(a, b)` — exact token match for 100 new tokens. Run it for B=1 and B=2.
- Stepwise logits: walk 20 steps; at each, compare the cached step's logits to `model(full_idx)[:, -1, :]` from a full uncached forward — `torch.allclose(atol=1e-5)` every step (tiny float drift from different reduction orders is expected; token-level equality above must still be exact for greedy).
- Cache shape invariant: after prefill of 16 + 10 steps, every layer's `cache[i]["k"].shape == (B, 4, 26, 32)`.
- Position regression test: temporarily hardcode offset = 0 and confirm the identity test FAILS (then revert) — proves the test has teeth.
- `bench_cache.py` runs and reports both times. Cached should be visibly faster for 200 tokens (approximately — CPU-dependent; record YOUR measured ratio in LOG.md, no target number).

**If stuck:** HF "Cache strategies" intro for the conceptual picture; your own Day-1 sdpa supports rectangular T_q ≠ T_k already — re-read its docstring; if logits drift starting at step 2, it's the position offset.

---

## Day 10.6 — End-to-end mini-LM: BPE model + chat CLI (~3.5–4h, deep build)
- [ ] done

**Goal:** Retrain DecoderLM on your own BPE tokens, compare it to the char model in the only fair unit (per character), and wrap checkpoint + tokenizer + sampler + cache into a streaming chat CLI.

**Learn:**
- **What changing the tokenizer changes:** with vocab 512, block_size 128 now spans ~128/0.55 ≈ 230+ characters of context (using your measured tokens-per-char from Day 3) — the model sees more text per window at the same compute. The embedding/head grow (V: 80 → 512); per-token CE rises (each token carries more information); per-CHAR CE is the fair scorecard. That unit discipline is today's lesson.
- **Streaming + UTF-8:** printing token-by-token means printing byte strings that can end mid-character (a 4-byte 🙂 split across two tokens). Buffer bytes; only print the longest decodable prefix.
- **This is the artifact:** model you built (W9), tokenizer you built (D10.2–3), sampler (D10.4), cache (D10.5), trainer (W5) — one working product, no library code on the critical path except torch ops.

**Build:**

1. `week10/train_bpe_lm.py`: train `BPETokenizer` on the *train split only* (vocab 512; save to `week10/bpe512.json` — training the tokenizer on val text is mild leakage; note this in LOG.md). Encode train/val splits once and cache the id tensors to disk (`torch.save`) — re-encoding the book every run wastes minutes. Model: `DecoderLM(512, 128, 4, 4, 128, dropout=0.1)` (~1.4M params — the bigger embedding; assert vs your formula). Train with the D9.5 recipe, 3000 steps (reduce if needed; log it).
2. The comparison (the heart of the day), printed and written to LOG.md as a table:
   - char model (D9.5): per-char val CE = its val CE (already per char).
   - BPE model: per-char val CE = `per_char_ce(val_ce_tokens, n_units=len(tok.encode(val_text)), n_chars=len(val_text))` — Day-1 function, same val text.
   - Also report both as bits-per-char (÷ ln 2). Record which won — either outcome is defensible at this tiny scale (subword wins on context span; char wins on easier per-step targets). Don't assert a winner; DO assert both numbers are finite and the BPE model's *raw* per-token CE > its per-char CE (guaranteed since tokens/char < 1 from Day 3).
3. `week10/chat_cli.py` — argparse: `--ckpt`, `--tokenizer`, `--temperature 0.8`, `--top-k`, `--top-p 0.95`, `--rep-penalty 1.0`, `--max-new 256`, `--seed`. Behavior:
   - Load model (eval) + `BPETokenizer.load`; REPL: `prompt = input("you> ")`, empty line exits.
   - Encode prompt (prepend a newline so the model starts mid-text naturally); prefill the KV cache; stream: per generated token, append `tok.vocab[id]` to a byte buffer and print the longest decodable prefix:
     ```python
     def flush_printable(buf: bytes) -> tuple[str, bytes]:
         """Return (text to print, leftover bytes). Try buf.decode('utf-8'); on
         UnicodeDecodeError at the TAIL, retry buf[:-1], buf[:-2], buf[:-3]
         (a UTF-8 char is <= 4 bytes); decode what's whole, keep the rest."""
     ```
     `print(text, end="", flush=True)`.
   - Stop at `--max-new` or when prompt+generated hits block_size (print `[ctx full]`).
4. Use it. Have three conversations at different temperatures; paste the best and worst into LOG.md with settings. (It's a 1.4M-param model trained on one book — calibrate expectations: it should produce book-flavored English-ish prose, not answers.)

**Verify — done when:**
- `train_bpe_lm.py` completes; init CE ≈ ln(512) ≈ 6.238 ± 0.1 (assert — the Day-9.4 guarantee, new vocab).
- The LOG.md table has all four numbers (2 models × nats-per-char, bits-per-char) computed on the same val text, plus tokens/char used in the conversion.
- `flush_printable` unit tests: a 4-byte emoji split 2+2 across calls prints nothing then the emoji; pure-ASCII passes through unchanged; leftover is always < 4 bytes.
- `python week10/chat_cli.py --ckpt ... --tokenizer ... --temperature 0.8` runs a real conversation; `--seed 7` twice gives identical output; streaming visibly streams (tokens appear incrementally).
- Generation inside the CLI goes through the KV-cache path (assert by instrumenting once or just grep your call sites).

**If stuck:** Day-5 identity test if CLI output looks corrupted (cache bug, not CLI bug); Day-3 round-trip tests if you see � on whole outputs (encode bug) vs only transiently at stream boundaries (expected, fixed by `flush_printable`).

---

## Day 10.7 — Review, quiz, redo-cold (~2h)
- [ ] done

**Goal:** Lock in the three algorithms of the week by rebuilding them from memory and self-testing.

**Redo-cold drills (closed book, blank file `week10/redo.py`):**
1. **(≤15 min)** BPE training loop from memory: `get_pair_counts`, `merge_pair`, the train loop with the tie-break. Must reproduce the Day-2 micro-example asserts on `"aaabdaaabac"`.
2. **(≤10 min)** `apply_top_p` from memory; must pass the Day-4 crafted-distribution support tests ({0.5, 0.3, 0.15, 0.05}, p ∈ {0.5, 0.8, 1.0}).
3. **(≤10 min)** In prose, no peeking: why does the KV cache change *nothing* about model outputs? (Target: causal K/V at past positions are functions of past tokens only, so caching is memoization of identical values; the new query attends over exactly the same K/V it would see uncached; only the arithmetic schedule changes — hence exact greedy-token equality, allclose logits.) Then list the two bugs that break the identity (position offset; masking the 1-token query wrongly).
4. **(≤5 min)** Write `per_char_ce` and `bits_per_byte` from memory with their unit-test values.

**Self-quiz** (answers at the bottom of this file):
1. Why can a byte-level tokenizer never produce an unknown token, and what does it trade away?
2. A string has 100 code points and 140 UTF-8 bytes. Your byte LM scores 1.4 nats/byte. Per-char CE? Bits-per-byte?
3. State the BPE training loop in ≤4 lines, including the tie-break and where new vocab bytes come from.
4. In `merge_pair`, what happens to `[a, a, a]` when merging (a, a), and why must counting and merging agree on overlap handling?
5. Why must encode apply merges in training order (lowest merge index first)? Construct the failure if it doesn't.
6. Why is `errors='replace'` needed in decode, and in which situation does it actually fire?
7. Exact definitions: what set does top-k keep? Top-p? Why is top-p called "adaptive"?
8. Write the temperature-limit facts you asserted (T→0, T→∞) and why order of tokens never changes with T.
9. The pinned order of logit operations in `sample()` — list it.
10. KV cache: what is cached, what shape, what is computed fresh per step, and why is no causal mask needed for a single-token query?
11. Why did per-token CE go UP when you switched to BPE while the model possibly got *better* per character?
12. Your D10.5 measured speedup for 200 tokens was X — explain where the saved compute comes from in big-O terms.

**Foreshadow:** Week 11 profiles this exact model and generation loop (where do the milliseconds go? batching, compile, fused attention) and speeds it up; Week 12 pretrains a scaled-up DecoderLM on a real multi-document dataset using this same `BPETokenizer` code with a bigger vocab. Keep the checkpoints and `bpe512.json` — they're inputs, not souvenirs.

---

## Answers (Day 10.7 quiz)

1. Its base alphabet is all 256 byte values, and any Unicode string serializes to UTF-8 bytes, so every input decomposes into known symbols. Trade-off: sequences are longer (≥1 token per byte before merges), so the same block_size spans less text and attention does more work per character.
2. Per-char CE = 1.4 × 140/100 = **1.96 nats/char**. Bits-per-byte = 1.4 / ln 2 ≈ **2.02 bpb**.
3. `ids = bytes of text`; repeat vocab_size−256 times: count adjacent pairs; pick max count, ties → smallest (id1, id2) tuple; replace its non-overlapping left-to-right occurrences with new id 256+i; record `merges[pair] = new_id` and `vocab[new_id] = vocab[a] + vocab[b]`.
4. It becomes `[new_id, a]` — leftmost match consumes positions 0–1, position 2 can't pair with consumed bytes. Counting saw (a,a) twice (overlapping) but merging applies non-overlapping left-to-right; that's fine as a *ranking* heuristic, but the merge semantics must be one fixed rule or training and encoding produce different sequences for the same text.
5. Encoding must reproduce the token boundaries training produced, and later merges were learned on text *already transformed* by earlier merges (merge 258 = (256, 257) only exists after 256 and 257 fire). Failure: with merges aa→256, ab→257, aaab→258 learned in that order, encoding "aaab" by applying (97,98) first yields [97, 97, 257] and then (97,97)→[256, 257] — but training produced [258]; out-of-order encoding emits token sequences the model never saw at train time.
6. Each vocab entry is valid bytes, but an arbitrary *slice* of generated ids can end mid-multibyte character (e.g. half of a 4-byte emoji); decoding that slice raises without `errors='replace'`, which substitutes U+FFFD instead. It fires during streaming/truncated decodes; full encode→decode round-trips never need it.
7. Top-k: the k tokens with the highest logits (fixed-size set). Top-p: the smallest set of highest-probability tokens whose cumulative probability ≥ p (top-1 always kept). Adaptive because its size depends on the shape of the distribution — confident → tiny set, flat → large set; top-k keeps k either way.
8. T→0: softmax(logits/T) → one-hot at the argmax (asserted max prob > 0.999 at T=0.01). T→∞: logits/T → 0, softmax → uniform 1/V (asserted ≈ 0.25 each for V=4 at T=1e6). Dividing by T>0 is monotone on logits, so the ranking — and hence the argmax — never changes.
9. Repetition penalty → (greedy short-circuit) → temperature divide → top-k → top-p → softmax → multinomial sample.
10. Cached: every layer's K and V over all past positions, each (B, H, T_cache, hs), appended along dim 2. Fresh per step: the new token's embedding(+position T_cache), its q/k/v, attention of the single query over the full cache, MLP, logits. No mask needed because the cache contains *only* positions ≤ the new one — the allowed attention set and the available set coincide.
11. Each BPE token packs ~1/0.55 ≈ 1.8 characters, so the per-step prediction target carries more information — more nats per decision. Normalizing by characters (CE_token × tokens/char) divides that back out; only then are the models comparable. Higher per-token CE with lower per-char CE means the model got better at modeling *text* while solving a harder per-step problem.
12. Uncached: step t re-runs attention over t tokens → Σt = O(T²) attention work (and O(T²) total token-forwards through the MLP too). Cached: step t computes one token's q/k/v and one query-row of attention over t cached keys → O(T) MLP/projection work total plus O(T²) only in the (cheap, matmul-free-per-step) score row — in practice the model-forward recompute dominates, which is what your measured ratio reflects.
