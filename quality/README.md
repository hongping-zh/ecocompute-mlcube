# The quality probe's evaluation text

`eval_text.txt` is the fixed text the perplexity probe scores. It is vendored rather
than downloaded so a run needs no network and no `datasets` dependency, and so every
contributor scores **the same bytes**.

```
sha256  22ac091a6383740d30f8e41ae144032c873c772db5e1c901112c1c330fdc5504
size    60600 bytes, ~10.6k words
```

Two public-domain excerpts, concatenated, ~30 KB each:

1. Lewis Carroll, *Alice's Adventures in Wonderland* (1865) — narrative and dialogue.
   [Project Gutenberg ebook 11](https://www.gutenberg.org/ebooks/11), from
   "Alice was beginning to get very tired".
2. Charles Darwin, *On the Origin of Species*, 1st ed. (1859) — expository prose.
   [Project Gutenberg ebook 1228](https://www.gutenberg.org/ebooks/1228), from
   "When we look to the individuals of the same variety".

Gutenberg's own header, footer and licence text are stripped: those are the parts
Project Gutenberg licenses. The works themselves are in the public domain. Illustration
markers and Gutenberg's `_underscore_` emphasis are removed, trailing whitespace and runs
of blank lines are normalised.

## What the number means, and what it does not

The probe reports the perplexity of one model at one precision on this text, and — when
the run also measures an FP16 baseline — the **relative** change `delta_vs_fp16_pct`.

- **The relative delta is the signal.** FP16 and the quantized model score the same bytes,
  in the same process, with the same tokenizer and window, so the difference isolates the
  quantization kernel.
- **The absolute perplexity is not comparable to published numbers.** Those are almost
  always WikiText-2 with a particular window; this is a different text. It is also almost
  certainly in the pretraining data of every model you will run, so it reads optimistically
  low. That does not affect the delta.
- **Perplexity is a proxy, not task quality.** A model can hold perplexity and still lose
  accuracy on reasoning or long-context tasks. Treat `delta_vs_fp16_pct` as a cheap
  screen — "did this quantization visibly damage the language model?" — not as a
  downstream-quality guarantee.

## Using your own text

```bash
--quality_text /path/to/your_heldout.txt
```

The report records the file's name, byte length and sha256 under
`quality.corpus`, so a run on a different text can never be silently compared with
one on this text.
