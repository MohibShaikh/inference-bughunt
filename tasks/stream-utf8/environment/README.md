# tinyinfer

A small byte-level language model inference stack in PyTorch.

- `tinyinfer/model.py`: decoder-only transformer with a key/value cache.
- `tinyinfer/generate.py`: batched generation (left padding, KV cache, EOS handling, sliding window).
- `tinyinfer/sampling.py`: repetition penalty, top-k, top-p and the sampler.
- `tinyinfer/tokenizer.py`: byte tokenizer and streaming detokenizer.

Run the smoke test with `python -m pytest /app/tests`.
