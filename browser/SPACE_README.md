---
title: bekko-system-one in your browser
emoji: 🐂
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
fullWidth: true
short_description: Small models for local Yes/No, Choice, and Score decisions.
license: mit
models:
  - hotchpotch/bekko-system-one-v0-17m
  - hotchpotch/bekko-system-one-v0-68m
  - hotchpotch/bekko-system-one-v0-400m
tags:
  - on-device-ai
  - text-classification
  - onnx
  - webgpu
pinned: false
---

# bekko-system-one in your browser

Try small System One models that answer Yes/No questions, choose between options,
or assign scores. This static demo runs inference directly on your device with
ONNX Runtime Web. Your input text is not sent to an inference server.

## Try it

1. Choose a model. Start with **17M** for the smallest download.
2. Select **Noul (Yes/No)**, **Choice**, or **Score**, then choose an example.
3. Select **Run model**. The first run downloads the model to your browser.
4. Edit the question, context, and answer definitions to try your own inputs.
   On mobile, select **View or edit input** to open these fields.

Use **CPU** or **WebGPU**. WebGPU is selected when a usable adapter is available,
but support depends on your browser and device; unsupported operations may run
on CPU. If WebGPU fails, switch to CPU.

## Models and download sizes

| Model | Model download |
| --- | ---: |
| [bekko-system-one-v0-17m](https://huggingface.co/hotchpotch/bekko-system-one-v0-17m) | 29.0 MB |
| [bekko-system-one-v0-68m](https://huggingface.co/hotchpotch/bekko-system-one-v0-68m) | 196.3 MB |
| [bekko-system-one-v0-400m](https://huggingface.co/hotchpotch/bekko-system-one-v0-400m) | 1,426.2 MB (about 1.4 GB) |

Sizes are decimal MB for the ONNX model alone. Tokenizer and runtime files are
additional downloads. The 400M model requires substantial device memory.
Model files are downloaded from public Hugging Face repositories; no account or
access token is required. Initial use requires an internet connection.

## Works best on familiar tasks

bekko-system-one-v0 was trained on 100+ task-specific dataset subsets covering
classification, selection, and scoring. It often works best when your question
and context resemble those tasks. The demo examples were deliberately selected
to show cases it handles well, rather than general-purpose accuracy.

Generalization to unfamiliar instructions and domains remains limited. Even
simple questions can produce confidently wrong answers. These models are
intended for English input, and are not a replacement for Jev's broad
generalization.

This demo explores how small models can make useful, structured decisions
locally—a starting point for more capable on-device AI.

## Learn more

- [Release article](https://huggingface.co/blog/hotchpotch/bekko-system-one-v0-release/)
- [Browser source and development guide](https://github.com/hotchpotch/bekko-system-one/tree/main/browser)
- [Training and inference toolkit](https://github.com/hotchpotch/bekko-system-one)

The application code is MIT-licensed.
