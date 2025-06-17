# SpeculativeNanoGPT: Accelerated Inference with Speculative Decoding

## 🚀 Overview

SpeculativeNanoGPT implements **speculative decoding** for accelerating inference in NanoGPT-style language models. By using a smaller "draft model" to propose tokens and a larger "target model" to verify them, this technique speeds up text generation while maintaining quality.

## ✨ Features

- **Speculative Decoding**: Efficient draft-and-verify algorithm.
- **Dual Model Architecture**: 
  - 19M parameter **draft model** (fast).
  - 124M parameter **target model** (high-quality).
- **Performance Comparison**: Speed and time comparisons between standard and speculative decoding.
- **Customizable Parameters**: Supports `temperature` and `top_k` for diverse text generation.

## 🧠 Model Architecture

Uses a simplified GPT architecture with:

- **CausalSelfAttention**: Efficient multi-head attention.
- **MLP**: Two-layer feed-forward network with GELU.
- **GPT**: Stacks Transformer blocks, shared token embeddings, and a language modeling head.

Two configurations are provided: `GPTConfig124M` and `GPTConfig19M`.

## 💾 Pre-trained Models

Download the pre-trained models from Hugging Face:

- [124M target model](https://huggingface.co/fridayfringe/nanogpt_124M/tree/main)
- [19M draft model](https://huggingface.co/fridayfringe/nanogpt_124M/tree/main)

Place these `.pth` files in the root directory.


## Example Output
  ```bash
      Prompt: "love in the air"
      Normal Generation (124M):
      Speed: 7.18 tokens/sec
      Time: 13.36 seconds
      
      Speculative Decoding (19M -> 124M):
      Speed: 10.67 tokens/sec
      Time: 9.09 seconds
      Speedup: 1.48x
      Acceptance Rate: 77%
```
⚙️ Customization
   ```bash
      Adjust these parameters in the notebook:
      max_length: Max tokens to generate.
      temperature: Controls randomness.
      top_k: Filters top K tokens.
      num_draft_tokens: Number of tokens for the draft model.
```
🙏 Acknowledgements
```bash
    Andrej Karpathy’s NanoGPT: Base for the GPT implementation.
    Speculative Decoding Researchers: For pioneering the technique.
```
