# SpeculativeNanoGPT: Accelerated Inference with Speculative Decoding

## 🚀 Overview

This repository presents an implementation of **speculative decoding** applied to a NanoGPT-style language model. Speculative decoding is an advanced inference technique designed to significantly speed up the generation process of large language models (LLMs) by leveraging a smaller, faster "draft model" to propose tokens, which are then efficiently verified by a larger, more accurate "target model".

The core idea is to reduce the number of costly forward passes through the larger model. Instead of predicting one token at a time with the target model, the draft model quickly predicts several tokens, and the target model then validates these predictions in parallel. This often leads to substantial speedups while maintaining the quality of the generated output.

## ✨ Features

* **Speculative Decoding Implementation**: A clear and functional implementation of the draft-and-verify speculative decoding algorithm.
* **Dual Model Architecture**: Utilizes two distinct GPT models:
    * A **smaller 19M parameter model** as the high-speed "draft model".
    * A **larger 124M parameter model** as the high-quality "target model".
* **Performance Comparison**: Includes utilities to compare the generation speed (tokens/second) and time between standard (non-speculative) decoding and speculative decoding.
* **Customizable Generation Parameters**: Supports `temperature` and `top_k` sampling for diverse text generation.
* **Modular NanoGPT Codebase**: Built upon a clean and modular implementation of the GPT architecture, inspired by Andrej Karpathy's NanoGPT.

## 🧠 Model Architecture

The repository uses a simplified GPT architecture, consisting of:

* **`CausalSelfAttention`**: Implements multi-head self-attention with a causal mask, leveraging PyTorch's `F.scaled_dot_product_attention` for efficiency (Flash Attention).
* **`MLP`**: A standard two-layer feed-forward network with GELU activation.
* **`Block`**: A single Transformer block combining Layer Normalization, Causal Self-Attention, another Layer Normalization, and an MLP.
* **`GPT`**: The main model class, stacking `Block`s and including token embeddings (`wte`), positional embeddings (`wpe`), and a language modeling head (`lm_head`). It also incorporates weight sharing between `wte` and `lm_head`.

Two configurations (`GPTConfig124M` and `GPTConfig19M`) are provided to define the respective model sizes.

## 💾 Pre-trained Models

To run the speculative decoding demonstration, you will need the pre-trained weights for both the 124M (target) and 19M (draft) parameter models.

You can download these `.pth` files from the Hugging Face repository:

* **Download from Hugging Face:** [https://huggingface.co/fridayfringe/nanogpt_124M/tree/main](https://huggingface.co/fridayfringe/nanogpt_124M/tree/main)

Please download the following two files and place them in the root directory of this repository:

* `124mpara.pth` (for the 124M target model)
* `19M.pth` (for the 19M draft model)

## 🛠️ Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/YourUsername/SpeculativeNanoGPT.git](https://github.com/YourUsername/SpeculativeNanoGPT.git) # Replace YourUsername
    cd SpeculativeNanoGPT
    ```

2.  **Install Dependencies:**
    This project requires PyTorch and `tiktoken`.
    ```bash
    pip install torch tiktoken
    ```
    *(Optional but recommended: Ensure you have a CUDA-enabled PyTorch installation if you plan to use a GPU for faster inference.)*

3.  **Download Models:**
    Follow the instructions in the "Pre-trained Models" section above to download `124mpara.pth` and `19M.pth` and place them in the project root.

## 🚀 Usage

The primary way to run the demonstration is through the provided Jupyter Notebook.

1.  **Open the Jupyter Notebook:**
    ```bash
    jupyter notebook speculative_nanogpt.ipynb
    ```
2.  **Run All Cells:** Execute all cells in the notebook. The notebook will:
    * Load the necessary modules and configurations.
    * Load the pre-trained 124M and 19M models.
    * Initialize the `SpeculativeDecoder`.
    * Perform standard generation using only the 124M model.
    * Perform speculative decoding using the 19M draft model and the 124M target model.
    * Print a comparison of the generated text, speeds, and speculative decoding statistics.

### Example Generation from Notebook Output


Prompt: love in the air Input tokens: 4
normal generation (124M model only):
Generated text: love in the air

Pushing through

No time (get your money)

Then you just fall up inside

It's so we're feeling lonely when we're missing with you

And I know you won't call your name again

There's no need in your blood

Keep me waiting so high (hey

Pushing it up and down

"
Turn On	"It's
Speed: 7.18 tokens/sec
Time: 13.36 seconds

Speculative decoding (19M draft -> 124M target):
Generated text: love in the air tonight tonight

We got no longer got but our plans ain't nothing wrong

Just let the darkness break in the

My friends who would laugh and watch me cry

But my eyes still think of me

That one time would be only one another one

And I have what I never know

Just let the darkness break in the clouds

So I said

So why'd
Speed: 10.67 tokens/sec
Time: 9.09 seconds
Acceptance rate: 0.77
Speedup: 1.48x

Total iterations: 50
Total draft tokens: 150
Total accepted tokens: 97


As demonstrated in the example output, speculative decoding provides a noticeable speedup:

* **Normal Generation Speed**: 7.18 tokens/sec
* **Speculative Decoding Speed**: 10.67 tokens/sec
* **Speedup**: **~1.48x**
* **Acceptance Rate**: 0.77 (meaning 77% of the tokens proposed by the smaller draft model were accepted by the larger target model).

These results highlight the efficiency gains achieved by using the speculative decoding strategy.

## ⚙️ Customization

You can modify the generation parameters within the `speculative_nanogpt.ipynb` notebook:

* `max_length`: The maximum number of tokens to generate.
* `temperature`: Controls the randomness of predictions. Higher values mean more random outputs.
* `top_k`: Filters the top K most likely tokens to sample from.
* `num_draft_tokens`: The number of tokens the draft model attempts to predict in each speculative step. Experimenting with this value can impact speedup and acceptance rate.

## 🙏 Acknowledgements

This project is built upon the foundational work of:

* **Andrej Karpathy's NanoGPT**: For the elegant and clear implementation of the GPT architecture, which forms the base of the models used here.
* **Researchers behind Speculative Decoding**: For pioneering this innovative inference acceleration technique.
---
