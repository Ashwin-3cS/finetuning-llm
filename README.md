# Fine-tuning Llama 2 with QLoRA

A hands-on notebook for fine-tuning **Llama 2 (7B chat)** on a single Colab GPU using **QLoRA** (4-bit quantization + LoRA via PEFT) and the `SFTTrainer` from TRL.

## Dataset

We fine-tune on a custom instruction dataset hosted on Hugging Face:

- **[`mlabonne/guanaco-llama2-1k`](https://huggingface.co/datasets/mlabonne/guanaco-llama2-1k)** — 1,000 samples reformatted to follow the Llama 2 chat prompt template (`<s>[INST] ... [/INST] ...`).

It is derived from [`timdettmers/openassistant-guanaco`](https://huggingface.co/datasets/timdettmers/openassistant-guanaco). The full reformatted version is available at [`mlabonne/guanaco-llama2`](https://huggingface.co/datasets/mlabonne/guanaco-llama2).

## Model

- Base: [`NousResearch/Llama-2-7b-chat-hf`](https://huggingface.co/NousResearch/Llama-2-7b-chat-hf)
- Output: `Llama-2-7b-chat-finetune` (LoRA adapter, then merged to FP16)

## How to run

1. Open `Fine_tune_Llama_2.ipynb` in Google Colab (**File → Open notebook → GitHub → `Ashwin-3cS/finetuning-llm`**).
2. Use a GPU runtime (T4 is enough).
3. Run the cells top to bottom. The first cell installs pinned dependencies and **restarts the runtime** (the "session crashed" message is expected — just continue).

> **Note:** Inference is run on the **merged FP16 model** (Step 7), not the 4-bit model. Generating directly from the 4-bit QLoRA model is numerically unstable on some GPUs / `bitsandbytes` versions (e.g. Tesla T4) and produces repeated-token garbage.

Excalidraw Diagram goes here:
