# From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model

> [](https://www.google.com/search?q=https://arxiv.org/abs/24XX.XXXXX)  Official PyTorch implementation for **"From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model"**

We propose an efficient framework for universal multimodal embeddings, which bridges the gap between the generative nature of Multimodal Large Language Models (MLLMs) and the needs of discriminative representation learning. This is achieved through two synergistic components:

1.  Our **hierarchical embedding prompt template** employs a two-level instruction architecture that forces the model to produce discriminative representations in a zero-shot manner.
2.  Our **self-aware hard negative sampling (SaHa)** strategy redefines the fine-tuning process by leveraging the model’s own understanding to efficiently mine challenging negatives while actively filtering out potential false negatives.


## 🚀 Key Features

  * **Training-Free Performance**: Achieves competitive performance on multimodal benchmarks like MMEB without any fine-tuning, using only our hierarchical prompt.
  * **Efficient Fine-Tuning**: Introduces Self-aware Hard Negative Sampling (SaHa), an offline sampling strategy that builds high-quality training batches to maximize performance and efficiency.
  * **State-of-the-Art Results**: Outperforms previous methods on the MMEB benchmark, especially in the classification and retrieval tasks, after efficient fine-tuning.


## 🔧 Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/yeongjoonJu/Gen2Embed.git
    cd Gen2Embed
    ```

2.  **Create a conda environment and install dependencies:**

    ```bash
    conda create -n gen2embed python=3.10 -y
    conda activate gen2embed
    pip install -r requirements.txt
    ```


## 📚 Dataset Preparation

Our experiments use the **[MMEB](https://github.com/TIGER-AI-Lab/VLM2Vec/tree/v1)**, **[SugarCrepe](https://github.com/RAIVNLab/sugar-crepe/tree/main)**, and **[SugarCrepe++](https://github.com/Sri-Harsha/scpp)** benchmarks. Please follow the instructions in the original repository to download and set up the dataset.


## ⚡️ How to Use

### 1. Zero-Shot Evaluation

You can evaluate the zero-shot performance of a base MLLM (e.g., Qwen2-VL) using our hierarchical prompt. This requires no training.

```bash
python evaluate.py \
    --model_name "Qwen/Qwen2-VL-2B" \
    --data_dir /path/to/your/data/MMEB \
    --prompt_mode "hierarchical" \
    --eval_split "test"
```

### 2. Data Sampling with SaHa

Before fine-tuning, process the training data using our Self-aware Hard Negative Sampling (SaHa) strategy. This is an offline, one-time process.

```bash
python sampling/saha_sampler.py \
    --model_name "Qwen/Qwen2-VL-2B" \
    --data_dir /path/to/your/data/MMEB \
    --output_dir /path/to/your/processed_data \
    --num_hard_negatives 7 \
    --pool_multiplier 6
```

### 3. Fine-Tuning

Fine-tune the model using the data pre-processed by SaHa. We use LoRA for efficient training.

```bash
torchrun --nproc_per_node=8 train.py \
    --model_name "Qwen/Qwen2-VL-2B" \
    --processed_data_path /path/to/your/processed_data/train_saha.json \
    --output_dir ./checkpoints/qwen2-vl-2b-saha \
    --lora_rank 8 \
    --learning_rate 5e-5 \
    --num_epochs 1.5 \
    --batch_size 16
```

### 4\. Evaluation of Fine-Tuned Model

Evaluate your fine-tuned model on the MMEB test set.

```bash
python evaluate.py \
    --model_name_or_path ./checkpoints/qwen2-vl-2b-saha \
    --data_dir /path/to/your/data/MMEB \
    --prompt_mode "hierarchical" \
    --eval_split "test"
```

## 📊 Results

Our framework significantly boosts the performance of base MLLMs.

**Zero-shot performance on MMEB using our hierarchical prompt:**

| Model | Params | Cla. | VQA | Ret. | Gro. | Avg. |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2-VL-2B (Baseline) | 2.2B | 21.4 | 8.9 | 9.5 | 21.2 | 13.9 |
| **Ours (Qwen2-VL-2B)** | **2.2B** | **48.6** | **33.3** | **44.1** | **52.6** | **43.3** |


**Fine-tuning with SaHa on MMEB:**

| Model (Base) | Method | Params | IND | OOD | Overall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| Qwen2-VL | VLM2Vec | 2.2B | 66.0 | 52.6 | 59.3 |
| Qwen2-VL | **Ours (SaHa)** | 2.2B | **71.2** | **62.1** | **67.1** |
| Qwen2-VL | VLM2Vec | 8.3B | 72.2 | 57.8 | 65.8 |
| Qwen2-VL | **Ours (SaHa)** | 8.3B | **76.4** | **67.4** | **72.4** |


## 🙏 Acknowledgements

This work was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant, funded by the Korea government (MSIT) (No. RS-2019-II190079 (Artificial Intelligence Graduate School Program (Korea University)), No. IITP-2025-RS-2024-00436857 (Information Technology Research Center (ITRC))).


## 📜 Citation

If you find our work useful, please consider citing our paper:

```bibtex
@article{ju2024generator,
  title={From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model},
  author={Ju, Yeong-Joon and Lee, Seong-Whan},
  journal={arXiv preprint arXiv:24XX.XXXXX},
  year={2025}
}
```