# From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model

<p align="center">
  <img src="assets/SaHa_logo.png" alt="SaHa Logo" style="width: 90%; max-width: 450px;">
</p>

> Official PyTorch implementation for [From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model](https://arxiv.org/abs/2508.00955)

<a target="_blank" href="https://arxiv.org/abs/2508.00955">
<img style="height:22pt" src="https://img.shields.io/badge/-Paper-black?style=flat&logo=arxiv"></a>
<a target="_blank" href="https://huggingface.co/Y-J-Ju/SaHa-Qwen2-VL-2B-Instruct">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20SaHa--2B-red?style=flat"></a>
<a target="_blank" href="https://huggingface.co/Y-J-Ju/SaHa-Qwen2-VL-7B-Instruct">
<img style="height:22pt" src="https://img.shields.io/badge/-🤗%20SaHa--7B-red?style=flat"></a>


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
CUDA_VISIBLE_DEVICES=0 python -m cli.sample_mmeb --model_name Qwen/Qwen2-VL-2B-Instruct \
  --model_backbone qwen2_vl \
  --encode_output_path outputs/qwen2vl-mmeb_negs/ \
  --max_len 2048 --image_resolution random \
  --pooling last --normalize True \
  --dataset_name TIGER-Lab/MMEB-train \
  --subset_name A-OKVQA HatefulMemes DocVQA MSCOCO_i2t MSCOCO_t2i OK-VQA ... \
  --per_device_eval_batch_size 16 \
  --image_dir /workspace/VLM2Vec/MMEB-train \
  --dataset_split diverse_instruction \
  --Q_prompt "Given an image, summarize the provided image in one word. Given only text, describe the text in one word." \
  --D_prompt "Given an image, summarize the provided image in one word. Given only text, describe the text in one word." \
  --dataloader_num_workers 4 \
  --top_k 15 \
  --pool_mult 3
```

### 3. Fine-Tuning

Fine-tune the model using the data pre-processed by SaHa. We use LoRA for efficient training.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=22447 --max_restarts=0 -m cli.train \
  --model_name Qwen/Qwen2-VL-7B-Instruct \
  --model_backbone qwen2_vl \
  --output_dir experiments/qwen2vl-7b-mmeb_neg15_mult3 \
  --bf16 --pooling last \
  --lora --lora_r 8 \
  --dataset_name outputs/qwen2vl-mmeb_negs \
  --split_name original \
  --subset_name A-OKVQA HatefulMemes DocVQA MSCOCO_i2t MSCOCO_t2i OK-VQA ... \
  --num_sample_per_subset 100000 \
  --image_dir /workspace/VLM2Vec/MMEB-train \
  --eval_strategy no \
  --dataloader_num_workers 8 \
  --image_resolution mid_low --max_len 1024 \
  --logging_steps 2 \
  --lr_scheduler_type linear --learning_rate 5e-5 \
  --max_steps 1500 \
  --warmup_steps 100 --save_steps 1000 --normalize True \
  --temperature 0.02 --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --grad_cache True --gc_q_chunk_size 4 --gc_p_chunk_size 4 \
  --save_safetensors False --remove_unused_columns False \
  --report_to wandb \
  --ddp_timeout 700000 \
  --hard_negatives 15 \
  --Q_prompt "Given an image, summarize the provided image in one word. Given only text, describe the text in one word." \
  --D_prompt "Given an image, summarize the provided image in one word. Given only text, describe the text in one word."
```

### 4\. Evaluation of Fine-Tuned Model

Evaluate your model on the MMEB test set.

```bash
python -m cli.evaluate --model_name Qwen/Qwen2-VL-7B-Instruct \
  --model_backbone qwen2_vl \
  # Remove the below line if you want to evaluate a zero-shot model
  --lora --checkpoint_path [your checkpoint path] \
  --encode_output_path outputs/qwen2vl-7b-mid_low_mmeb_zero/ \
  --max_len 2048 --image_resolution random \
  --pooling last --normalize True \
  --dataset_name TIGER-Lab/MMEB-eval \
  --subset_name MSCOCO_t2i MSCOCO_i2t NIGHTS WebQA OVEN FashionIQ \
  --dataset_split test --per_device_eval_batch_size 16 \
  --image_dir [image path] \
  --Q_prompt "Given an image, summarize the provided image in one word. Given only text, describe the text in one word." \
  --D_prompt "Given an image, summarize the provided image in one word. Given only text, describe the text in one word."
```

Evaluate your model on the SugarCrepe.

```bash
python -m cli.evaluate_sugarcrepe \
  --model_name Qwen/Qwen2-VL-7B-Instruct \
  --model_backbone qwen2_vl \
  # Remove the below line if you want to evaluate a zero-shot model
  --lora --checkpoint_path [your checkpoint path] \
  --dataset_name sugar_crepe/data \
  --image_dir [image path] \
  --encode_output_path outputs/qwen2vl-7b_sugar/ \
  --max_len 2048 --image_resolution random \
  --pooling last --normalize True \
  --per_device_eval_batch_size 16
```

Evaluate your model on the SugarCrepe++.

```bash
python -m cli.evaluate_sugarcrepe_plus \
  --model_name Qwen/Qwen2-VL-2B-Instruct \
  --model_backbone qwen2_vl \
  # Remove the below line if you want to evaluate a zero-shot model
  --lora --checkpoint_path [your checkpoint path] \
  --dataset_name sugar_crepe_plus \
  --image_dir [image path] \
  --encode_output_path outputs/qwen2vl-2b-sugar_plus/ \
  --max_len 2048 --image_resolution random \
  --pooling last --normalize True \
  --per_device_eval_batch_size 16
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
@misc{ju2025generatorembedder,
      title={From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model}, 
      author={Yeong-Joon Ju and Seong-Whan Lee},
      year={2025},
      eprint={2508.00955},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2508.00955}, 
}
```