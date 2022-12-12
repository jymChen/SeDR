
## SeDR: Segment Representation Learning for Long Documents Dense Retrieval

This repository contains all source code for the paper  ["SeDR: Segment Representation Learning for Long Documents Dense Retrieval"](https://arxiv.org/abs/2211.10841) .

## Requirements

```bash
transformers == 4.12.2
torch
faiss-gpu 
boto3
```

## Data Download

1. Download the datasets in need by running:
```bash
bash download_data.sh
```

2. Since we use STAR as warmup model for all experiments setting for long documents input to enable a fair comparison, please download [it](https://drive.google.com/drive/folders/18GrqZxeiYFxeMfSs97UxkVHwIhZPVXTc) and put it under `./data/star` for training.

## Data preprocess

Preprocess all datasets by running:

```bash
python SeDR_preprocess.py --data_path ./data --max_seq_length 2048
```

Then use the warmup model to generate the static hard negative by follow command:
```bash
python ./SeDR/inference.py --model_class STAR_MaxP --max_query_length 32 --max_seg_length 512 --max_seg_num 1 --model_path ./data/star --faiss_gpus 0 --gen_hardneg
```

## SeDR

Run the following code to train the model:

```bash
python ./SeDR/train.py --model_class SeDR --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --max_bsize 17  --preprocess_dir ./data/preprocess --hardneg_path ./data/hard_negative.json --init_path ./data/star --output_dir ./data/models --logging_dir ./data/log --learning 5e-5 --use_gradient_checkpointing --fp16 --hardneg_topk 100 --cache_size 50  --gradient_accumulation_steps 4
```

Run the following code to evaluate trained models:

```bash
python ./SeDR/inference.py --model_class SeDR --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --model_path ./data/models/epoch-4-512-4-17-SeDR --faiss_gpus 0
```

Last, futher train SeDR with ADORE mechanism ( with automatic evaluation) by running:
```bash
python ./SeDR/adore_train.py --metric_cut 200 --learning_rate 3e-6 --init_path ./data/models/epoch-4-512-4-17-SeDR --pembed_dir ./data/evaluate/epoch-4-512-4-17-SeDR-inf512-4 --model_save_dir ./data/adoremodels --log_dir ./data/log --preprocess_dir ./data/preprocess --model_gpu_index 0 --faiss_gpu_index 0
```

## SeDR-MaxP
Training:

```bash
python ./SeDR/train.py --model_class SeDR_MaxP --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --max_bsize 17  --preprocess_dir ./data/preprocess --hardneg_path ./data/hard_negative.json --init_path ./data/star --output_dir ./data/models --logging_dir ./data/log --learning 5e-5 --use_gradient_checkpointing --fp16 --hardneg_topk 100 --cache_size 50  --gradient_accumulation_steps 4
```
Inference:

```bash
python ./SeDR/inference.py --model_class SeDR_MaxP --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --model_path ./data/models/epoch-4-512-4-17-SeDR_MaxP --faiss_gpus 0
```

## SeDR-Transformer-Head

Training:

```bash
python ./SeDR/train.py --model_class SeDR_Transformer_Head --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --max_bsize 17  --preprocess_dir ./data/preprocess --hardneg_path ./data/hard_negative.json --init_path ./data/star --output_dir ./data/models --logging_dir ./data/log --learning 5e-5 --use_gradient_checkpointing --fp16 --hardneg_topk 100 --cache_size 50  --gradient_accumulation_steps 4
```
Inference:

```bash
python ./SeDR/inference.py --model_class SeDR_Transformer_Head --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --model_path ./data/models/epoch-4-512-4-17-SeDR_Transformer_Head --faiss_gpus 0
```

## SeDR-Longformer

To run on Longormer, you need to change Transformer version to 3.0.2 and use STAR checkpoint to initialize the STAR_Longformer by running

```bash
python ./SeDR_Longformer/Star2Star_Longformer.py
```

Training:

```bash
python ./SeDR_Longformer/train.py --max_query_length 32 --max_doc_length 2048 --max_seg_num 4 --max_bsize 7 --preprocess_dir ./data/preprocess --hardneg_path ./data/hard_negative.json --init_path ./data/starlongformer --output_dir ./data/models --logging_dir ./data/log --learning 5e-5 --use_gradient_checkpointing --fp16 --hardneg_topk 100 --cache_size 50  --gradient_accumulation_steps 4
```

Inference:

```bash
python ./SeDR_Longformer/inference.py --max_query_length 32 --max_doc_length 2048 --max_seg_num 4  --model_path ./data/models/epoch-4-2048-4-7-SeDR_Longformer --faiss_gpus 0
```

## SeDR-Global-Attention

Training:

```bash
python ./SeDR/train.py --model_class SeDR_Global_Attention --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --max_bsize 17  --preprocess_dir ./data/preprocess --hardneg_path ./data/hard_negative.json --init_path ./data/starlongformer --output_dir ./data/models --logging_dir ./data/log --learning 5e-5 --use_gradient_checkpointing --fp16 --hardneg_topk 100 --cache_size 50  --gradient_accumulation_steps 4
```

Inference:

```bash
python ./SeDR/inference.py --model_class SeDR_Global_Attention --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --model_path ./data/models/epoch-4-512-4-17-SeDR_Global_Attention --faiss_gpus 0
```

## STAR(MaxP)

Training:

```bash
python ./SeDR/train.py --model_class STAR_MaxP --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --max_bsize 17  --preprocess_dir ./data/preprocess --hardneg_path ./data/hard_negative.json --init_path ./data/star --output_dir ./data/models --logging_dir ./data/log --use_gradient_checkpointing --hardneg_topk 200 --gradient_accumulation_steps 4 --fp16
```

Inference:

```bash
python ./SeDR/inference.py --model_class STAR_MaxP --max_query_length 32 --max_seg_length 512 --max_seg_num 4 --model_path ./data/models/epoch-4-512-4-17-STAR_MaxP --faiss_gpus 0
```

## STAR-Multi

Training:

```bash
python ./SeDR/train.py --model_class STAR_Multi --max_query_length 32 --max_seg_length 512 --max_seg_num 1 --max_bsize 17  --preprocess_dir ./data/preprocess --hardneg_path ./data/hard_negative.json --init_path ./data/star --output_dir ./data/models --logging_dir ./data/log  --use_gradient_checkpointing --fp16 --hardneg_topk 200
```

Inference:

```bash
python ./SeDR/inference.py --model_class STAR_Multi --max_query_length 32 --max_seg_length 512 --max_seg_num 1 --model_path ./data/models/epoch-4-512-1-17-STAR_Multi --faiss_gpus 0
```

