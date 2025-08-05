# Performance Testing - README

This README is to be used as a step-by-step guide on using [NVIDIA's performance scripts](https://github.com/NVIDIA/NeMo/tree/main/scripts/performance) on AWS EC2 instances, regardless of platform and orchestration service. This is for Slurm only, and builds on top of [AWS' NeMo 2.0 sample](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/megatron/nemo/slurm).

## 0. Follow pre-requisites for NeMo setup
```bash
conda create --name nemo python==3.10.12
conda activate nemo

git clone https://github.com/NVIDIA/NeMo
git checkout v2.5.0rc0
cd NeMo
pip install -e '.[all]'
```

## 1. Create Custom Dockerfile
Latest Dockerfile can be found [here](https://github.com/aws-samples/awsome-distributed-training/blob/main/3.test_cases/megatron/nemo/Dockerfile).

## 2. Build and convert container with `enroot`:
```bash
# Build Docker image
sudo docker build --progress=plain -t aws-nemo:25.07 -f Dockerfile .

# Convert to enroot squash file
enroot import -o ~/aws-nemo-25-07.sqsh dockerd://aws-nemo:25.07
```

## 3. Modify `slurm_executor` implementation in performance scripts to use EFA
This fork already has this change, but for information, the configuration looks like this for the `slurm_executor` [definition](https://github.com/amanshanbhag/NeMo-AWS/blob/main/scripts/performance/executors.py):
```python
    PERF_ENV_VARS = {
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",  # Disable caching NCCL communication buffer memory
        "TRANSFORMERS_OFFLINE": "1",  # Enable online downloads from HuggingFace
        "TOKENIZERS_PARALLELISM": "False",  # Restrict warning message prints
        "NCCL_NVLS_ENABLE": "0",  # Disable NVLink SHARP to save memory
        "NVTE_FLASH_ATTN": "1",  # Enable Flash Attention, which is needed to enable cuDNN fused attention
        "NVTE_FUSED_ATTN": "1",  # Enable cuDNN fused attention
        "NEMO_LOG_MEMORY_USAGE": "1",  # Print memory allocation
        "FI_PROVIDER": "efa",
        "FI_EFA_USE_HUGE_PAGE": "0",
        "NCCL_DEBUG": "INFO"
    }
```

## 4. To change `seq_len` (8192 by default)
Edit the individual recipe file. Example: [Llama3-70B](https://github.com/amanshanbhag/NeMo-AWS/blob/067d83c9b2da632df2b12a562b7f19854eb3b20b/nemo/collections/llm/recipes/llama3_70b.py#L192)

## 5. To modify `gradient_accumulation`:
While there is no direct knob to change the `ga` value, we would have to calculate it manually using the knobs available to us:
```
ga = (gbs) / (mbs * dp)
dp = (num_gpus) / (tp * pp * cp)
```

## 6. To get results
Your resulting training will likely look like:
```
...
Training epoch 0, iteration 74/99 | lr: 1.124e-05 | global_batch_size: 128 | global_step: 74 | max_memory_reserved: 64286097408 | max_memory_allocated: 62777982976 | reduced_train_loss: 11.87 | train_step_timing in s: 8.768 | TFLOPS_per_GPU: 865.7 | consumed_samples: 9600
Training epoch 0, iteration 75/99 | lr: 1.139e-05 | global_batch_size: 128 | global_step: 75 | max_memory_reserved: 64286097408 | max_memory_allocated: 62777982976 | reduced_train_loss: 11.87 | train_step_timing in s: 8.748 | TFLOPS_per_GPU: 867.7 | consumed_samples: 9728
Training epoch 0, iteration 76/99 | lr: 1.154e-05 | global_batch_size: 128 | global_step: 76 | max_memory_reserved: 64286097408 | max_memory_allocated: 62777982976 | reduced_train_loss: 11.87 | train_step_timing in s: 8.756 | TFLOPS_per_GPU: 866.9 | consumed_samples: 9856
...
```

The relevant metrics are `TFLOPS_per_GPU` and `train_step_timing`.
Throughput (in tps) can be calculated with:
```
tp = (seq_len * gbs) / (train_step_timing)
```
