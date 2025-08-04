# Performance Testing - README

This README is to be used as a step-by-step guide on using [NVIDIA's performance scripts](https://github.com/NVIDIA/NeMo/tree/main/scripts/performance) on AWS EC2 instances, regardless of platform and orchestration service. This is for Slurm only, and builds on top of [AWS' NeMo 2.0 sample](https://github.com/aws-samples/awsome-distributed-training/tree/main/3.test_cases/megatron/nemo/slurm).

## 0. Follow pre-requisites for NeMo setup
```bash
conda create --name nemo python==3.10.12
conda activate nemo

pip install "nemo_toolkit[all]"

git clone https://github.com/NVIDIA/NeMo
cd NeMo
pip install '.[all]'
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

## 3. Create an `env_vars.json` file:
```json
{
    "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
    "NVTE_DP_AMAX_REDUCE_INTERVAL": "0",
    "NVTE_ASYNC_AMAX_REDUCTION": "1",
    "NVTE_FUSED_ATTN": "0",
    "FI_PROVIDER": "efa",
    "FI_EFA_USE_HUGE_PAGE": "0",
    "NCCL_DEBUG": "INFO"
}
```

## 4. Modify `slurm_executor` implementation in performance scripts to use EFA

## 5. To change `seq_len` (8192 by default)
Edit the individual recipe file. Example: [Llama3-70B](https://github.com/amanshanbhag/NeMo-AWS/blob/067d83c9b2da632df2b12a562b7f19854eb3b20b/nemo/collections/llm/recipes/llama3_70b.py#L192)

## 6. Modify `gradient_accumulation`:
While there is no direct knob to change the `ga` value, we would have to calculate it manually using the knobs available to us:
```
ga = (gbs) / (mbs * dp)
dp = (num_gpus) / (tp * pp * cp)
```
