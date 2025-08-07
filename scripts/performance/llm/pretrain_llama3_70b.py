# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import basename, splitext

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import nemo_run as run

from nemo.collections.llm.recipes.llama3_70b import pretrain_recipe
# Diff: Importing buffers for 4096 and 2048
from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
    # 8192 sequence length configs
    userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
    userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
    userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
    # 4096 sequence length configs
    userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096,
    userbuffers_bf16_b200_h18432_tp8_mbs1_seqlen4096,
    userbuffers_fp8_b200_h18432_tp8_mbs1_seqlen4096,
    # 2048 sequence length configs
    userbuffers_bf16_h100_h6144_tp2_mbs2_seqlen2048,
    userbuffers_fp8_h100_h6144_tp2_mbs2_seqlen2048,
    userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
    userbuffers_fp8_h100_h12288_tp4_mbs1_seqlen2048,
    userbuffers_bf16_b200_h12288_tp4_mbs1_seqlen2048,
    userbuffers_fp8_b200_h12288_tp4_mbs1_seqlen2048,
)

from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.run.plugins import MemoryProfilePlugin, NsysPlugin, PerfEnvPlugin

from ..argument_parser import parse_cli_args
from ..executors import slurm_executor
from ..helpers import args_sanity_check, get_user_configs, set_exp_logging_configs, set_primary_perf_configs
from ..utils import dump_config_diff_from_base_recipe, get_comm_overlap_callback_idx, hf_tokenizer


# Diff: Creating a function to replace the ub_cfg dict. Selects buffer based on sequence length
def get_user_buffer_config(gpu_type, compute_dtype, seq_len, tp_size, mbs):
    """Get appropriate user buffer config based on parameters."""
    
    # Map configurations by (gpu_type, compute_dtype, seq_len, tp_size, mbs)
    config_map = {
        # 8192 sequence length configs
        ("h100", "bf16", 8192, 4, 1): userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
        ("h100", "fp8", 8192, 4, 1): userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
        ("b200", "bf16", 8192, 2, 1): userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
        ("b200", "fp8", 8192, 2, 1): userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
        ("gb200", "bf16", 8192, 2, 1): userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
        ("gb200", "fp8", 8192, 2, 1): userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
        
        # 4096 sequence length configs
        ("b200", "bf16", 4096, 2, 1): userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096,
        ("b200", "bf16", 4096, 8, 1): userbuffers_bf16_b200_h18432_tp8_mbs1_seqlen4096,
        ("b200", "fp8", 4096, 8, 1): userbuffers_fp8_b200_h18432_tp8_mbs1_seqlen4096,
        ("gb200", "bf16", 4096, 2, 1): userbuffers_bf16_b200_h6144_tp2_mbs1_seqlen4096,
        ("gb200", "bf16", 4096, 8, 1): userbuffers_bf16_b200_h18432_tp8_mbs1_seqlen4096,
        ("gb200", "fp8", 4096, 8, 1): userbuffers_fp8_b200_h18432_tp8_mbs1_seqlen4096,

        # 2048 sequence length configs
        ("h100", "bf16", 2048, 2, 2): userbuffers_bf16_h100_h6144_tp2_mbs2_seqlen2048,
        ("h100", "fp8", 2048, 2, 2): userbuffers_fp8_h100_h6144_tp2_mbs2_seqlen2048,
        ("h100", "bf16", 2048, 4, 1): userbuffers_bf16_h100_h12288_tp4_mbs1_seqlen2048,
        ("h100", "fp8", 2048, 4, 1): userbuffers_fp8_h100_h12288_tp4_mbs1_seqlen2048,
        ("b200", "bf16", 2048, 4, 1): userbuffers_bf16_b200_h12288_tp4_mbs1_seqlen2048,
        ("b200", "fp8", 2048, 4, 1): userbuffers_fp8_b200_h12288_tp4_mbs1_seqlen2048,
        ("gb200", "bf16", 2048, 4, 1): userbuffers_bf16_b200_h12288_tp4_mbs1_seqlen2048,
        ("gb200", "fp8", 2048, 4, 1): userbuffers_fp8_b200_h12288_tp4_mbs1_seqlen2048,
    }
    
    key = (gpu_type, compute_dtype, seq_len, tp_size, mbs)
    return config_map.get(key)

def override_recipe_configs(
    args: str,
    num_nodes: int,
    mbs: int,
    gbs: int,
    tp_size: int,
    pp_size: int,
    cp_size: int,
    vp_size: int,
    ep_size: int,
    enable_cuda_graphs: bool,
    use_mcore_fsdp: bool,
    recompute_layers: int,
    activation_offload_layers: int,
    recompute_modules: list,
    keep_fsdp_fp8_transpose_cache: bool,
    use_user_buffer_registration: bool,
    use_sharp: bool,
):
    """
    llama3 70b pre-train recipe aimed at achieving best possible performance.

    NOTE: Use fp8 precision training with caution. It might not give desirable results.
    """

    # Diff: Get sequence length from args, default to 8192
    seq_len = getattr(args, 'seq_len', 8192)

    recipe = pretrain_recipe(performance_mode=True)

    # Diff: Set sequence length in data config
    recipe.data.seq_length = seq_len

    # Diff: Set sequence length in model config
    recipe.model.config.seq_length = seq_len

    recipe = set_primary_perf_configs(
        recipe,
        "pre_train",
        num_nodes,
        args.gpus_per_node,
        mbs,
        gbs,
        args.max_steps,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        enable_cuda_graphs=enable_cuda_graphs,
        use_mcore_fsdp=use_mcore_fsdp,
        use_user_buffer_registration=use_user_buffer_registration,
        use_fsdp_double_buffer=args.use_fsdp_double_buffer,
        use_sharp=use_sharp,
        recompute_layers=recompute_layers,
        activation_offload_layers=activation_offload_layers,
        compute_dtype=args.compute_dtype,
        fp8_recipe=args.fp8_recipe,
        nccl_communicator_config_path=args.nccl_communicator_config_path,
        keep_fsdp_fp8_transpose_cache=keep_fsdp_fp8_transpose_cache,
    )
    recipe = set_exp_logging_configs(
        recipe, "pre_train", "llm", "llama3", args.tensorboard, args.wandb, args.wandb_prj_name, args.wandb_job_name
    )

    gpu_type = args.gpu.lower()

    # data module configs
    if args.use_hf_tokenizer:
        recipe.data.tokenizer = hf_tokenizer("meta-llama/Meta-Llama-3-70B")
    else:
        recipe.data.tokenizer = run.Config(
            get_nmt_tokenizer, library="null", model_name="NullTokenizer", vocab_size=128256
        )
        recipe.model.tokenizer = recipe.data.tokenizer

    # Diff: Only use this dict as a fallback (in case exact tp, pp configs are not found)
    ub_cfg = {
        "h100": {
            "bf16": userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_h100_h8192_tp4_mbs1_seqlen8192,
        },
        "b200": {
            "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
        },
        "gb200": {
            "bf16": userbuffers_bf16_b200_h8192_tp2_mbs1_seqlen8192,
            "fp8": userbuffers_fp8_b200_h8192_tp2_mbs1_seqlen8192,
        },
    }

    # comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    # assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    # tp_comm_overlap_cfg = ub_cfg[gpu_type][args.compute_dtype]
    # # needed as tp_overlap_configs.userbuffers are dataclass objects which are unserializable
    # tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
    # recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg
    # return recipe

    # Diff: Use a combination of existing logic (ub_cfg) and new logic (newly defined get_user_buffer_config())
    comm_overlap_callback_idx = get_comm_overlap_callback_idx(recipe.trainer.callbacks)
    assert comm_overlap_callback_idx is not None, "MegatronCommOverlapCallback missing. Required for performance."

    # Try to get  match first
    tp_comm_overlap_cfg = get_user_buffer_config(gpu_type, args.compute_dtype, seq_len, tp_size, mbs)

    # If no exact match, fall back to original behavior
    if tp_comm_overlap_cfg is None:
        # For sequence length 8192, fall back to default dict
        if seq_len == 8192:
            tp_comm_overlap_cfg = ub_cfg[gpu_type][args.compute_dtype]
        # For other sequence lengths, disable user buffers to avoid memory errors (tensor size mismatch)
        else:
            print(f"Warning: No user buffer config for seq_length={seq_len}, tp={tp_size}, pp={pp_size}, mbs={mbs}. Disabling user buffers. This may have perf implications.")
            recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap = False
            return recipe
                
    # Needed as tp_overlap_configs.userbuffers are dataclass objects which are unserializable
    tp_comm_overlap_cfg = fdl.cast(run.Config, fdl_dc.convert_dataclasses_to_configs(tp_comm_overlap_cfg))
    recipe.trainer.callbacks[comm_overlap_callback_idx].tp_comm_overlap_cfg = tp_comm_overlap_cfg

    return recipe




if __name__ == "__main__":
    args = parse_cli_args().parse_args()
    args_sanity_check(args)

    kwargs = get_user_configs(args.gpu.lower(), "pre_train", "llama3", "70b", args)
    (
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        _,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
        recompute_modules,
        keep_fsdp_fp8_transpose_cache,
        use_user_buffer_registration,
        use_sharp,
    ) = kwargs[:17]

    recipe = override_recipe_configs(
        args,
        num_nodes,
        mbs,
        gbs,
        tp_size,
        pp_size,
        cp_size,
        vp_size,
        ep_size,
        enable_cuda_graphs,
        use_mcore_fsdp,
        recompute_layers,
        activation_offload_layers,
        recompute_modules,
        keep_fsdp_fp8_transpose_cache,
        use_user_buffer_registration,
        use_sharp,
    )

    exp_config = f"{num_nodes}nodes_tp{tp_size}_pp{pp_size}_cp{cp_size}_vp{vp_size}_{mbs}mbs_{gbs}gbs"
    exp_name = f"{splitext(basename(__file__))[0]}_{args.compute_dtype}_{exp_config}"

    executor = slurm_executor(
        args.gpu.lower(),
        args.account,
        args.partition,
        args.log_dir,
        num_nodes,
        args.gpus_per_node,
        args.time_limit,
        args.container_image,
        custom_mounts=args.custom_mounts,
        custom_env_vars={},
        hf_token=args.hf_token,
        nemo_home=args.nemo_home,
        wandb_key=args.wandb_key,
        network='sharp' if use_sharp else None,
    )

    plugins = [
        PerfEnvPlugin(
            enable_vboost=True,
            nccl_pp_comm_chunksize=2097152 if pp_size > 1 else None,
            gpu_sm100_or_newer=(args.gpu.lower() in ['b200', 'gb200']),
            user_buffer_registration=use_user_buffer_registration,
        )
    ]
    if args.enable_nsys:
        plugins.append(NsysPlugin(start_step=5, end_step=6))
    if args.enable_memory_profile:
        assert args.memory_profile_out_path is not None
        plugins.append(MemoryProfilePlugin(dir=args.memory_profile_out_path))

    with run.Experiment(exp_name) as exp:
        exp.add(
            recipe,
            executor=executor,
            name=exp_name,
            plugins=plugins,
        )

        if not args.dryrun:
            exp.run(sequential=True, detach=True)
        else:
            exp.dryrun()

    if args.dump_config_diff_from_base_recipe:
        output_dir = exp.jobs[0].executor.job_dir
        # dump difference from base recipe
        base_recipe = pretrain_recipe(performance_mode=False)
        file_name = f"diff_from_base_recipe_{args.compute_dtype}.diff"
        dump_config_diff_from_base_recipe(base_recipe, recipe, output_dir, file_name=file_name)
        # dump difference from default perf recipe
        default_perf_recipe = pretrain_recipe(performance_mode=True)
        file_name = f"diff_from_default_perf_recipe_{args.compute_dtype}.diff"
        dump_config_diff_from_base_recipe(default_perf_recipe, recipe, output_dir, file_name=file_name)
