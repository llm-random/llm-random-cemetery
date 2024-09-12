{'name': 'ms_repeater', 'tags': ['ms', 'repeater', 'test', 'loss_checkpoint_chungs=0M'], 'expansion_rate': 8, 'n_steps': 2000, 'batch_size': 100, 'cutoff': 256, 'save_weights_path': 'test_saving/0', 'load_weights_path': 'test_saving/0', 'save_weights_interval': 100, 'dmodel': 512, 'n_blocks': 8, 'n_att_heads': 8, 'ff_mode': 'token_choice', 'capacity_factor': 1.0, 'activation_type': 'silu', 'moe_inner_expert': 'ff_gated', 'granularity': 1, 'get_router_values_from': 'weights', 'layer_norm_in_expert_choice': False, 'final_lr_step': -1, 'lr_warmup_percent': 0.01, 'lr_warmup_steps': 0, 'scheduler': 'cosine', 'final_lr_fraction': 0.1, 'init_scale': 0.1, 'learning_rate': 0.0002, 'weight_decay': 0.1, 'mixed_precision': True, 'mixed_precision_dtype': 'bfloat16', 'flash_attention': True, 'loss_checkpoint_chungs': 0, 'fsdp_enabled': True, 'fsdp_modules_to_wrap': 'EmbeddingLayer,PredictionHead,TransformerBlock', 'activation_checkpointing_modules': 'EmbeddingLayer,PredictionHead,TransformerBlock', 'fsdp_selective_precision_modules': 'AttentionMechanism,MoeGating,RoPE', 'softmax_over': 'experts', 'group_granular_moe_by_batch': True, 'use_torch_bmm': True, 'torch_compile': False, 'granular_moe_one_hot_impl': True, 'effective_dff_x': 4, 'init_type': 'truncated_normal', 'model_type': 'gpt', 'dataset_type': 'c4', 'logger_types': 'neptune', 'project_name': 'pmtest/llm-random', 'logging_interval_heavy': 5000, 'logging_interval_loss': 1000, 'grad_clip': 0.5, 'git_branch': 'ms_repeater_2024-09-12_11-16-16', 'path_to_entry_config': 'configs/ms-test/repeater_test.yaml', 'all_config_paths': 'configs/baselines/common.yaml,configs/baselines/gpt/dense/common.yaml,configs/baselines/gpt/expert_choice/base.yaml,configs/baselines/gpt/expert_choice/common.yaml,configs/baselines/gpt/expert_choice/granularity/4/base.yaml,configs/experiments/constrained_scaling_laws/25M/model.yaml,configs/experiments/constrained_scaling_laws/common.yaml,configs/ms-test/repeater_test.yaml', 'n_gpus': 1, 'train_dataset_path': '/home/ubuntu/llm-random-group/datasets/c4_train', 'validation_dataset_path': '/home/ubuntu/llm-random-group/datasets/c4_validation'}
{'parent': 'configs/experiments/constrained_scaling_laws/25M/model.yaml', 'md5_parent_hash': '3067b2550c57a4b87c0c6a1287e6d902', 'time': '10:05:00', 'n_gpus': 1, 'cpus_per_gpu': 8, 'interactive_debug_session': False, 'interactive_debug': False, 'runner': 'research.conditional.train.cc_train', 'gres': 'gpu:1', 'mem_per_gpu': 125, 'nodelist': None, 'hf_datasets_cache': '/home/ubuntu/.cache', 'runs_multiplier': 1, 'train_dataset_path': '/home/ubuntu/llm-random-group/datasets/c4_train', 'validation_dataset_path': '/home/ubuntu/llm-random-group/datasets/c4_validation', 'common_directory': '/home/ubuntu/llm-random-group', 'singularity_image': '/home/ubuntu/llm-random-group/images/sparsity_2024.02.06_16.14.02.sif', 'grid_entrypoint': 'lizrd/grid/grid_entrypoint.sh'}