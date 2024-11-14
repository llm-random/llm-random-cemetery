import argparse

# [ '--dataset_type',
#  'c4',
#  '--git_branch',
#  'test_backends_2024-11-14_12-06-05',
#  '--path_to_entry_config',
#  'research/dummy_backends_project/config.yaml',
#  '--all_config_paths',
#  'research/dummy_backends_project/config.yaml',
#  '--n_gpus',
#  '0',
#  '--train_dataset_path',
#  '/home/ubuntu/llm-random-group/datasets/c4_train',
#  '--validation_dataset_path',
#  '/home/ubuntu/llm-random-group/


def introduce_parser_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument("--name", type=str)
    parser.add_argument("--dataset_type", type=str)
    parser.add_argument("--git_branch", type=str)
    parser.add_argument("--path_to_entry_config", type=str)
    parser.add_argument("--all_config_paths", type=str)
    parser.add_argument("--n_gpus", type=int)
    parser.add_argument("--train_dataset_path", type=str)
    parser.add_argument("--validation_dataset_path", type=str)
    parser.add_argument("--tags", type=str, nargs="*")

    return parser
