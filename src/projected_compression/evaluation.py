
from email.policy import strict
from transformers import AutoConfig, LlamaForCausalLM, AutoTokenizer
from lm_eval import evaluator
from src.projected_compression.initialization import create_model
import torch.distributed.checkpoint as dcp
import os
import torch
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions


def load_pc_state_dict_to_llama(state_dict, original_llama="meta-llama/Llama-3.1-8B"):
    conf = AutoConfig.from_pretrained(original_llama)

    # conf.num_hidden_layers = # stays the same
    # conf.num_key_value_heads = # stays the same
    # conf.num_attention_heads = # stays the same

    conf.hidden_size = state_dict['target_model.encoder.blocks.0.attention_layer.norm.weight'].shape[0]
    conf.intermediate_size = state_dict['target_model.encoder.blocks.0.ff_layer.layer.gate.weight'].shape[0]

    llama = LlamaForCausalLM(conf)

    new_state_dict = {
        'model.embed_tokens.weight': state_dict['target_model.embedding'],
        'model.norm.weight': state_dict['target_model.head.norm.weight'],
        'lm_head.weight': state_dict['target_model.head.linear.weight'],
    }

    # embed_tokens
    for layer in range(conf.num_hidden_layers):
        prefix_src = f'target_model.encoder.blocks.{layer}.'
        prefix_tgt = f'model.layers.{layer}.'

        # attention
        new_state_dict[f'{prefix_tgt}input_layernorm.weight'] = state_dict[f'{prefix_src}attention_layer.norm.weight']
        new_state_dict[f'{prefix_tgt}self_attn.q_proj.weight'] = state_dict[f'{prefix_src}attention_layer.layer.q_proj.weight']
        new_state_dict[f'{prefix_tgt}self_attn.k_proj.weight'] = state_dict[f'{prefix_src}attention_layer.layer.k_proj.weight']
        new_state_dict[f'{prefix_tgt}self_attn.v_proj.weight'] = state_dict[f'{prefix_src}attention_layer.layer.v_proj.weight']
        new_state_dict[f'{prefix_tgt}self_attn.o_proj.weight'] = state_dict[f'{prefix_src}attention_layer.layer.o_proj.weight']
        # mlp
        new_state_dict[f'{prefix_tgt}mlp.gate_proj.weight'] = state_dict[f'{prefix_src}ff_layer.layer.gate.weight']
        new_state_dict[f'{prefix_tgt}mlp.up_proj.weight'] = state_dict[f'{prefix_src}ff_layer.layer.ff_pre_act.weight']
        new_state_dict[f'{prefix_tgt}mlp.down_proj.weight'] = state_dict[f'{prefix_src}ff_layer.layer.ff_post_act.weight']
        new_state_dict[f'{prefix_tgt}post_attention_layernorm.weight'] = state_dict[f'{prefix_src}ff_layer.norm.weight']

    llama.load_state_dict(new_state_dict, strict=True)
    return llama
    
def log_eval(metric_logger, eval_results: dict):
    """Log evaluation results to Neptune."""
    for task_name, metrics in eval_results["results"].items():
        for metric_name, value in metrics.items():
            clean_metric_name = metric_name.replace(",none", "")
            metric_logger.run[f"eval/{task_name}/{clean_metric_name}"] = value

    metric_logger.run["eval/limit"] = eval_results["config"]["limit"]

def evaluation(cfg, metric_logger):
    model = create_model(cfg.model, cfg.projected_compression)
    original_llama = "meta-llama/Llama-3.1-8B"
    path_to_load = "/storage_nvme_1/mpioro/pc/model" # "/net/scratch/hscra/plgrid/plgcrewtool/tutaj_pc_hej_8b_minitron_testy_17/11472834/0/step_1023/model"
    path_to_save = os.path.join(path_to_load, "converted_to_hf")
    tasks = 'arc_easy,arc_challenge'
    eval_batch_size = 64

    dcp.load(model.state_dict(), checkpoint_id=path_to_load)
    model.prepare_compressed_weights()

    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
    )
    if os.environ.get("RANK", "0") == "0":
        tokenizer = AutoTokenizer.from_pretrained(original_llama)
        llama = load_pc_state_dict_to_llama(model_state_dict, original_llama=original_llama)
        tokenizer.save_pretrained(path_to_save)
        llama.save_pretrained(path_to_save)


        eval_results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={path_to_save},tokenizer={path_to_save}",
            tasks=tasks,
            device='cuda',
            log_samples=False,
            batch_size=eval_batch_size
        )
        print(eval_results['results'])
        log_eval(metric_logger, eval_results)

    
    return None, None, None, None, None

    


