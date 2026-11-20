"""Rebuttal: lm-eval on HF checkpoints with the April settings of the paper's evals
(evaluate_hf_models.py + configs/pc_project_mas/eval_hf_models.yaml): the nine tasks,
zero-shot, hf backend, batch_size 4, tokenizer meta-llama/Llama-3.2-1B, no limit.

    python scripts/rebuttal/lmeval_ckpts.py --out-dir DIR NAME=GLOB [NAME=GLOB ...]

Each GLOB is resolved at run time (the newest match is used), so it can name a checkpoint
folder that only exists once its training job has finished. Results go to
DIR/NAME_results.json in the same format as mastest_evals/evals_llama1b/*_results.json.
"""
import argparse
import glob
import json
import os
import sys

# social_iqa ships a dataset script. lem.yaml exports this, entropy.yaml does not, and
# without it the task aborts asking for interactive confirmation.
os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "openbookqa",
         "sciq", "social_iqa", "lambada_openai"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("models", nargs="+", help="NAME=GLOB")
    args = ap.parse_args()
    from lm_eval import evaluator

    os.makedirs(args.out_dir, exist_ok=True)
    failed = 0
    for spec in args.models:
        name, pattern = spec.split("=", 1)
        matches = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not matches:
            print(f"[lmeval] {name}: no checkpoint matches {pattern}", flush=True)
            failed += 1
            continue
        path = matches[-1]
        print(f"[lmeval] {name}: {path}", flush=True)
        results = evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={path},tokenizer={args.tokenizer},batch_size={args.batch_size},trust_remote_code=True",
            tasks=TASKS,
            limit=args.limit,
            num_fewshot=0,
            device=args.device,
            log_samples=False,
        )
        results["rebuttal_checkpoint_path"] = path
        out = os.path.join(args.out_dir, f"{name}_results.json")
        with open(out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        accs = {t: results["results"][t].get("acc,none") for t in TASKS}
        mean = sum(v for v in accs.values() if v is not None) / len(TASKS)
        print(f"[lmeval] {name}: mean acc {mean:.4f} -> {out}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
