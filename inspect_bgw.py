import wandb
import json

api = wandb.Api()
runs = api.runs("ideas_cv/tml-bgw")
print(f"Total runs found in ideas_cv/tml-bgw: {len(runs)}")

if len(runs) > 0:
    run = runs[0]
    print(f"Latest run: {run.name} ({run.id})")
    
    data = {
        "name": run.name,
        "id": run.id,
        "created_at": run.created_at,
        "tags": run.tags,
        "config": run.config,
        "summary": run.summary._json_dict
    }
    
    with open("scratch_inspect.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Saved run details to scratch_inspect.json")
else:
    print("No runs found in the project!")
