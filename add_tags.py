import neptune


NEPTUNE_PROJECT = "pmtest/llm-random"


project = neptune.init_project(
  project=NEPTUNE_PROJECT,
  mode="sync"
)

all_runs = project.fetch_runs_table(owner="szysad", tag=["short", "reduced_bs"]).to_pandas()
all_runs = all_runs[all_runs["sys/name"].str.startswith("std_v4_c_lr_grid_placement_short")]

for run_id in all_runs["sys/id"]:
  run = neptune.init_run(project=NEPTUNE_PROJECT, with_id=run_id)
  run["sys/tags"].add("std_v4_c_lr_grid_placement_short")
  run.sync()

print(all_runs.shape)