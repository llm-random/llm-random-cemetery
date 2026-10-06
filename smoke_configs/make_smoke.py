"""Generate fully-composed smoke-test configs (like run_exp's grid expansion), small sizes, stdout logging."""
import resolver  # noqa
from hydra import compose, initialize
from omegaconf import OmegaConf
from grid_generator.generate_configs import create_grid_config

SMOKE = "/storage_nvme_4/nano/j321m/pc/slicegpt_smoke"
STDOUT = ["infrastructure.metric_logger.type=stdout"]

def gen(config_dir, name, overrides, out, edit=None):
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name=name, overrides=overrides)
    d = create_grid_config(cfg)[0][0]          # first grid point
    if edit: edit(d)
    OmegaConf.save(OmegaConf.create(d), f"smoke_configs/{out}.yaml")
    print("wrote", out)

common_slice = ["slicegpt.calibration.n_samples=64", "slicegpt.eval.n_samples=16", *STDOUT]
# S1 / V1 on the real 1B model: no dff pruning, no slicing (d = D) -> losses must be unchanged
gen("../configs/slicegpt", "llama_1B_slice_hyb_entropy",
    common_slice + ["slicegpt.target_d=2048", "slicegpt.target_dff=8192", "slicegpt.orientation=pca", "slicegpt.save_path=''"], "s1_invariance")
# S2: real hybrid r50 slicing (small calibration) + save -> input for S3/S4
gen("../configs/slicegpt", "llama_1B_slice_hyb_entropy",
    common_slice + ["slicegpt_ratio=r50", f"slicegpt.save_path={SMOKE}", "slicegpt.save_name=llama1b_slicegpt_hyb_r50_smoke"], "s2_slice_r50")
# S4 / V5: RFT smoke, 4 GPUs FSDP2, 30 steps, loads the S2 checkpoint (sliced_nano)
def rft_edit(d):
    d["trainer"]["n_steps"] = 30
    d["trainer"]["checkpoint"]["save"]["path"] = f"{SMOKE}/rft_ckpt"
for name in ("llama_1B_slicegpt_hyb_entropy", "llama_1B_slicegpt_hyb_distillation_entropy"):
    gen("../configs/pc_project_mas", name,
        ["slicegpt_ratio=r50", f"trainer.checkpoint.load.path={SMOKE}/llama1b_slicegpt_hyb_r50_smoke.pt", *STDOUT],
        "s4_rft" + ("_distil" if "distil" in name else ""), rft_edit)
