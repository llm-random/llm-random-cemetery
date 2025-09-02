from collections import OrderedDict
import os
from pathlib import Path
import re
import torch

from src.core.utils import print_state_dict_info


def load_finalized_pc_checkpoint(model, load_config):
    checkpoint = torch.load(str(Path(load_config.path, load_config.model_checkpoint_filename)))
    if os.environ["RANK"] == "0": #dev
        print_state_dict_info(checkpoint["model"])
        print()
        print('---------------------------------')
        print()
        print_state_dict_info(model.state_dict())

    # raise Exception("FIN")
    model.load_state_dict(checkpoint["model"])