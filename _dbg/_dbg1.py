import os
import sys

PROJECT_ROOT = "/home/shonosukehida/work/LeWorldModel"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import jepa
import torch

path = "/home/shonosukehida/.stable_worldmodel/lewm_epoch_100_object.ckpt"
obj = torch.load(path, map_location="cpu")

print("type:", type(obj))
print(hasattr(obj, "get_cost"))