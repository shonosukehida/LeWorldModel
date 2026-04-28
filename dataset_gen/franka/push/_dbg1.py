import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import yaml
import pandas as pd
import imageio
from tqdm import tqdm
from dm_control import mujoco

from env.franka.env import FrankaSimEnv
from PIL import Image 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import glob

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import h5py
from dataset_gen.franka.push.gen_data_xyz_fazzy import FrankaDatasetGenerator



with open("dataset_gen/franka/push/config.yaml", "r") as f:
    config = yaml.safe_load(f)

dataset_generator = FrankaDatasetGenerator(config)

target_xyz = np.array([0.515, 0.0, 0.05])
dataset_generator._test_fk(target_xyz)

loop = 100
dataset_generator._test_loop(loop)