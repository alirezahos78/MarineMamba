import os
import random
import sys

import numpy as np
import torch


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        ensure_dir(os.path.dirname(filename))
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"⚙️ Global Seed set to {seed}")
