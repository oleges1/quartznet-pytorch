import torch
import random
import numpy as np

def fix_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def remove_from_dict(the_dict, keys):
    for key in keys:
        the_dict.pop(key, None)
    return the_dict