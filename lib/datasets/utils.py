import torch
import numpy as np
import random

# Function to provide consistent seeds to Dataloader workers
# to ensure Reproducability. Whether Reproducability is desired
# should be defined ih the configuration file and passed to the Datamodule
def deterministic_init_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
