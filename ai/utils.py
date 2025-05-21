# Standard library imports
import random
import math
import numpy as np
from collections import deque
import pickle
import os
from datetime import datetime

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# Game constants
MAX_STEPS_PER_EPISODE = 10000  # Maximum steps per training episode
SAVE_MODEL_DIR = "models"       # Directory to save AI models

# Create model directory if it doesn't exist
os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
