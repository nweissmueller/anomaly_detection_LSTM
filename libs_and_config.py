# import standard libs
import numpy as np
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
import copy

# import sklearn functionality
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, KFold

# import pytorch
import torch
from torch import nn, optim
import torch.nn.functional as F

# configure layout
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format='retina'")

# plot settings
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
COLORS_PALETTE = ["#FF006D", "#808080"]
sns.set_palette(sns.color_palette(COLORS_PALETTE))

# randomization settings
RANDOM_SEED = 24
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
