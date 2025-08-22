import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.optimizers.legacy import Adam


SEED = 1234
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Training configuration
BATCH_SIZE = 64
EPOCHS = 3000
N_WARMUP_EPOCHS = 400
KL_WEIGHT = tf.Variable(initial_value=1e-6, trainable=False)
KL_MAX = 0.0012

OPTIMIZER = Adam(learning_rate=0.0003)
PRIOR_SCALE = 0.2
DROPOUT_RATE = 0.1
NUM_MONTE_CARLO_SAMPLES = 200


# Directories
SAVE_DIR = "save_models"
RESULTS_DIR = "results"
