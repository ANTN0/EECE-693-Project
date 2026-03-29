"""
config.py - All project settings in one place.

Change MODEL_VARIANT (1-4) to switch between model versions:
    Model 1: U-Net, RGB output, L1 loss (baseline)
    Model 2: U-Net, Lab output, L1 + L2 loss (Lab color space)
    Model 3: Model 2 + VGG perceptual loss
    Model 4: Model 3 + reference-guided cross-attention
"""

import os

# =============================================================================
# MODEL VARIANT — change this to train different models
# =============================================================================
MODEL_VARIANT = 1

# =============================================================================
# PATHS
# =============================================================================
DATA_ROOT = r"D:\Pictures for Project"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# Training folders: color images only (we need ground-truth colors)
TRAIN_SUBFOLDERS = [
    "Arab and Lebanese Diaspora",
    "Baptism",
]

# Test/inference folders: already damaged or B&W images (never used for training)
TEST_SUBFOLDERS = [
    "Studio",
    "alfred archive",
]

# =============================================================================
# IMAGE SETTINGS
# =============================================================================
IMAGE_SIZE = 128          # Training crop size (128x128 fits in 6GB VRAM)
IMAGE_CHANNELS = 1        # Grayscale input
OUTPUT_CHANNELS = 3       # RGB (model 1) or Lab (models 2-4), both are 3 channels

# =============================================================================
# SYNTHETIC DEGRADATION
# =============================================================================
DAMAGE_TYPES = [
    "scratches",
    "noise",
    "blur",
    "aging",
    "water_damage",
    "fire_damage",
    "wear_and_tear",
    "mold_foxing",
    "light_leak",
    "crease",
]
NUM_DAMAGE_TYPES = len(DAMAGE_TYPES)
DEGRADATION_PROB = 0.4
MAX_DAMAGES_PER_IMAGE = 3
CLEAN_RATIO = 0.3         # 30% of images kept clean (identity mapping)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
# U-Net encoder channels (scaled down from proposal for 6GB VRAM)
# Proposal: 64->128->256->512, bottleneck 1024
# Ours:     32->64->128->256,  bottleneck 512
ENCODER_CHANNELS = [32, 64, 128, 256]
BOTTLENECK_CHANNELS = 512
DROPOUT_RATE = 0.3

# =============================================================================
# TRAINING
# =============================================================================
BATCH_SIZE = 8            # Reduce to 4 for Model 4 if you run out of VRAM
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Loss weights
PIXEL_LOSS_WEIGHT = 1.0           # L1 loss (all models)
L2_LOSS_WEIGHT = 0.5              # L2 loss on ab channels (models 2-4)
CLASSIFICATION_LOSS_WEIGHT = 0.5  # Damage classification (all models)
PERCEPTUAL_LOSS_WEIGHT = 0.1      # VGG perceptual loss (models 3-4)

# =============================================================================
# REFERENCE GUIDANCE (Model 4 only)
# =============================================================================
NUM_REFERENCES = 3        # K=3 reference images, as per proposal

# =============================================================================
# MISC
# =============================================================================
USE_AMP = True            # Mixed precision training (saves VRAM)
MAX_IMAGES = None         # None = use all images; set to e.g. 1000 for quick tests
VAL_SPLIT = 0.1           # 10% validation
NUM_WORKERS = 2           # DataLoader workers (keep low on Windows)
SEED = 42
DEVICE = "cuda"

LOG_EVERY = 50            # Print loss every N batches
SAVE_EVERY = 5            # Save checkpoint every N epochs
SAMPLE_EVERY = 1          # Save sample outputs every N epochs
NUM_SAMPLES = 8           # Number of samples to save per epoch
