"""Shared pieces of the static sign recognizer: paths, dataset sources and splits."""
import re
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
HAND_MODEL = PROJECT_DIR / "models" / "hand_landmarker.task"
CACHE_DIR = Path.home() / "datasets" / "signwave_cache"  # crops + landmarks, kept out of OneDrive
CKPT_DIR = PROJECT_DIR / "checkpoints" / "recognition"

# Whole signers kept out of training. Validation picks the best epoch, test is touched once:
# both measure how the model does on people it has never seen, which is what the live demo needs.
VAL_GROUPS = ("surrey4", "massey4")
TEST_GROUPS = ("surrey5", "office", "massey5")


def source_group(filename):
    """Source dataset / signer of a combine_asl image, from its file name."""
    m = re.match(r"color_\d+_\d+(?: \((\d+)\))?\.png$", filename)
    if m:  # ASL Fingerspelling (Surrey): 5 signers merged, " (k)" suffix = signer k
        return f"surrey{m.group(1) or 1}"
    m = re.match(r"hand(\d+)_", filename)
    if m:  # Massey University: black background, hand1..hand5
        return f"massey{m.group(1)}"
    if re.fullmatch(r"[A-Za-z]\d+\.jpg", filename):
        return "kaggle"  # Kaggle ASL Alphabet: one signer, one background
    if re.fullmatch(r"[A-Za-z]\d+(?: \(\d+\)| - Copy)\.jpg", filename):
        return "office"  # another signer photographed in an office
    return "other"


def split_indices(groups):
    groups = np.asarray(groups)
    val = np.isin(groups, VAL_GROUPS)
    test = np.isin(groups, TEST_GROUPS)
    return np.flatnonzero(~val & ~test), np.flatnonzero(val), np.flatnonzero(test)
