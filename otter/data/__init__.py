from .dataset import icrt_restructure, libero_restructure, droid_restructure
from .oxe import OXE_NAMED_MIXES
DATASET_MAPPING = {
    "libero": libero_restructure,
    "droid": droid_restructure,
    "icrt": icrt_restructure,
}

OXE_NAMES = list(OXE_NAMED_MIXES.keys())