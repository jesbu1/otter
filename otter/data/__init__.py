from .dataset import dexycb_restructure, hoi4d_restructure, icrt_restructure, libero_restructure, droid_restructure 
from .oxe import OXE_NAMED_MIXES
DATASET_MAPPING = {
    "dexycb": dexycb_restructure,
    "libero": libero_restructure,
    "droid": droid_restructure,
    "hoi4d": hoi4d_restructure,
    "icrt": icrt_restructure,
}

OXE_NAMES = list(OXE_NAMED_MIXES.keys())