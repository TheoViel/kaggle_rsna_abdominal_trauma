NUM_WORKERS = 8

DATA_PATH = "../input/"
LOG_PATH = "../logs/"
OUT_PATH = "../output/"

DEVICE = "cuda"

NEPTUNE_PROJECT = "KagglingTheo/RSNA-Abdominal-Trauma-Detection"

PATIENT_TARGETS = ["bowel_injury", "extravasation_injury", "kidney", "liver", "spleen"]
IMAGE_TARGETS = ["bowel_injury", "extravasation_injury"]
SEG_TARGETS = ['pixel_count_liver', 'pixel_count_spleen', 'pixel_count_left-kidney', 'pixel_count_right-kidney', 'pixel_count_bowel']

WEIGHTS = {
    'bowel_injury': {0: 1, 1: 2},
    'extravasation_injury': {0: 1, 1: 6},
    'kidney': {0: 1, 1: 2, 2: 4},
    'liver': {0: 1, 1: 2, 2: 4},
    'spleen': {0: 1, 1: 2, 2: 4},
    'any_injury': {0: 1, 1: 6},
}
