NUM_WORKERS = 8

# DATA_PATH = "../input/"
# LOG_PATH = "../logs/"
# OUT_PATH = "../output/"

DATA_PATH = '/content/drive/MyDrive/2024/Learning ML (AIO)/rsna_2024/rsna_2023/input/'
LOG_PATH = '/content/drive/MyDrive/2024/Learning ML (AIO)/rsna_2024/rsna_2023/logs/'
OUT_PATH = 'content/drive/MyDrive/2024/Learning ML (AIO)/rsna_2024/rsna_2023/output/'

DEVICE = "cuda"

NEPTUNE_PROJECT = "KagglingTheo/RSNA-Abdominal-Trauma-Detection"

PATIENT_TARGETS = [
    "bowel_injury", "extravasation_injury", "kidney", "liver", "spleen"
]
CROP_TARGETS = ["kidney", "liver", "spleen"]
IMAGE_TARGETS = ["bowel_injury", "extravasation_injury"]

SEG_TARGETS = [
    "pixel_count_liver",
    "pixel_count_spleen",
    "pixel_count_left-kidney",
    "pixel_count_right-kidney",
    "pixel_count_bowel",
]
# SEG_TARGETS = [
#     "pixel_count_liver",
#     "pixel_count_spleen",
#     "pixel_count_kidney",
#     "pixel_count_bowel",
# ]

IMG_TARGETS_EXTENDED = [
    "bowel_injury",
    "extravasation_injury",
    "kidney_injury",
    "liver_injury",
    "spleen_injury",
]

WEIGHTS = {
    "bowel_injury": {
        0: 1,
        1: 2
    },
    "extravasation_injury": {
        0: 1,
        1: 6
    },
    "kidney": {
        0: 1,
        1: 2,
        2: 4
    },
    "liver": {
        0: 1,
        1: 2,
        2: 4
    },
    "spleen": {
        0: 1,
        1: 2,
        2: 4
    },
    "any_injury": {
        0: 1,
        1: 6
    },
}
