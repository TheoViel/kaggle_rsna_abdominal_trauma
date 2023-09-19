import os
import time
import json
import torch
import warnings
import pandas as pd

from data.preparation import prepare_data
from util.torch import init_distributed
from inference.extract_features import Config, kfold_inference
from params import DATA_PATH


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

#     EXP_FOLDER = "../logs/2023-09-06/0/"
#     EXP_FOLDER = "../logs/2023-09-06/5/"  # seg
#     EXP_FOLDER =  "../logs/2023-09-15/37/"
    EXP_FOLDER = "../logs/2023-09-19/5/"
#     EXP_FOLDER = "../logs/2023-09-19/10/"  # seg efficient

    config = Config(json.load(open(EXP_FOLDER + "config.json", "r")))
    init_distributed(config)

    if config.local_rank == 0:
        print("\nStarting !")

    assert config.distributed

    df_patient, df_img = prepare_data(DATA_PATH)
    
    USE_FP16 = True
    SAVE = True

    if config.local_rank == 0:
        print(f"\n- Model {config.name}")
        print(f"- Exp folder {EXP_FOLDER}")
        print("\n -> Extracting features")

    kfold_inference(
        df_patient,
        df_img,
        EXP_FOLDER,
        use_fp16=USE_FP16,
        save=SAVE,
        distributed=True,
        config=config,
    )

    if config.local_rank == 0:
        print("\nDone !")
