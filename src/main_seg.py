import os
import time
import torch
import warnings
import argparse
import pandas as pd

from data.preparation import prepare_seg_data
from util.torch import init_distributed
from util.logger import create_logger, save_config, prepare_log_folder, init_neptune

from params import DATA_PATH, SEG_TARGETS


def parse_args():
    """
    Parses arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fold",
        type=int,
        default=-1,
        help="Fold number",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="Device number",
    )
    parser.add_argument(
        "--log_folder",
        type=str,
        default="",
        help="Folder to log results to",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0,
        help="learning rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Batch size",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.,
        help="Weight decay",
    )
    return parser.parse_args()


class Config:
    """
    Parameters used for training
    """
    # General
    seed = 42
    verbose = 1
    device = "cuda"
    save_weights = True

    # Data
    resize = (384, 384)
    crop = True
    aug_strength = 1
    for_classification = False

    # k-fold
    k = 4
    folds_file = f"../input/folds_{k}.csv"
    selected_folds = [0]  # , 1, 2, 3]

    # Model
    name = "tf_efficientnetv2_s"  # "resnet18d"
    decoder_name = "Unet"
    pretrained_weights = None
    increase_stride = False  # True
    use_cls = False

    num_classes = 5
    num_classes_aux = 5
    n_channels = 3

    # Training    
    loss_config = {
        "name": "ce",
        "smoothing": 0,
        "activation": "sigmoid",
        "aux_loss_weight": 0.,
        "name_aux": "bce",
        "smoothing_aux": 0,
        "activation_aux": "sigmoid",
        "num_classes": num_classes,
    }

    data_config = {
        "batch_size": 32,
        "val_bs": 32,
        "mix": "mixup",
        "sched": False,
        "mix_proba": 0.,
        "mix_alpha": 4.,
        "additive_mix": False,
        "num_classes": num_classes,
        "num_workers": 8,
    }

    optimizer_config = {
        "name": "AdamW",
        "lr": 5e-4,
        "warmup_prop": 0.,
        "betas": (0.9, 0.999),
        "max_grad_norm": 10.,
        "weight_decay": 0.,
    }

    epochs = 5

    use_fp16 = True
    verbose = 1
    verbose_eval = 50
    
    fullfit = True
    n_fullfit = 1


if __name__ == "__main__":
    warnings.simplefilter("ignore", UserWarning)

    config = Config
    init_distributed(config)

    if config.local_rank == 0:
        print("\nStarting !")
    args = parse_args()

    if not config.distributed:
        device = args.fold if args.fold > -1 else args.device
        time.sleep(device)
        print("Using GPU ", device)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        assert torch.cuda.device_count() == 1

    log_folder = args.log_folder
    if not log_folder:
        from params import LOG_PATH

        if config.local_rank == 0:
            log_folder = prepare_log_folder(LOG_PATH)
            print(f'\n -> Logging results to {log_folder}\n')

    if args.model:
        config.name = args.model

    if args.epochs:
        config.epochs = args.epochs

    if args.lr:
        config.optimizer_config["lr"] = args.lr

    if args.weight_decay:
        config.optimizer_config["weight_decay"] = args.weight_decay

    if args.batch_size:
        config.data_config["batch_size"] = args.batch_size
        config.data_config["val_bs"] = args.batch_size

    run = None
    if config.local_rank == 0:
#         run = init_neptune(config, log_folder)

        if args.fold > -1:
            config.selected_folds = [args.fold]
            create_logger(directory=log_folder, name=f"logs_{args.fold}.txt")
        else:
            create_logger(directory=log_folder, name="logs.txt")

        save_config(config, log_folder + "config.json")
        if run is not None:
            run["global/config"].upload(log_folder + "config.json")

    if config.local_rank == 0:
        print("Device :", torch.cuda.get_device_name(0), "\n")

        print(f"- Model {config.name}")
        print(f"- Epochs {config.epochs}")
        print(
            f"- Learning rate {config.optimizer_config['lr']:.1e}   (n_gpus={config.world_size})"
        )
        print("\n -> Training\n")

        
    df = prepare_seg_data(data_path=DATA_PATH)
#     df = df[df['patient_id'] == 10217].reset_index(drop=True)
#     df = df[df[[c for c in df.columns if "norm" in c]].max(1) > 0.1].reset_index(drop=True)

    from training.main_seg import k_fold
    k_fold(config, df, log_folder=log_folder, run=run)

    if config.local_rank == 0:
        print("\nDone !")
