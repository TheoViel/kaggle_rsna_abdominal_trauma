import os
import re
import sys
import math
import json
import shutil
import neptune
import datetime
import subprocess
import numpy as np

from params import NEPTUNE_PROJECT


class Config:
    """
    Placeholder to load a config from a saved json
    """

    def __init__(self, dic):
        for k, v in dic.items():
            setattr(self, k, v)


class Logger(object):
    """
    Simple logger that saves what is printed in a file
    """

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()

    def close(self):
        pass


def create_logger(directory="", name="logs.txt"):
    """
    Creates a logger to log output in a chosen file

    Args:
        directory (str, optional): Path to save logs at. Defaults to "".
        name (str, optional): Name of the file to save the logs in. Defaults to "logs.txt".
    """
    log = open(directory + name, "a", encoding="utf-8")
    file_logger = Logger(sys.stdout, log)

    sys.stdout = file_logger
    sys.stderr = file_logger


def prepare_log_folder(log_path):
    """
    Creates the directory for logging.
    Logs will be saved at log_path/date_of_day/exp_id.

    Args:
        log_path (str): Directory
    Returns:
        str: Path to the created log folder
    """
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        os.mkdir(log_today)

    exps = []
    for f in os.listdir(log_today):
        try:
            exps.append(int(f))
        except Exception:
            continue
    exp_id = np.max(exps) + 1 if len(exps) else 0

    log_folder = log_today + f"{exp_id}/"

    assert not os.path.exists(log_folder), "Experiment already exists"
    os.mkdir(log_folder)

    return log_folder


def get_last_log_folder(log_path):
    """
    Get the path to the most recent log folder from today or the previous day.

    Args:
        log_path (str): The base path to the log folders.

    Returns:
        str: The path to the most recent log folder.
    """
    today = str(datetime.date.today())
    log_today = f"{log_path}{today}/"

    if not os.path.exists(log_today):
        today = today - datetime.timedelta(days=1)
        log_today = f"{log_path}{today}/"

    exps = []
    for f in os.listdir(log_today):
        try:
            exps.append(int(f))
        except Exception:
            continue
    exp_id = np.max(exps)

    log_folder = log_today + f"{exp_id}/"
    return log_folder


def save_config(config, path):
    """
    Saves a config as a json and pandas dataframe.

    Args:
        config (Config): Config.
        path (str): Path to save at.

    Returns:
        pandas dataframe: Config as a dataframe.
    """
    dic = config.__dict__.copy()
    del (dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"])

    if not path.endswith(".json"):
        path += ".json"

    with open(path, "w") as f:
        json.dump(dic, f)


def init_neptune(config, log_folder):
    """
    Initializes the neptune run.

    Args:
        config (Config): Config.
        log_folder (str): Log folder.

    Returns:
        Neptune run: Run.
    """
    print()
    run = neptune.init_run(
        project=NEPTUNE_PROJECT, git_ref=neptune.types.GitRef.DISABLED
    )

    run["global/log_folder"] = log_folder

    dic = config.__dict__.copy()
    del (dic["__doc__"], dic["__module__"], dic["__dict__"], dic["__weakref__"])
    for k in dic.keys():
        if not isinstance(dic[k], (dict, int, float, str)):
            dic[k] = str(dic[k])

    run["global/parameters/"] = dic
    print()
    return run


def get_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            # skip if it is symbolic link
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)

    total_size = total_size / math.pow(1024, 3)
    return total_size


def upload_to_kaggle(folders, directory, dataset_name, update_folders=True):
    """
    TODO
    Args:
        folders (_type_): _description_
        directory (_type_): _description_
        dataset_name (_type_): _description_
    """
    os.makedirs(directory, exist_ok=True)

    def custom_ignore(src, names):
        folders_to_skip = ["masks"]
        return [name for name in names if name in [folders_to_skip]]

    for folder in folders:
        print(f"- Copying {folder} ...")
        name = "_".join(folder[:-1].split("/")[-2:])
        try:
            shutil.copytree(
                folder,
                directory + name,
                ignore=lambda src, names: [n for n in names if "mask" in n],
            )
        except FileExistsError:
            if update_folders:
                shutil.rmtree(directory + name)
                shutil.copytree(
                    folder,
                    directory + name,
                    ignore=lambda src, names: [n for n in names if "mask" in n],
                )
        for i in range(4):
            try:
                os.remove(f"{directory + name}/fts_val_{i}.npy")
            except Exception:
                pass

    print(f"\nDataset size : {get_size(directory):.3f} Go")

    if os.path.exists(directory + "dataset-metadata.json"):
        # Update version
        print("- Update existing dataset !")
        command = f'kaggle d version -m "update" -p {directory} --dir-mode zip'
    else:
        print("- Create new dataset ! ")
        # Create dataset-metadata.json
        with open(directory + "dataset-metadata.json", "w") as f:
            slug = re.sub(" ", "-", dataset_name.lower())
            dic = {
                "title": f"{dataset_name}",
                "id": f"theoviel/{slug}",
                "licenses": [{"name": "CC0-1.0"}],
            }
            json.dump(dic, f)

        command = f"kaggle d create -p {directory} --dir-mode zip"

    # Upload dataset
    print("- Uploading ...")
    try:
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()
        print("\nOutput :\n", output)
        print("\nError :\n", error)
    except Exception:
        print("\nUpload failed, Run command manually :\n", command)
