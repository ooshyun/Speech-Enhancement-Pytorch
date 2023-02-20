import os
import yaml
import json
import glob
import torch
from torch.nn.functional import pad
import numpy as np
import typing as tp
import warnings 
from omegaconf import OmegaConf

def pad_last(tensor, size:int):
    padding = [0,]*2*len(tensor.shape)
    padding[1] = size         
    return pad(tensor, pad=padding, mode="constant", value=0.)

def get_filtered_snr_file(config):
    snr_range = [0, 5]  
    
    assert isinstance(snr_range, list) and len(snr_range) == 2, f"SNR range should have minimum and maximum value"
    print(f"\t {config.dset.name} dataset SNR: {snr_range[0]} <= SNR < {snr_range[1]}")

    filtered_file_list = []
    snr_min, snr_max = snr_range

    if config.dset.name == "VoiceBankDEMAND":
        path_log = './data/VoiceBankDEMAND/DS_10283_2791/logfiles'
        text_files = glob.glob(
            os.path.join(path_log, "*trainset*")
        )
        metadata = {}
        for text_file in text_files:
            with open(text_file, "r") as tmp:
                text = tmp.read().split("\n")
                for i, t in enumerate(text):
                    text[i] = t.split(" ")
                    if len(text[i]) == 3:
                        metadata[text[i][0]] = {'type':text[i][1], 'SNR':int(text[i][2])}
        for name, values in metadata.items():
            if snr_min <= values['SNR'] and values['SNR'] < snr_max:
                filtered_file_list.append(name)

    elif config.dset.name == "Clarity":
        path_log = '/home/olive-samba/sambashare/data/Sound/clarity_ICASSP2023/clarity_CEC2_data/clarity_data/custom_metadata/scenes.train.snr.json'
        metadata = OmegaConf.to_container(OmegaConf.load(path_log))
        for scene_name, snr in metadata.items():
            if snr_min <= snr and snr < snr_max:
                filtered_file_list.append(scene_name)

    return filtered_file_list

def split_list(data:list, ratio: list):
    assert (np.sum(ratio) - 1) < 1e-5, "The summation of ratio should be 1..."
    train_ratio = ratio[0]+ratio[1]
    index = np.arange(len(data))
    np.random.shuffle(index)
    data_result = [data[i] for i in index]
    middle = int(train_ratio*len(data_result))

    return data_result[:middle], data_result[middle:]


def sample_fixed_length_data_aligned(data_list: list, sample_length, start=None):
    """sample with fixed length from several dataset

        time = [start, end]
    """
    assert isinstance(data_list, list), f"data format should be list..."
    
    if data_list[0].shape[-1] <= sample_length:
        warnings.warn(f"data length <= segment sample length, len(data_a) is {data_list[0].shape[-1]}, sample_length is {sample_length}.")
        for i in range(len(data_list)):
            data_list[i] = pad_last(data_list[i], size=sample_length-data_list[i].shape[-1])
            
    length_data = data_list[0].shape[-1]
    
    if start is None:
        start = np.random.randint(length_data - sample_length + 1)
    # print(f"Random crop from: {start}")

    end = start + sample_length

    data_result_list = [None]*len(data_list)
    for i in range(len(data_list)):
        data_result_list[i] = data_list[i][..., start:end]

    return data_result_list

    


def prepare_device(n_gpu: int, cudnn_deterministic=False):
    """
    Copy from https://github.com/haoxiangsnr/A-Convolutional-Recurrent-Neural-Network-for-Real-Time-Speech-Enhancement
    ----------------
    Choose to use CPU or GPU depend on "n_gpu".
    Args:
        n_gpu(int): the number of GPUs used in the experiment.
            if n_gpu is 0, use CPU;
            if n_gpu > 1, use GPU.
        cudnn_deterministic (bool): repeatability
            cudnn.benchmark will find algorithms to optimize training. if we need to consider the repeatability of experiment, set use_cudnn_deterministic to True
    """
    if n_gpu == 0:
        print("Using CPU in the experiment.")
        device = torch.device("cpu")
    else:
        if cudnn_deterministic:
            print("Using CuDNN deterministic mode in the experiment.")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        device = torch.device("cuda:0")

    return device


def find_folder(name: str, path: str):
    path_list = []

    for root, folders, files in os.walk(path, followlinks=True):
        if name in root:
            print("ROOT: ", root)
            print("Folder: ", folders)
            if len(files) > 10:
                files = files[:10]
            print("Files: ", files)
            print()
            path_list.append(root)

    path_list = sorted(path_list)
    return path_list

# declaring a class
class Config:
    pass

def load_yaml(path: str, *args, **kwargs) -> Config:
    with open(path, "r") as tmp:
        try:
            _dict = yaml.safe_load(tmp)
            _dict["root"] = path
            return dict2obj(_dict)

        except yaml.YAMLError as exc:
            print(exc)


def dict2obj(d):
    # checking whether object d is a
    # instance of class list
    if isinstance(d, list):
        d = [dict2obj(x) for x in d]

    # if d is not a instance of dict then
    # directly object is returned
    if not isinstance(d, dict):
        return d

    # constructor of the class passed to obj
    obj = Config()

    for k in d:
        obj.__dict__[k] = dict2obj(d[k])
    return obj


def load_json(path: str, *args, **kwargs) -> tp.Dict[str, list]:
    """Tested at study/test_save_optimizer.py"""
    with open(path, "r") as tmp:
        data: dict = json.load(tmp, *args, **kwargs)
        for key, value in data.items():
            if key == "args":
                continue
            else:
                for ival, val in enumerate(value):
                    data[key][ival] = np.array(
                        val,
                        dtype=type(val) if not isinstance(val, list) else type(val[0]),
                    )
    return data


def save_json(data: tp.Dict[str, tp.List[np.ndarray]], path: str, *args, **kwargs):
    """Tested at study/test_save_optimizer.py"""
    for key, value in data.items():
        if isinstance(value, Config):
            data[key] = obj2dict(value)

    with open(path, "w") as tmp:
        json.dump(data, tmp, cls=NumpyEncoder, *args, **kwargs)


def obj2dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key, val in obj.__dict__.items():
        if key.startswith("_"):
            continue
        element = []
        if isinstance(val, list):
            for item in val:
                element.append(obj2dict(item))
        else:
            element = obj2dict(val)
        result[key] = element
    return result


class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types
    Tested at study/test_save_optimizer.py
    Reference. https://github.com/mpld3/mpld3/issues/434#issuecomment-340255689
    """

    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
