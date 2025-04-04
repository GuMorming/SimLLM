import logging
from collections import defaultdict

import torch
import random
import numpy as np
import os


def filter_and_keep_first_duplicate_j(coordinates):
    seen_j = set()  # 用来记录已经遇到的 j
    result = []  # 用来存储最终的坐标

    for coord in coordinates:
        i, j = coord
        if j not in seen_j:
            result.append(coord)  # 如果 j 没见过, 就保留
            seen_j.add(j)  # 记录已经遇到的 j

    return result


def filter_and_keep_random_duplicate_j(coordinates):
    j_to_coords = defaultdict(list)  # 创建字典来存储每个 j 对应的坐标

    # 将所有坐标按 j 分组
    for coord in coordinates:
        i, j = coord
        j_to_coords[j].append(coord)

    result = []

    # 对每个 j，如果有重复的，随机保留一个
    for j, coords in j_to_coords.items():
        result.append(random.choice(coords))  # 从重复的 j 中随机选择一个

    return result



def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


def set_logger(args):
    # 格式化日志
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
    # 打开指定的文件并将其用作日志记录流
    file_handler = logging.FileHandler(args.output_dir + "logs.log")
    file_handler.setFormatter(formatter)
    # 记录控制台的输出
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    # 记录日志
    logger = logging.getLogger(args.model_type)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    logger.setLevel(logging.INFO)
    return logger
