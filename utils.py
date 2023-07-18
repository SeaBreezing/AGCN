# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 23:12:38 2019

@author: Administrator
"""
import numpy as np
import torch
import logging

def get_logger(log_name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(formatter)

    vlog = logging.getLogger(log_name)
    vlog.setLevel(level)
    vlog.addHandler(fileHandler)

    return vlog

def cprint(words: str):
    print(f"\033[0;30;43m{words}\033[0m")


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)





