import os
import torch
import random
import numpy as np
import json


def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent = '\t')


def read_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def create_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def add_txt(path, string):
    with open(path, 'a+') as f:
        f.write(string + '\n')


def log_print(message, path):
    print(message)
    add_txt(path, message)


class Average_Meter:
    def __init__(self, keys):
        self.keys = keys
        self.clear()
    
    def add(self, dic):
        for key, value in dic.items():
            self.data_dic[key].append(value)

    def get(self, keys=None, clear=False):
        if keys is None:
            keys = self.keys
        
        dataset = [float(np.mean(self.data_dic[key])) for key in keys]
        if clear:
            self.clear()

        if len(dataset) == 1:
            dataset = dataset[0]
            
        return dataset
    
    def clear(self):
        self.data_dic = {key : [] for key in self.keys}
    

