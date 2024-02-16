import argparse
import yaml
import torch
import time
import numpy as np
from collections import defaultdict, OrderedDict
from src.model_handler import ModelHandler

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def main(config):
    print_config(config)
    set_random_seed(config['seed'])
    model = ModelHandler(config)
    f1_mac_test, f1_1_test, f1_0_test, auc_test, gmean_test = model.train()
    print("F1-Macro: {}".format(f1_mac_test))
    print("AUC: {}".format(auc_test))
    print("G-Mean: {}".format(gmean_test))

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config = yaml.load(setting, Loader=yaml.FullLoader)
    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', required=True, type=str, help='path to the config file')
    parser.add_argument('--multi_run', action='store_true', help='flag: multi run')
    args = vars(parser.parse_args())
    return args

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val = config[key]
        keystr = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")

if __name__ == '__main__':
    cfg = get_args()
    config = get_config(cfg['config'])
    main(config)
