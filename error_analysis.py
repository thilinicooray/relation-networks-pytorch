import torch
from imsitu_encoder import imsitu_encoder
from imsitu_loader import imsitu_loader
from imsitu_scorer_updated_with_pred import imsitu_scorer
import json
import model_vsrl_small_finetune
import os
import utils
import time
import csv
import random
#from torchviz import make_dot
#from graphviz import Digraph



def main():

    import argparse
    parser = argparse.ArgumentParser(description="imsitu VSRL. Training, evaluation and prediction.")
    parser.add_argument("--gpuid", default=-1, help="put GPU id > -1 in GPU mode", type=int)
    #parser.add_argument("--command", choices = ["train", "eval", "resume", 'predict'], required = True)
    parser.add_argument('--resume_training', action='store_true', help='Resume training from the model [resume_model]')
    parser.add_argument('--resume_model', type=str, default='', help='The model we resume')
    parser.add_argument('--verb_module', type=str, default='', help='pretrained verb module')
    parser.add_argument('--train_role', action='store_true', help='cnn fix, verb fix, role train from the scratch')
    parser.add_argument('--finetune_verb', action='store_true', help='cnn fix, verb finetune, role train from the scratch')
    parser.add_argument('--finetune_cnn', action='store_true', help='cnn finetune, verb finetune, role train from the scratch')
    parser.add_argument('--output_dir', type=str, default='./trained_models', help='Location to output the model')
    parser.add_argument('--evaluate', action='store_true', help='Only use the testing mode')
    #todo: train role module separately with gt verbs

    args = parser.parse_args()

    batch_size = 640
    #lr = 5e-6
    lr = 0.0001
    lr_max = 5e-4
    lr_gamma = 0.1
    lr_step = 25
    clip_norm = 50
    weight_decay = 1e-4
    n_epoch = 500
    n_worker = 3

    dataset_folder = 'imsitu_data'
    imgset_folder = 'resized_256'

    train_set = json.load(open(dataset_folder + "/train.json"))
    encoder = imsitu_encoder(train_set)



    with open("gt_rn_only.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(gt_labels)

    with open("pred_rn_only.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(pred_labels)

    print('Writing predictions to file completed !')




if __name__ == "__main__":
    main()












