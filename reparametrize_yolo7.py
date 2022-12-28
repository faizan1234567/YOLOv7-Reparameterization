"""script to reparametrize yolov7 and it's variants

This supports the following models
 - yolov7
 - yolov7x
 - yolov7E6
 - yolov7D6
 - yolov7E6E
 - yolov7W6
 - yolov7n (to be added soon)
 
 Reparameterization maybe important for depolyment purposes, therefore, it might be neccessary to process 
 the model file before it."""

from pathlib import Path
import sys
import os

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov7') not in sys.path:
    sys.path.append(str(ROOT / 'yolov7'))  # add yolov5 ROOT to PAT

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
import torch
import argparse
from reparameterize_models.yolov7_reparametrize import model_yolov7
from reparameterize_models.yolov7E6_reparameterize import model_yolov7E6
from reparameterize_models.yolov7E6E_reparameterize import model_yolov7E6E
from reparameterize_models.yolov7w6_reparametrize import model_yolov7w6
from reparameterize_models.yolov7x_reparametrize import model_yolov7x
from reparameterize_models.yolov7D6_reparametrize import model_yolov7D6

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default= "", type= str, help= "path to the weight file of the model")
    parser.add_argument("--model_config", default= "yolov7/cfg/deploy/yolov7", type= str, help= "path to the config file of a model")
    parser.add_argument("--save", default="", type= str, help= "path to save the processed model checkpoints")
    parser.add_argument("--yolov7_type", default=None, type= str, help= "yolov7 model to be convert i.e. yolov7, yolo7x, or other variants..")
    opt = parser.parse_args()
    return opt


def reparametrize(converter, weights, config, save):
    converter(weights, config, save)
    print('done!!')


def main():
    args = read_args()
    if args.yolov7_type == "yolov7":
        function = model_yolov7
    elif args.yolov7_type == "yolov7D6":
        function = model_yolov7D6
    elif args.yolov7_type == "yolov7E6":
        function = model_yolov7E6
    elif args.yolov7_type == "yolov7E6E":
        function = model_yolov7E6E
    elif args.yolov7_type == "yolov7W6":
        function = model_yolov7w6
    elif args.yolov7_type == "yolov7X":
        function = model_yolov7x
    else:
         print("Type didn't match, please write yolov7xx (variants names)")
         function = None

    print("Reparametrize the selected model...\n")
    print("--"*40)
    reparametrize(function, args.weights, args.model_config, args.save)
    print('Succeffully reparametrized and saved.')


if __name__ == "__main__":
    main()