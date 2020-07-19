import cv2
import numpy as np
import argparse
from stages.1.blastoff.vision import parse_and_preprocess, exec_evaluator, model_infer

def parse_args():

    parser = argparse.ArgumentParser("Run inference on an input video")
    parser.add_argument("--from-stage", default=None, help='', required=True)
    parser.add_argument("--to-stage", default=None, help='', required=True)
    parser.add_argument("--run-html", default=None, help='', required=False, type=int)
    parser.add_argument("--dataset", default=None, help='', required=False, type=str)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    all_models = True

    if args.from_stage == "blastoff" and args.to_stage == "requirements":

        if all_models:
            print("Executing the blastoff stage")

            

    elif args.from_stage == "requirements" and args.to_stage == "stock":

        if all_models:
            