import cv2
import numpy as np
import argparse
from torch import nn
import pickle
from stages.bblastoff.config import param_dict
from stages.bblastoff.inference import Inference
from stages.bblastoff.vision import parse_and_preprocess, exec_evaluator, model_infer

def parse_args():

    parser = argparse.ArgumentParser("Run inference on an input video")
    parser.add_argument("--from-stage", default=None, help='', required=True)
    parser.add_argument("--to-stage", default=None, help='', required=True)
    parser.add_argument("--run-html", default=None, help='', required=False, type=int)
    parser.add_argument("--dataset", default=None, help='', required=False, type=str)
    parser.add_argument("--imagesets_dir", default=None, help='', required=False, type=str)
    parser.add_argument("--input_graph", default=None, help='', required=False, type=str)
    parser.add_argument("--input_weights", default=None, help='', required=False, type=str)
    parser.add_argument("--in_blob_name", default=None, help='', required=False, type=str)
    parser.add_argument("--out_blob_name", default=None, help='', required=False, type=str)
    parser.add_argument("--need_reshape", default=None, help='', required=False, type=str)
    parser.add_argument("--predict_net", default=None, help='', required=False, type=str)
    parser.add_argument("--init_net", default=None, help='', required=False, type=str)
    parser.add_argument("--is_run_init", default=1, help='', required=False, type=int)
    parser.add_argument("--is_create_net", default=1, help='', required=False, type=int)
    parser.add_argument("--input_model", default=1, help='', required=False, type=str)
    parser.add_argument("--encoder", default=None, help='', required=False, type=str)
    parser.add_argument("--output_blob", default=1, help='', required=False, type=int)
    parser.add_argument("--param_args", default={}, help='', required=False, type=dict)
    parser.add_argument("--model_file", default=None, help='', required=False, type=str)
    parser.add_argument("--model", default=None, help='', required=False, type=str)
    parser.add_argument("--device", default="CPU", help='', required=False, type=str)
    parser.add_argument("--cpu_extension", default=None, help='', required=False, type=str)
    parser.add_argument("--batch_size", default=1, help='', required=False, type=int)
    parser.add_argument("--input_blob", default=None, help='', required=False, type=str)

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    all_models = True
    models = ['caffe', 'caffe2', 'onnx', 'openvino', 'pytorch', 'tensorflow']

    if args.from_stage == "blastoff" and args.to_stage == "requirements":

        if all_models:
            print("Executing the blastoff stage")

            param_dict['data_location'] = args.dataset
            param_dict['imagesets_dir'] = args.imagesets_dir

            args.param_args = param_dict
            args.input_blob = cv2.imread(args.input_blob)

            inference_object = Inference(framework=models[0])
            inference_object.create_instance(input_graph=args.input_graph, input_weights=args.input_weights, 
            in_blob_name=args.in_blob_name, 
            out_blob_name=args.out_blob_name, need_reshape=args.need_reshape)
            inference_object.load_model()
            infer_object = model_infer(param_dict, inference_object, factor=1e-1, output_log="output_"+str(models[0])+"_log.log")
            infer_object.run_inference({'input_blob': args.input_blob})

            print("Created " + "output_"+str(models[0])+"_log.log file")

            inference_object = Inference(framework=models[1])
            inference_object.create_instance(args.predict_net, init_net=args.init_net, 
            is_run_init=args.is_run_init, is_create_net=args.is_create_net)
            inference_object.load_model()
            infer_object = model_infer(param_dict, inference_object, factor=1e-1, output_log="output_"+str(models[0])+"_log.log")
            infer_object.run_inference({'input_blob': args.input_blob})

            print("Created " + "output_"+str(models[1])+"_log.log file")

            inference_object = Inference(framework=models[2])
            inference_object.create_instance(args.input_model)
            inference_object.load_model()
            infer_object = model_infer(param_dict, inference_object, factor=1e-1, output_log="output_"+str(models[0])+"_log.log")
            infer_object.run_inference({'input_blob': args.input_blob})

            print("Created " + "output_"+str(models[2])+"_log.log file")

            # inference_object = Inference(framework=models[3])
            # inference_object.load_model()
            # infer_object = model_infer(param_dict, inference_object, factor=1e-1, output_log="output_"+str(models[0])+"_log.log")
            # infer_object.run_inference({'input_blob': args.input_blob})

            # print("Created " + "output_"+str(models[3])+"_log.log file")

            # inference_object = Inference(framework=models[4])
            # encoder = pickle.load(args.encoder)
            # inference_object.create_instance(encoder, output_blob)
            # inference_object.load_model(args.encoder, args.model_file)
            # infer_object = model_infer(param_dict, inference_object, factor=1e-1, output_log="output_"+str(models[0])+"_log.log")
            # infer_object.run_inference({'input_blob': args.input_blob})

            # print("Created " + "output_"+str(models[4])+"_log.log file")

            inference_object = Inference(framework=models[5])
            inference_object.create_instance(param_args=args.param_args)
            inference_object.load_model(args.model_file)
            infer_object = model_infer(param_dict, inference_object, factor=1e-1, output_log="output_"+str(models[0])+"_log.log")
            infer_object.run_inference({'input_blob': args.input_blob})

            print("Created " + "output_"+str(models[5])+"_log.log file")

    elif args.from_stage == "requirements" and args.to_stage == "stock":

        if all_models:
            print("Executing the routes of requirements stage")

            param_dict['data_location'] = args.dataset
            param_dict['imagesets_dir'] = args.imagesets_dir

