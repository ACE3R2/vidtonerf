import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training,training_report
import requests
import pika
import json
import time
from multiprocessing import Process
import os
import argparse
import sys
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False



def splatting_worker(args):
    rabbitmq_domain = "rabbitmq"
    credentials = pika.PlainCredentials(str(os.getenv("RABBITMQ_DEFAULT_USER")), str(os.getenv("RABBITMQ_DEFAULT_PASS")))
    parameters = pika.ConnectionParameters(
        rabbitmq_domain, 5672, "/", credentials, heartbeat=300
    )

    connection = pika.BlockingConnection(parameters)
    channel = connection.channel()
    channel.queue_declare(queue="nerf-in")
    channel.queue_declare(queue="nerf-out")

    def process_nerf_job(ch, method, properties, body):
        print("Entered consumption",flush=True)

        torch.autograd.set_detect_anomaly(args.detect_anomaly)
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

        print("\nTraining complete.",flush=True)
        ch.basic_ack(delivery_tag=method.delivery_tag)


    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="nerf-in", on_message_callback=process_nerf_job)
    channel.start_consuming()







def set_up_arguments():
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print("Optimizing " + args.model_path,flush=True)
    # Initialize system state (RNG)
    safe_state(args.quiet)


if __name__ == "__main__":

    cmdlineargs = set_up_arguments()

    nerfprocess = Process(splatting_worker, args=cmdlineargs)
