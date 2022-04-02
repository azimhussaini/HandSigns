"""Evaluate the model"""

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime

import tensorflow as tf
assert tf.__version__ >= "2.0"

from model.utils import Params
from model.utils import set_logger
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.model_fn import evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default="experiments/base_model", metavar='',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_SIGNS', metavar='',
                    help = "Directory containing the dataset")
parser.add_argument("trained_weights", metavar='',
                    help="Directory or file containing the trained weights")


if __name__ == '__main__':
    # Set random seed for whole graph for reproducible experiements
    tf.random.set_seed(42)

    # Load the parameters from json file stored in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json config file found at {}".format(json_path)
    params = Params(json_path)

    # Set the logger and tf summary
    time_stamp =  datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.model_dir, "test_log")
    set_logger(log_path)

     # Create input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    test_data_dir = os.path.join(data_dir, "test_signs")
    
    # Get filenames from test paths
    test_filenames = [os.path.join(test_data_dir, f) for f in os.listdir(test_data_dir)]

    # Lables will be between 0 and 5 included (6 classes in total)
    test_labels = [f[0] for f in os.listdir(test_data_dir)]

    # Specify the sizes of the datasets
    params.train_size = len(test_filenames)

    # input data
    test_input = input_fn(is_training=False, filenames=test_filenames, labels=test_labels, params=params)

    # Restore trained weights
    trained_weights_path = Path(args.trained_weights) / "weights"
    logging.info(f"Restored trained weights from {args.trained_weights}")

    # Define model
    logging.info("Testing model with trained weights...")
    evaluate(test_input, params, trained_weights_path)
