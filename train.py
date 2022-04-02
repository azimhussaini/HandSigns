"""Train the model"""

import argparse
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.0"

from model.utils import Params
from model.utils import set_logger
from model.input_fn import input_fn
from model.model_fn import model_fn
from model.hp_search import hp_search


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default="experiments/base_model", metavar='',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/64x64_SIGNS', metavar='',
                    help = "Directory containing the dataset")
parser.add_argument("--restore_from", default=None, metavar='',
                    help="Optional, directory or file containing the weights to reload before training")
parser.add_argument("-H", "--hp_search", default='False', metavar='',
                    help="Set it to True if you want to run hyperparameters search using tensorboard")

if __name__ == '__main__':
    # Set random seed for whole graph for reproducible experiements
    tf.random.set_seed(42)

    # Load the parameters from json file stored in model_dir
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json config file found at {}".format(json_path)
    params = Params(json_path)

    # Check if we are not overwriting some previous experiment
    model_dir_has_best_weights = os.path.isdir(os.path.join(args.model_dir, "model_weights", "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # check restore directory
    if args.restore_from is not None:
        restore_path = Path(args.restore_from) / "weights"
    else:
        restore_path = None

    # Set the logger and tf summary
    time_stamp =  datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = os.path.join(args.model_dir, "train_log")
    print(log_path)
    set_logger(log_path)
    tf_summary_log_path = os.path.join(args.model_dir, "tf_logs", time_stamp)
    hp_search_log_path = os.path.join(args.model_dir, "hp_search")


    # Create input data pipeline
    logging.info("Creating the datasets...")
    data_dir = args.data_dir
    train_data_dir = os.path.join(data_dir, "train_signs")
    dev_data_dir = os.path.join(data_dir, "dev_signs")

    # Get filenames from train & dev paths
    train_filenames = [os.path.join(train_data_dir, f) for f in os.listdir(train_data_dir)]
    eval_filenames = [os.path.join(dev_data_dir, f) for f in os.listdir(dev_data_dir)]

    # Lables will be between 0 and 5 included (6 classes in total)
    train_labels = [f[0] for f in os.listdir(train_data_dir)]
    eval_labels = [f[0] for f in os.listdir(dev_data_dir)]

    # Specify the sizes of the datasets
    params.train_size = len(train_filenames)
    params.eval_size = len(eval_filenames)

    train_input = input_fn(True, train_filenames, train_labels, params)
    eval_input = input_fn(False, eval_filenames, eval_labels, params)


    # Training and evaluation
    if args.hp_search=="True":
        logging.info("Starting hyperparameter search for {params.hp_search_epochs} epoch(s)")
        hp_search(train_input, eval_input, params, epochs=params.hp_search_epochs, hp_log_path=hp_search_log_path)
    else:
        logging.info(f"Starting training for {params.num_epochs} epoch(s)")
        model = model_fn(train_input, eval_input, params, tf_summary_log_path, restore_weights=restore_path)
        save_weights_path = save_weights_path = Path(args.model_dir) / "model_weights" / time_stamp / "weights"
        logging.info(f"Saving weights at {save_weights_path}")
        model.save_weights(save_weights_path)

        





