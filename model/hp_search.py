"""Hyperparameter Search"""

from cmath import log
import numpy as np
import time
import os
import logging
import shutil

import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from model.model_fn import build_model
from model.model_fn import train_step
from model.model_fn import test_step

# Setup hyperparameters
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-1, 1e-3, 1e-5]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.4]))
HP_BN_M = hp.HParam('bn_momentum', hp.Discrete([0.1, 0.4, 0.8]))


def hp_search(train_data, val_data, params, epochs, hp_log_path, clear_log_history=True):
    """Setup hyperparameter search by computing training and validation step on each set of hyperparameters

    Arguments:
        train_data: (tf.data.Dataset) training dataset
        val_data: (tf.data.Dataset) validation dataset
        params: (Params) hyperparameter of the model
        tf_log_path: (str) tf summary logs directory
        clear_log_history: (bool) clear tf history to restart. Default is 'True'
    """

    run_id = 0

    # Clear HP log directory
    if clear_log_history and os.path.isdir(hp_log_path):
        shutil.rmtree(hp_log_path)

    for learning_rate in HP_LR.domain.values:
        for dropout in HP_DROPOUT.domain.values:
            for bn in HP_BN_M.domain.values:
                hparams = {
                    HP_LR: learning_rate,
                    HP_DROPOUT: dropout,
                    HP_BN_M: bn
                }

                params.bn_momentum = hparams[HP_BN_M]
                params.dropout_rate = hparams[HP_DROPOUT]


                # HP log dir:
                run_dir = str(learning_rate) + "lr_" + str(dropout) + "dropout_", str(bn) + "bn"
                log_path = os.path.join(hp_log_path, str(run_dir))
                
                # Instantiate an optimizer, loss, acc_metrics
                model_metrices = {"optimizer": keras.optimizers.Adam(learning_rate=hparams[HP_LR]),
                        "loss_fn": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                        "train_loss": tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
                        "test_loss": tf.keras.metrics.Mean('test_loss', dtype=tf.float32),
                        "train_acc_metric": keras.metrics.SparseCategoricalAccuracy(),
                        "val_acc_metric": keras.metrics.SparseCategoricalAccuracy()}

                # Get model
                model = build_model(params)
                
                logging.info(f"---Starting Run: {run_id}:")
                [logging.info(f"{h.name}: {hparams[h]}") for h in hparams]

                for epoch in range(epochs):
                    start_time = time.time()
                    # Iterate over training batches in dataset
                    for x_batch_train, y_batch_train in train_data:
                        logits = train_step(model, x_batch_train, y_batch_train, model_metrices)
                    # Display metrics at the end of each epoch
                    train_acc = model_metrices['train_acc_metric'].result()
                    train_loss = model_metrices['train_loss'].result()

                    for x_batch_val, y_batch_val in val_data:
                        test_step(model, x_batch_val, y_batch_val, model_metrices)

                    val_acc = model_metrices['val_acc_metric'].result()
                    val_loss = model_metrices['test_loss'].result()
                    
                    # Reset metrics after each epoch
                    model_metrices['train_acc_metric'].reset_states()
                    model_metrices['val_acc_metric'].reset_states()
                    model_metrices['train_loss'].reset_states()
                    model_metrices['test_loss'].reset_states()


                with tf.summary.create_file_writer(log_path).as_default():
                    hp.hparams(hparams)
                    # accuracy = val_acc
                    tf.summary.scalar("accuracy", val_acc, step=epoch)
                
                logging.info(f"train loss: {train_loss:.4f}, train acc: {train_acc:.4f}")
                logging.info(f"val loss: {val_loss:.4f}, val acc: {val_acc:.4f}")
                logging.info(f"Time taken: {(time.time() - start_time):.2f}")

                run_id += 1

