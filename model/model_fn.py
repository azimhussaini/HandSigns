"""Define the model"""

import enum
import logging
# from numpy import dtype
import numpy as np
import time
import os

import tensorflow as tf
from tensorflow import keras
from tensorboard.plugins.hparams import api as hp
from model.input_fn import input_fn

from model.utils import plot_to_image, image_grid, get_confusion_matrix, plot_confusion_matrix



def build_model(params):
    """Compute logits for the model
    
    Args:
        params: (Params) hyperparameters
    
    Returns:
        output: (tf.Keras.Model) model
    """
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum
    size = params.image_size
    num_labels = params.num_labels
    dropout_rate = params.dropout_rate

    channels = [num_channels, num_channels*2, num_channels*4, num_channels*8]
    
    inputs = keras.Input(shape=(size, size, 3))
    outputs = inputs
    for i, c in enumerate(channels):
        outputs = keras.layers.Conv2D(filters=c, kernel_size=(3, 3), padding="same", name="Conv2D_"+str(i))(outputs)
        if params.use_batch_norm:
            outputs = keras.layers.BatchNormalization(momentum=bn_momentum, name="BN_"+str(i))(outputs)
        outputs = tf.nn.relu(outputs)
        outputs = tf.keras.layers.MaxPool2D(pool_size=(2,2), name="MP_"+str(i))(outputs)
    outputs = keras.layers.Flatten()(outputs)
    outputs = keras.layers.Dense(512, activation='relu')(outputs)
    outputs = keras.layers.Dense(128, activation='relu')(outputs)
    outputs = keras.layers.Dropout(0.2)(outputs)
    outputs = keras.layers.Dense(64, activation='relu')(outputs)
    if params.use_dropout:
        outputs = keras.layers.Dropout(dropout_rate)(outputs)
    outputs = keras.layers.Dense(num_labels, activation="softmax")(outputs)
    model = keras.Model(inputs=inputs, outputs=outputs, name="Signs_Model")

    return model


# @tf.function
def train_step(model, x, y, metrices):
    """Compute training steps
    
    Arguments:
        model: (tf.Keras.Model) model
        x: (tf.tensor) input data X
        y: (tf.tensor) labels
        metrices: (dict) dictionary of loss function, optimizers and metrics
    
    Return:
        logits
    """

    loss_fn = metrices["loss_fn"]
    optimizer = metrices['optimizer']
    train_acc_metric = metrices["train_acc_metric"]
    train_loss = metrices['train_loss']

    with tf.GradientTape() as tape:
        # run forward pass of the layer
        logits = model(x, training=True)
        # calculate loss value
        loss_value = loss_fn(y, logits)

    # retrieve gradients
    grads = tape.gradient(loss_value, model.trainable_weights)
    # run gradient descent to update the value of variable
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    # update training metrics
    train_acc_metric.update_state(y, logits)
    # update mean loss
    train_loss(loss_value)
    return logits


# @tf.function
def test_step(model, x, y, metrices):
    """Compute test steps
    
    Arguments:
        model: (tf.Keras.Model) model
        x: (tf.tensor) input data X
        y: (tf.tensor) labels
        metrices: (dict) dictionary of loss function, optimizers and metrics
    """
    val_acc_metric = metrices["val_acc_metric"]
    loss_fn = metrices["loss_fn"]
    test_loss = metrices['test_loss']

    # run forward pass of the layer
    val_logits = model(x, training=False)
    # update val metrics
    val_acc_metric.update_state(y, val_logits)
    # update mean loss
    loss_value = loss_fn(y, val_logits)
    test_loss(loss_value)



def model_fn(train_data, val_data, params, tf_log_path, reuse=False):
    """Compute training and validation steps. Also setup tensorboard for visualizations.
    
    Arguments:
        train_data: (tf.data.Dataset) training dataset
        val_data: (tf.data.Dataset) validation dataset
        params: (Params) hyperparameter of the model
        tf_log_path: (str) tf summary logs directory
    """
    # Instantiate an optimizer, loss, acc_metrics
    model_metrices = {"optimizer": keras.optimizers.Adam(learning_rate=params.learning_rate),
            "loss_fn": keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            "train_loss": tf.keras.metrics.Mean('train_loss', dtype=tf.float32),
            "test_loss": tf.keras.metrics.Mean('test_loss', dtype=tf.float32),
            "train_acc_metric": keras.metrics.SparseCategoricalAccuracy(),
            "val_acc_metric": keras.metrics.SparseCategoricalAccuracy()}

    # Setup summary writers
    train_log_dir = os.path.join(tf_log_path, "train")
    test_log_dir = os.path.join(tf_log_path, "test")
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # Get model
    model = build_model(params)
    
    step = 0
    epochs = params.num_epochs
    # epochs = 1
    for epoch in range(epochs):
        logging.info(f"\nStart of each epoch {epoch}")
        start_time = time.time()
        # Setup confusion matrix
        confusion = np.zeros((params.num_labels, params.num_labels))

        # Iterate over training batches in dataset
        for batch_idx, (x_batch_train, y_batch_train) in enumerate(train_data):
            logits = train_step(model, x_batch_train, y_batch_train, model_metrices)

            # Visualize figure
            if step <= 6:
                figure = image_grid(x_batch_train, y_batch_train)
                with train_summary_writer.as_default():
                    tf.summary.image("Visualize Images", plot_to_image(figure), step=step)
                    step += 1

            # update confusion matrix
            confusion += get_confusion_matrix(y_batch_train, logits, params.num_labels)
            
        # Display metrics at the end of each epoch
        train_acc = model_metrices['train_acc_metric'].result()
        train_loss = model_metrices['train_loss'].result()

        # log metrics (accuracy and loss) during training
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss, step=epoch)
            tf.summary.scalar('accuracy', train_acc, step=epoch)
            tf.summary.image("Confusion Matrix", plot_confusion_matrix(confusion / batch_idx, params.num_labels), step=epoch)
        logging.info(f"Training loss over epoch {epoch}: {float(train_loss):.4f}")
        logging.info(f"Training acc over epoch {epoch}: {float(train_acc):.4f}")
        

        # Run validation loop at the end of each epoch
        for x_batch_val, y_batch_val in val_data:
            test_step(model, x_batch_val, y_batch_val, model_metrices)
        
        val_acc = model_metrices['val_acc_metric'].result()
        val_loss = model_metrices['test_loss'].result()
        with test_summary_writer.as_default():
            tf.summary.scalar('loss', val_loss, step=epoch)
            tf.summary.scalar('accuracy', val_acc, step=epoch)
        logging.info(f"Validation loss: {float(val_loss):.4f}")
        logging.info(f"Validation acc: {float(val_acc):.4f}")
        logging.info(f"Time taken: {(time.time() - start_time):.2f}")

        # Reset metrics after each epoch
        model_metrices['train_acc_metric'].reset_states()
        model_metrices['val_acc_metric'].reset_states()
        model_metrices['train_loss'].reset_states()
        model_metrices['test_loss'].reset_states()








