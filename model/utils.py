"""General utility functions"""

import json
import logging
import matplotlib.pyplot as plt
import io
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix


class Params():
    """Class that loads hyperparameters from json file
    
    Example:
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5 # Change value of learning rate
    """
    
    def __init__(self, json_path):
        self.update(json_path)
        
    def __repr__(self):
        rep = f"lr: {self.learning_rate}, dropout: {self.dropout_rate}, bn:{self.bn_momentum}"
        return rep

    def update(self, json_path):
        """Loads parameters from json path"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
    
    def save(self, json_path):
        """"Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by 'params.dict['learning_rate']'"""
        return self.__dict__



def set_logger(log_path):
    """Sets the logger to log info in terminal and file `log_path`
    The output to terminal is saved in a permanent file `model_dir/train.log`.
    
    Args:
        log_path: (str) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


# Copied from https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image


def image_grid(data, labels):
    # Data shape should be (BATCH_SIZE, H, W, C)
    assert data.ndim == 4

    figure = plt.figure(figsize=(10, 10))
    num_images = data.shape[0]
    size = int(np.ceil(np.sqrt(num_images)))
    labels = labels.numpy()

    for i in range(num_images):
        plt.subplot(size, size, i + 1, title="Label "+str(labels[i]))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        if data.shape[3] == 1:
            plt.imshow(data[i], cmap=plt.cm.binary)
        else:
            plt.imshow(data[i])
    
    return figure


def get_confusion_matrix(y_labels, logits, num_labels):
    preds = np.argmax(logits, axis=1)
    cm = confusion_matrix(y_labels, preds, labels=np.arange(num_labels))
    return cm


def plot_confusion_matrix(cm, num_labels):
    size = num_labels
    figure = plt.figure(figsize=(size, size))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    indices = np.arange(num_labels)
    plt.xticks(indices, np.arange(num_labels), rotation=45)
    plt.yticks(indices, np.arange(num_labels))

    # Normalize confusion matrix
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=3)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2.0
    for i in range(size):
        for j in range(size):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(i, j, cm[i, j], horizontalalignment='center', color=color)

    plt.tight_layout()
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')

    cm_image = plot_to_image(figure)
    return cm_image

