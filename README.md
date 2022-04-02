# Hands Signs Recognition with Tensorflow (CS230)
This is a practice project based on Standford CS230 dataset the original cs230 repository can be found **[here](https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision)**

The project is built on Tensorflow 2x and the scripts are transformed accordingly. It also include tensorboard visualization sample images, confusion matrix and hyperparameter search.

## Requirements
Virtual Environment and Python 3 are recommended.

```
# Setting up virtualenv for windows
python -m venv HS_env
HS_env\Scripts\activate.bat
pip install -r requirements.txt
```

## Task
Given a image of a hand representing sign of 0, 1, 2, 3, 4, or 5, predict the correct label.

## Dataset
Download SIGNS dataset here which is hosted on google drive **[here](https://drive.google.com/file/d/1ufiR6hUKhXoAyiBNsySPkUwlvE_wfEHC/view)**

Here is the dataset structure
```
SIGNS/
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```

The images are named following `{label}_IMG_{id}.jpg` where the label is in `[0, 5]`
The training set contains 1,080 images and test set contains 120 images.

Run the script `build_dataset.py` which will resize the images to size `(64, 64)`. The resized dataset will be located in `data/64x64_SIGNS`:

```bash
python build_dataset.py --data_dir data/SIGNS --output_dir data/64X64_SIGNS
```

## Quickstart

1. **Build the dataset of size 64x64**: make sure you complete this step before training

```bash
python build_dataset.py --data_dir {input_data directory} --output_dir {output directory}
```

2.  **Setup experiment** Setup model parameters input file under `model_dir` that should contain a file `params.json` which sets up parameters for training. it looks like:

```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10
    ...
}
```
The default `model_dir` is `experiments/base_model` directory

3. **Training** model.

```
python train.py --data_dir data/64x64_SIGNS --model_dir experiments/base_model
```
It will instantiate a model and train it on training set and evaluate on dev (validation) set based on the parameters specified in `params.json` under model_dir.

The training log will be generated under `model_dir/train_logs` and tensorboard logs will be generated under `model_dir/tf.logs`.

The tensorboard will generate the following:
- Training and Validation Accuracy
- Training and Validation Loss
- Sample images visualization
- Confusion Matrix

To view tensorboard training logs, simply run

```
tensorboard --logdir ./model_dir/tf_logs/log_directory
```
where `log_directory` is the tensorboard logs generated after training.

Furthermore the weights are saved under `model_dir/model_weights`. The training can be continued by restoring saved weights by providing `restore_from` path argument.

```
python ./train.py --restore_from ./experiments/base_model/model_weights/20220331-091205/
```

4. **Hyperparameters Search**
The `hp_search.py` can be found in model directory. The list of hyperparmeters search can changed by updated following lines
```
HP_LR = hp.HParam('learning_rate', hp.Discrete([1e-1, 1e-3, 1e-5]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.2, 0.4]))
HP_BN_M = hp.HParam('bn_momentum', hp.Discrete([0.1, 0.4, 0.8]))
```
Training for hyperparameter search is similar to training a model by passing `--hp_search True` optional argument. This is shown below.
```
python train.py --hp_search True
```
This will generate tensorboard logs to visualize hyperparemeter search results. To run tensorboard logs, run following command:
```
tensorboard --logdir ./model_dir/hp_search
```

5. **Evaluate**
After selecting best hyperparementers, model and weights, evaluate the model on test set. Run `evaluate.py` and provide best weights directory as shown below
```
python evaluate.py ./experiments/base_model/model_weights/best_weights/
```




