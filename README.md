# Hands Signs Recognition with Tensorflow (CS230)
This is a practice project based on Standford CS230.

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
