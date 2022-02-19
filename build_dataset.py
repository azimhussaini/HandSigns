"""Split the Signs dataset into train/dev/test and resize images to 64x64

Original images have size (3024, 3024)
We have already have a test set created, so we only need to split "train_signs" into train and dev sets.
Dev set will represent 20% of "train_signs" as dev set.

"""

import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 64

parser = argparse.ArgumentParser(description="Build dataset based on input_data directory and output directory")
parser.add_argument('--data_dir', default='data/SIGNS', metavar='' ,help="Directory with the SIGNS dataset")
parser.add_argument('--output_dir', default='data/64x64_SIGNS', metavar='', help="Where to write the new data")


def resize_and_save(data_dir, filename, output_dir, size=SIZE):
    "Resize the image contained in 'filename' and save it to the 'output_dir'"
    image = Image.open(os.path.join(data_dir, filename))
    # Use bilinear interpolation instead of default "nearest neighbor" method
    image = image.resize((size,size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename))


if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_signs')
    test_data_dir = os.path.join(args.data_dir, 'test_signs')

    # Get filenames in each directory (Train and Test)
    filenames = os.listdir(train_data_dir)
    filenames = [f for f in filenames if f.endswith('.jpg')]

    test_filenames = os.listdir(test_data_dir)
    test_filenames = [f for f in test_filenames if f.endswith('jpg')]

    # Split the images in 'train_signs' into 80% train and 20% dev
    # Make sure to always shuffle with a fixed seed to make it reproducible
    random.seed(42)
    filenames.sort()
    random.shuffle(filenames)

    split = int(0.8 * len(filenames))
    train_filenames = filenames[:split]
    dev_filenames = filenames[split:]

    filenames = {'train': train_filenames, 'dev': dev_filenames, 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    
    # Preprocess train and dev
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_signs'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print("Warning: dir {} already exists".format(output_dir_split))
        
        print("Processing {} data, saving preprocessed data to {}".format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            if split == "test":
                data_dir = test_data_dir
            else:
                data_dir = train_data_dir
            resize_and_save(data_dir, filename, output_dir_split, size=SIZE)
    
    print("Done building dataset")

