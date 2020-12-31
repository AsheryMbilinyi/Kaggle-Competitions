import tensorflow as tf
import numpy as np

from skimage.io import imread
from skimage.transform import resize


class CustomGenerator(tf.keras.utils.Sequence):

    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size: (idx + 1) * self.batch_size]

        return np.array([resize(imread('/home/mbiash00/kaggle/cassava-leaf-disease-classification/dataset/train_images/' + str(file_name)), (224, 224, 3))
                         for file_name in batch_x]) / 255.0, np.array(batch_y)