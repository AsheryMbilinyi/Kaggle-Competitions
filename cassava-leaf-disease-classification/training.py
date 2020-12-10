import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tqdm import tqdm

def reading_data():
    return pd.read_csv('cassava-leaf-disease-classification/train.csv')


def loading_images():
    train_image = []
    train = reading_data()
    for i in tqdm(range(train.shape[0])):
        img = image.load_img('cassava-leaf-disease-classification/train_images/' + train['image_id'][i],
                             color_mode='rgb', target_size=None, interpolation='nearest')
        img = image.img_to_array(img)
        img = img / 255
        train_image.append(img)

    X = np.array(train_image)

    return X


def shuffle_data():
    train = pd.read_csv('cassava-leaf-disease-classification/train.csv')
    filenames = train['image_id']
    filenames = filenames.to_numpy()
    labels = train['label'].values
    labels_one_hot_encoded = to_categorical(labels)

    return shuffle(filenames,labels_one_hot_encoded)


#train test split

def train_test():
    shuffled_filenames,shuffled_labels = shuffle_data()
    X_train, X_test, y_train, y_test = train_test_split(shuffled_filenames, shuffled_labels, random_state=42, test_size=0.2)
    print(X_train)
    print(y_train)


def main():
    train_test()



if __name__ == '__main__':
    print("dealing with main")
    main()
    print("finish dealing with main")






