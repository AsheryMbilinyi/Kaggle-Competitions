

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from tqdm import tqdm
from myCustomGenerator import CustomGenerator


def reading_data():
    return pd.read_csv('dataset/train.csv')


def loading_images():
    train_image = []
    train = reading_data()
    for i in tqdm(range(train.shape[0])):
        img = image.load_img('dataset/train_images/' + train['image_id'][i],
                             color_mode='rgb', target_size=None, interpolation='nearest')
        img = image.img_to_array(img)
        img = img / 255
        train_image.append(img)

    X = np.array(train_image)

    return X


def shuffle_data():
    train = pd.read_csv('dataset/train.csv')
    filenames = train['image_id']
    filenames = filenames.to_numpy()
    labels = train['label'].values
    labels_one_hot_encoded = to_categorical(labels)

    return shuffle(filenames,labels_one_hot_encoded)


#train test split


def train_val_generation(batch_size):
    #batch_size = 32
    shuffled_filenames,shuffled_labels = shuffle_data()
    X_train, X_val, y_train, y_val = train_test_split(shuffled_filenames, shuffled_labels,
                                                      random_state=42, test_size=0.2)
    training_batch_generator = CustomGenerator(X_train, y_train, batch_size)
    validation_batch_generator = CustomGenerator(X_val, y_val, batch_size)
    # print(X_train.shape)
    # print(X_val.shape)

    return training_batch_generator,validation_batch_generator

def creating_model():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(80, 80, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))

    return model


def main():
    batch_size = 32
    train_batch_generator,val_batch_generator = train_val_generation(batch_size)
    model = creating_model()
    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    model.summary()

    model.fit(train_batch_generator,
                        steps_per_epoch=int(17117 // batch_size),
                        epochs=10,
                        verbose=1,
                        validation_data=val_batch_generator,
                        validation_steps=int(4280 // batch_size))
    #train_val_generation()



if __name__ == '__main__':
    print("start to learn")
    main()
    print("finished learning")






