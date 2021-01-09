

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D,MaxPooling2D,BatchNormalization,Input,Dropout,Flatten,Dense
from tensorflow.keras.applications.resnet50 import ResNet50
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


class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self. conv1 = Conv2D(filters=64, kernel_size=(5, 5), activation='relu')
        self.batch1 = BatchNormalization(axis=3)
        self.conv1_2 =  Conv2D(filters=64, kernel_size=(5, 5), activation='relu')
        self.max1 = MaxPooling2D(pool_size=(2, 2))
        self.batch1_2 = BatchNormalization(axis=3)
        self.drop1 = Dropout(0.25)

        self.conv2 = Conv2D(filters=128, kernel_size=(5, 5), activation='relu')
        self.batch2 = BatchNormalization(axis=3)
        self.conv2_2 = Conv2D(filters=128, kernel_size=(5, 5), activation='relu')
        self.max2 = MaxPooling2D(pool_size=(2, 2))
        self.batch2_2 = BatchNormalization(axis=3)
        self.drop2 = Dropout(0.25)

        self.conv3 = Conv2D(filters=256, kernel_size=(5, 5), activation='relu')
        self.batch3 = BatchNormalization(axis=3)
        self.conv3_2 = Conv2D(filters=256, kernel_size=(5, 5), activation='relu')
        self.max3 = MaxPooling2D(pool_size=(2, 2))
        self.batch3 = BatchNormalization(axis=3)
        self.drop3 = Dropout(0.5)

        self.flat4 = Flatten()
        self.dense4 = Dense(512, activation='relu')
        self.batch4 = BatchNormalization()
        self.drop4 = Dropout(0.5)

        self.dense5  = Dense(60, activation="relu")
        self.batch5 = BatchNormalization()
        self.drop5 = Dropout(0.5)

        self.dense6 = Dense(5,activation='softmax')









    def call(self, inputs, training=None, mask=None):

        x = self.conv1(inputs)
        x = self.batch1(x)
        x = self.conv1_2(x)
        x = self.max1(x)
        x = self.batch1_2(x)
        x= self.drop1(x)

        x = self.conv2(x)
        x = self.batch2(x)
        x = self.conv2_2(x)
        x = self.max2(x)
        x = self.batch2_2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x =self.batch3(x)
        x = self.conv3_2(x)
        x = self.max3(x)
        x = self.batch3(x)
        x = self.drop3(x)

        x = self.flat4(x)
        x = self.dense4(x)
        #x = self.batch4(x)
        x = self.drop4(x)

        x = self.dense5(x)
        #x = self.batch5(x)
        x = self.drop5(x)

        x = self.dense6(x)

        return x


    def summary(self):
        inputs = Input(shape=(224,224,3))
        outputs = self.call(inputs)
        model = Model(inputs=inputs,outputs=outputs)
        return model.summary()


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


# def creating_model():
#     model = Sequential()
#     model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(80, 80, 3)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(Conv2D(32, (3, 3)))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     #model.add(Dropout(0.25))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     #model.add(Dropout(0.5))
#     model.add(Dense(5, activation='softmax'))
#
#     return model


def main():
    batch_size = 16
    train_batch_generator,val_batch_generator = train_val_generation(batch_size)
    #model = creating_model()
    model = MyModel()
    opt = tf.keras.optimizers.Adadelta(learning_rate=0.01)
    loss = tf.keras.losses.CategoricalCrossentropy()
    model.compile(loss=loss, optimizer=opt, metrics=['categorical_accuracy'])
    model.summary()

    model.fit(train_batch_generator,
                        steps_per_epoch=int(17117 // batch_size),
                        epochs=1,
                        verbose=1,
                        validation_data=val_batch_generator,
                        validation_steps=int(4280 // batch_size))

    model.save_weights('cassava_mode_ya_nne',save_format='tf')



if __name__ == '__main__':
    print("start to learn")
    main()
    print("finished learning")






