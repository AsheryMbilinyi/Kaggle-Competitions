#
# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# for gpu in gpus:
#   tf.config.experimental.set_memory_growth(gpu, True)




import tensorflow as tf
import numpy as np
#from skimage.io import imread
#from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def prediction():
    test_image = []
    img = image.load_img('dataset/test_images/2216849948.jpg', target_size=(224, 224, 3))
    img = image.img_to_array(img)
    img = img / 255
    test_image.append(img)
    x = np.array(test_image)
    print(f"the shape is {x.shape}")

    return x,load_model('cassava_model')



if __name__ == '__main__':
    data, model = prediction()
    print("programming is really funny")
    x = model(data)
    print(f"the tensor is {x}")
    #converting tensor to numpy array
    y = x.numpy()
    print(f"the numpy array is {y}")
    #getting the integer
    print(f"the interger is {np.argmax(y,axis=1)}")

    print("awesome")