import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import cv2


class happySadModel:
    def __init__(self):
        self.MODEL = None

    def predict(self, image_src):
        img = cv2.imread(image_src)
        resize = tf.image.resize(img, (256,256))
        yhat = self.MODEL.predict(np.expand_dims(resize/255, 0))
        return (yhat, image_src)

    def display(self, results):
        predictions = []
        predictions.append(results)
        for elem in predictions:
            print("-"*20)
            img = cv2.imread(elem[1])
            img = img/255
            resize = tf.image.resize(img, (256,256))
            yhat = elem[0]
            if yhat > 0.5:
                print(f"Predicted class for {elem[1]} is Sad")
                depiction = "SAD"
            else:
                print(f"Predicted class for {elem[1]} is Happy")
                depiction = "HAPPY"
            plt.title(depiction)
            plt.imshow(resize)
            plt.show()


instance = happySadModel()
instance.MODEL = tf.keras.models.load_model(os.path.join('models', "happysadmodel.h5"))
instance.display(instance.predict("kevin_test.jpg"))