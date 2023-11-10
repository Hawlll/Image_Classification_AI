import tensorflow as tf #Access gpu power
import os #Acces file strucutres
import cv2 #Computer Vision
import imghdr #Check file extensions for data
from matplotlib import pyplot as plt
import numpy as np
#Make sure interpreter is python 3.9 for tensor-gpu compabability


# Limit tensorflow's access to vram, preventing out of memory error
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = "data"

image_extensions = ["jpeg", "jpg", "bmp", "png"] # File extensions we want

#Loops through data in happy_data and sad_data and removes unwanted extensions
for image_class in os.listdir(data_dir):
    for image in os.listdir(os.path.join(data_dir, image_class)):
        image_path = os.path.join(data_dir, image_class, image)
        try:
            img = cv2.imread(image_path) #Load image
            tip = imghdr.what(image_path) # Store ext
            if tip not in image_extensions:
                print(f"Image not in desired exentions lists {image_path}")
                os.remove(image_path)
        except Exception as e: #Happens if something wrong with loading
            print(f"Issue with image {image_path}")

#Builds images dataset (Resize images, shuffle them, label, classifies). This is a generator and does not store in memory (DATA PIPELINE).
data = tf.keras.utils.image_dataset_from_directory(data_dir)

#Scales down images for optimization
scaled_data = data.map(lambda x,y: (x/255, y))

#train data to train model, validation data to evalutate model, test data to test (post training)
#size represent number of batches
train_size = int(len(data) *0.7)
val_size = int(len(data)*0.2)+1
test_size = int(len(data) * 0.1)+1

#Grab Batches. Skip means skips spots. Take is take
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size+val_size).take(test_size)

scaled_data_iterator = tf.data.NumpyIterator(scaled_data)

#Each batch has 32 data. image in array and label(1 or 0 (happy or sad))
batch = scaled_data_iterator.next()

HAPPY = 0
SAD = 1


# Display data analysis
figure, ax = plt.subplots(ncols=4)

for index, img, in enumerate(batch[0][:4]):
    ax[index].imshow(img)
    if batch[1][index] == HAPPY:
        ax[index].title.set_text("Happy")
    elif batch[1][index] == SAD:
        ax[index].title.set_text("Sad")
    else:
        ax[index].title.set_text("Couldn't Decide")
    
plt.show()
    
data = data.map(lambda x,y: (x/255, y))