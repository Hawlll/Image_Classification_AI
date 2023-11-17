import tensorflow as tf #Access gpu power
import os #Acces file strucutres
import cv2 #Computer Vision
import imghdr #Check file extensions for data
from matplotlib import pyplot as plt  # To plot data and ai performance
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

#Sequential is an api geared toward linear model (one input)
model = tf.keras.Sequential()
layer = tf.keras.layers

#Layers

# Conv2d slides over input and learns certain features like edges, patterns

#MaxPool2d grabs most important info by partioning image and returns max values 

#Dense takes nodes in a fully connected layer and learns patterns based on the node's weight or influence

#Flatten makes layer ready for others by transforming data into pixel values

#Dropout temporarliy turns of nodes for better generilzations 

#Add layers in model
model.add(layer.Conv2D(16, (3,3), 1, activation="relu", input_shape=(256,256,3)))
model.add(layer.MaxPool2D())

model.add(layer.Conv2D(32, (3,3), 1, activation="relu"))
model.add(layer.MaxPool2D())

model.add(layer.Conv2D(16, (3,3), 1, activation="relu"))
model.add(layer.MaxPool2D())

model.add(layer.Flatten())

model.add(layer.Dense(256, activation="relu"))
model.add(layer.Dense(1, activation="sigmoid"))

#Assemble layers
model.compile('adam', loss=tf.losses.BinaryCrossentropy(), metrics=['accuracy'])
model.summary()

# Track model Performance in logs file
# logdir = 'logs'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

#Train model by declaring how long what data for training and validation and then log performance
#Desirable results: Lower loss(deviation) and higher accuracy
#callbacks=[tensorboard_callback]
hist = model.fit(train, epochs=20, validation_data=val)
print(hist)

#Plot Performance

# fig = plt.figure()
# plt.plot(hist.history['loss'], color='teal', label='loss')
# plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
# plt.plot(hist.history['accuracy'], color='red', label='accuracy')
# fig.suptitle('Loss', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()

metrics = tf.keras.metrics

precision = metrics.Precision()
recall = metrics.Recall()
accuracy = metrics.BinaryAccuracy()
test_data_iterator = tf.data.NumpyIterator(test)

for batch in test_data_iterator:
    X, Y = batch
    yhat = model.predict(X)
    precision.update_state(Y, yhat)
    recall.update_state(Y, yhat)
    accuracy.update_state(Y, yhat)

print(f"Precision: {precision.result()}, Recall: {recall.result()}, Accuracy: {accuracy.result()}")

#Test model with random images
test_img = ['adam_test.jpg', 'kevin_test.jpg']
for elem in test_img:
    print("-"*20)
    img = cv2.imread(elem)
    resize = tf.image.resize(img, (256,256))
    plt.imshow(resize)
    plt.show()

    yhat = model.predict(np.expand_dims(resize/255, 0))
    print(yhat)
    if yhat > 0.5:
        print(f"Predicted class for {elem} is Sad")
    else:
        print(f"Predicted class for {elem} is Happy")



#Save model 
models = tf.keras.models
model.save(os.path.join('models','happysadmodel.h5'))

#Grab certain model
# new_model = models.load_model(os.path.join('models','happysadmodel.h5'))