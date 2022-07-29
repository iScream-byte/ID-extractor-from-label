import pandas as pd
import numpy as np 
import itertools
import keras
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img 
from keras.models import Sequential 
from keras import optimizers
from keras.preprocessing import image
from keras.layers import Dropout, Flatten, Dense 
from keras import applications 
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import math 

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau



#Default dimensions we found online
img_width, img_height = 270, 270 
 
#Create a bottleneck file
top_model_weights_path = 'bottleneck_fc_model.h5'# loading up our datasets
train_data_dir = 'D:\\iScream\\tfod\\Tensorflow\\workspace\\images\\training_set' 
validation_data_dir = 'D:\\iScream\\tfod\\Tensorflow\\workspace\\images\\test_set' 
test_data_dir = 'D:\\iScream\\tfod\\Tensorflow\\workspace\\images\\test_set'
 
# number of epochs to train top model 
epochs = 50 #this has been changed after multiple model run 
# batch size used by flow_from_directory and predict_generator 
batch_size = 3 



#Loading vgc16 model
vgg16 = applications.VGG16(include_top=False, weights="imagenet")
datagen = ImageDataGenerator(rescale=1. / 255) 
#needed to create the bottleneck .npy files



#__this can take an hour and half to run so only run it once. 
#once the npy files have been created, no need to run again. Convert this cell to a code cell to run.__start = datetime.datetime.now()
 
generator = datagen.flow_from_directory( 
    train_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False) 
 
nb_train_samples = len(generator.filenames) 
num_classes = len(generator.class_indices) 
 
predict_size_train = int(math.ceil(nb_train_samples / batch_size)) 
 
bottleneck_features_train = vgg16.predict(generator, predict_size_train) 
 
np.save('bottleneck_features_train.npy', bottleneck_features_train)



generator = datagen.flow_from_directory( 
    test_data_dir, 
    target_size=(img_width, img_height), 
    batch_size=batch_size, 
    class_mode=None, 
    shuffle=False)
nb_test_samples = len(generator.filenames) 
num_classes = len(generator.class_indices) 
 
predict_size_test = int(math.ceil(nb_test_samples / batch_size)) 
 
bottleneck_features_test = vgg16.predict(generator, predict_size_test) 
 
np.save('bottleneck_features_test.npy', bottleneck_features_test)



#training data

generator_top = datagen.flow_from_directory( 
   train_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False)
 
nb_train_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
train_data = np.load('bottleneck_features_train.npy') 
 
# get the class labels for the training data, in the original order 
train_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
train_labels = to_categorical(train_labels, num_classes=num_classes)



#test data
generator_top = datagen.flow_from_directory( 
   test_data_dir, 
   target_size=(img_width, img_height), 
   batch_size=batch_size, 
   class_mode='categorical', 
   shuffle=False) 
 
nb_test_samples = len(generator_top.filenames) 
num_classes = len(generator_top.class_indices) 
 
# load the bottleneck features saved earlier 
test_data = np.load('bottleneck_features_test.npy') 
 
# get the class labels for the training data, in the original order 
test_labels = generator_top.classes 
 
# convert the training labels to categorical vectors 
test_labels = to_categorical(test_labels, num_classes=num_classes)

rlr = ReduceLROnPlateau()
checkpoint = ModelCheckpoint('cnn_best_model.h5',save_weights_only= True, save_best_only= True, mode = 'max',monitor='val_accuracy')
callbacks = [rlr,checkpoint]


model = Sequential() 
model.add(Flatten(input_shape=train_data.shape[1:])) 
model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.5)) 
model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
model.add(Dropout(0.3)) 
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(learning_rate=1e-4),
              metrics='accuracy')

history = model.fit(train_data, train_labels, 
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    validation_data=(test_data, test_labels))

model.load_weights('cnn_best_model.h5')

model.save_weights(top_model_weights_path)
(eval_loss, eval_accuracy) = model.evaluate( 
    test_data, test_labels, batch_size=batch_size,verbose=1)

print('[INFO] accuracy: {:.2f}%'.format(eval_accuracy * 100)) 
print('[INFO] Loss: {}'.format(eval_loss)) 


model.save("D:\\iScream\\tfod\\CNNmodel")

newmodel=keras.models.load_model("D:\\iScream\\tfod\\CNNmodel")
newmodel.load_weights('cnn_best_model.h5')

def read_image(file_path):
   print('[INFO] loading and preprocessing image…') 
   image = load_img(file_path, target_size=(256, 256)) 
   image = img_to_array(image) 
   image = np.expand_dims(image, axis=0)
   image /= 255. 
   return image

def test_single_image(path):  
  images = read_image(path)  
  bt_prediction = vgg16.predict(images) 
  preds = model.predict(bt_prediction)    
  return preds.argmax()

path = 'D:\\iScream\\tfod\\Tensorflow\\workspace\\images\\training_set\\X\\29.jpg'
pred=test_single_image(path)
print(pred)
