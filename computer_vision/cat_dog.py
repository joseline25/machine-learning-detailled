"""
Programming a computer and designing algorithms for understanding what is in
these images is the field of computer vision. Computer vision powers applications like
image search, robot navigation, medical image analysis, photo management and many
more.



Computer vision is the automated extraction of information from images.
"""
import random
import pickle
from zipfile import ZipFile
import matplotlib.image as mpimg
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.models import Sequential
from keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# unzip files

data_path = '../../dog_and_cat.zip'

with ZipFile(data_path, 'r') as zip:
    zip.extractall()
    print('The data set has been extracted.')


# ********************* Data Visualization ****************************

path = '../../dataset/training_set'
classes = os.listdir(path)
print(classes)

# ['cats', 'dogs']


fig = plt.gcf()
fig.set_size_inches(16, 16)

cat_dir = os.path.join('../../dataset/training_set/cats')
dog_dir = os.path.join('../../dataset/training_set/dogs')
cat_names = os.listdir(cat_dir)
dog_names = os.listdir(dog_dir)

pic_index = 210

cat_images = [os.path.join(cat_dir, fname)
              for fname in cat_names[pic_index-32:pic_index]]
dog_images = [os.path.join(dog_dir, fname)
              for fname in dog_names[pic_index-32:pic_index]]

for i, img_path in enumerate(cat_images + dog_images):
    sp = plt.subplot(8, 8, i+1)
    sp.axis('Off')

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()


# *********************** Data Preparation for Training ************************


# In this section, we will classify the dataset into train and validation set.


base_dir = '../../dataset/training_set'

# Create datasets for trainning and validation and resize the image to w=200, h=200

train_datagen = image_dataset_from_directory(base_dir,
                                             image_size=(200, 200),
                                             subset='training',
                                             seed=1,
                                             validation_split=0.1,
                                             batch_size=32)
test_datagen = image_dataset_from_directory(base_dir,
                                            image_size=(200, 200),
                                            subset='validation',
                                            seed=1,
                                            validation_split=0.1,
                                            batch_size=32)


"""
Found 8000 files belonging to 2 classes.
Using 7200 files for training.

Found 8000 files belonging to 2 classes.
Using 800 files for validation.
"""


# ***********************Model Architecture**************************

"""
The model will contain the following Layers:

Four Convolutional Layers followed by MaxPooling Layers.

The Flatten layer to flatten the output of the convolutional layer.

Then we will have three fully connected layers followed by the output of the
flattened layer.

We have included some BatchNormalization layers to enable stable and fast training
and a Dropout layer before the final layer to avoid any possibility of overfitting.


The final layer is the output layer which has the activation function sigmoid to 
classify the results into two classes.
"""

model = tf.keras.models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(200, 200, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.1),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.2),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])


# ********************** summary of the model’s architecture **********************


print(model.summary())

"""
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 198, 198, 32)      896

 max_pooling2d (MaxPooling2  (None, 99, 99, 32)        0
 D)

 conv2d_1 (Conv2D)           (None, 97, 97, 64)        18496

 max_pooling2d_1 (MaxPoolin  (None, 48, 48, 64)        0
 g2D)

 conv2d_2 (Conv2D)           (None, 46, 46, 64)        36928

 max_pooling2d_2 (MaxPoolin  (None, 23, 23, 64)        0
 g2D)

 conv2d_3 (Conv2D)           (None, 21, 21, 64)        36928

 max_pooling2d_3 (MaxPoolin  (None, 10, 10, 64)        0
 g2D)

 flatten (Flatten)           (None, 6400)              0

 dense (Dense)               (None, 512)               3277312

 batch_normalization (Batch  (None, 512)               2048
 Normalization)

 dense_1 (Dense)             (None, 512)               262656

 dropout (Dropout)           (None, 512)               0

 batch_normalization_1 (Bat  (None, 512)               2048
 chNormalization)

 dense_2 (Dense)             (None, 512)               262656

 dropout_1 (Dropout)         (None, 512)               0

 batch_normalization_2 (Bat  (None, 512)               2048
 chNormalization)

 dense_3 (Dense)             (None, 1)                 513

=================================================================
Total params: 3902529 (14.89 MB)
Trainable params: 3899457 (14.88 MB)
Non-trainable params: 3072 (12.00 KB)
_________________________________________________________________

"""

# to understand the summary let's plot the model

keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_activations=True
)
plt.show()

# You must install pydot (`pip install pydot`) and
# install graphviz (see instructions at https://graphviz.gitlab.io/download/)
# for plot_model to work.

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ****************** Model Training  *******************

"""
 the model is working fine on epochs = 10, but you can perform
 hyperparameter tuning for better results.
"""

# history = model.fit(train_datagen,
#                     epochs=5,
#                     validation_data=test_datagen)


# ****************** Model Evaluation ********************

# Let’s visualize the training and validation accuracy with each epoch.

# history_df = pd.DataFrame(history.history)
# history_df.loc[:, ['loss', 'val_loss']].plot()
# history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
# plt.show()


"""
Epoch 1/10
225/225 [==============================] - 397s 2s/step - loss: 0.7810 - accuracy: 0.5496 - val_loss: 1.1095 - val_accuracy: 0.4950
Epoch 2/10
225/225 [==============================] - 439s 2s/step - loss: 0.6794 - accuracy: 0.6051 - val_loss: 0.6257 - val_accuracy: 0.6500
Epoch 3/10
225/225 [==============================] - 423s 2s/step - loss: 0.6346 - accuracy: 0.6543 - val_loss: 0.5781 - val_accuracy: 0.6913
Epoch 4/10
225/225 [==============================] - 395s 2s/step - loss: 0.5748 - accuracy: 0.7051 - val_loss: 0.6668 - val_accuracy: 0.6475
Epoch 5/10
225/225 [==============================] - 447s 2s/step - loss: 0.5094 - accuracy: 0.7588 - val_loss: 1.4832 - val_accuracy: 0.5487
Epoch 6/10
225/225 [==============================] - 447s 2s/step - loss: 0.4686 - accuracy: 0.7837 - val_loss: 0.8609 - val_accuracy: 0.5938
Epoch 7/10
225/225 [==============================] - 455s 2s/step - loss: 0.4204 - accuracy: 0.8072 - val_loss: 0.5082 - val_accuracy: 0.7500
Epoch 8/10
225/225 [==============================] - 1395s 6s/step - loss: 0.3641 - accuracy: 0.8428 - val_loss: 0.6417 - val_accuracy: 0.7362
Epoch 9/10
225/225 [==============================] - 388s 2s/step - loss: 0.3118 - accuracy: 0.8650 - val_loss: 0.5621 - val_accuracy: 0.7750
Epoch 10/10
225/225 [==============================] - 426s 2s/step - loss: 0.4358 - accuracy: 0.7835 - val_loss: 0.6447 - val_accuracy: 0.6288
"""

# ******************** Save the model **********************


# Save the model to the file model_saved

# with open('model_saved.pkl', 'wb') as file:
#     pickle.dump(model, file)


#  load the saved model later for prediction or further training,
# we use the pickle.load() function


with open('model_saved.pkl', 'rb') as file:
    loaded_model = pickle.load(file)


"""
We can use a different file extension instead of .pkl to save the model.
The file extension is simply a naming convention, and you can choose any
extension that makes sense for your use case. The .pkl extension is 
commonly used for files serialized using pickle, but it's not mandatory.

For example, you can use .sav as the file extension




While there is no universally recommended file extension for saving 
machine learning models, certain conventions have emerged within the
machine learning community. Here are a few commonly used file extensions 
for saving machine learning models:

.pkl or .pickle: These extensions are often associated with the pickle 
module in Python, which is commonly used for serializing Python objects,
including machine learning models.

.joblib: This extension is associated with the joblib library, which is
another popular option for serializing machine learning models in Python.
It is often used when dealing with large NumPy arrays efficiently.

.h5 or .hdf5: These extensions are commonly used for saving models in the
Hierarchical Data Format (HDF5) format. The HDF5 format allows for efficient
storage and retrieval of large numerical datasets, including machine learning
models.

.pth or .pt: These extensions are commonly used for saving models in PyTorch,
a popular deep learning framework. It is particularly suitable for saving models
that contain learnable parameters (e.g., neural network weights).

It's important to note that the choice of file extension is primarily a
matter of convention and personal preference. The key consideration is 
to use a file extension that is descriptive and helps identify the file
as a serialized machine learning model.



We can do the same with the library joblib

import joblib

# Assuming you have a trained model named 'model'
# Save the model to a file named model_filename

joblib.dump(model, 'model_filename.pkl')

To load the saved model later for prediction or further training,


loaded_model = joblib.load('model_filename.pkl')




Note that joblib is commonly used for efficiently serializing large NumPy arrays,
which can be useful when dealing with machine learning models.
"""


# ******************* Model Testing and Prediction ***************

# Let’s check the model for random images

base_dir_test = '../../dataset/test_set/cats'

# Create datasets for trainning and validation and resize the image to w=200, h=200

# test_datagen = image_dataset_from_directory(base_dir_test,
#                                              image_size=(200, 200),
#                                              subset='test',
#                                              seed=1,
#                                              validation_split=0.2,
#                                              batch_size=32)
# #Input image

flist = os.listdir(base_dir_test)
test_img = random.choice(flist)
test_img = os.path.join(base_dir_test, test_img)


# For show image

from PIL import Image
pil_im = Image.open(test_img)
plt.show
# plt.imshow(test_img)
# test_image = image.img_to_array(test_img)
# test_image = np.expand_dims(test_img, axis=0)

# Result array
result = loaded_model.predict(test_datagen)

# Mapping result array with the main name list
i = 0
if (result >= 0.5):
    print("Dog")
else:
    print("Cat")
