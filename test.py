import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np
import seaborn as sns


# preparing dataset
base_dir = '../dataset/'
test_dir = os.path.join(base_dir, 'test')

test_danger_dir = os.path.join(test_dir, 'dangerous')
test_danger_dir = os.path.join(test_dir, 'normal')

# The test data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow test images in batches of 20 using test_datagen generator
test_generator = test_datagen.flow_from_directory( test_dir,  batch_size = 32, class_mode = 'binary', target_size = (224, 224))

base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.load_weights('./tumor_classification.h5')

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

test_results = model.evaluate(test_generator)
print("test loss, test acc:", test_results)

predictions = model.predict(test_generator, batch_size=None, verbose=0, steps=None, callbacks=None)

print(predictions)
predictions = np.argmax(predictions, axis = 1)

print(predictions)
print(test_generator.labels)

print('accuracy score:   ', accuracy_score(test_generator.labels, predictions))
sns.heatmap(confusion_matrix(test_generator.labels, predictions),  annot=True)
plt.savefig('confusion_matrix.png')
plt.show()

print(test_generator.labels, test_generator)

# print('hello world', dir(tf.math.confusion_matrix(test_generator.labels, predictions)))
print('hello world',confusion_matrix(test_generator.labels, predictions))
print(precision_recall_fscore_support(test_generator.labels, predictions, average='macro'))
