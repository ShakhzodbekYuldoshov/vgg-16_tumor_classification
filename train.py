import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt

from tensorflow.keras.applications.vgg16 import VGG16


# preparing dataset
base_dir = '../dataset/'
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')

train_danger_dir = os.path.join(train_dir, 'dangerous')
train_norm_dir = os.path.join(train_dir, 'normal')

test_danger_dir = os.path.join(test_dir, 'dangerous')
test_danger_dir = os.path.join(test_dir, 'normal')

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
# The test data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 32, class_mode = 'binary', target_size = (224, 224))

# Flow test images in batches of 20 using test_datagen generator
test_generator = test_datagen.flow_from_directory( test_dir,  batch_size = 32, class_mode = 'binary', target_size = (224, 224))

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

base_model = VGG16(input_shape = (224, 224, 3), # Shape of our images
include_top = False, # Leave out the last fully connected layer
weights = 'imagenet')

for layer in base_model.layers:
    layer.trainable = False

# Flatten the output layer to 1 dimension
x = layers.Flatten()(base_model.output)

# Add a fully connected layer with 512 hidden units and ReLU activation
x = layers.Dense(512, activation='relu')(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Add a final sigmoid layer with 1 node for classification output
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = tf.keras.optimizers.RMSprop(lr=0.0001), loss = 'binary_crossentropy',metrics = ['acc'])

vgghist = model.fit(train_generator, validation_data = test_generator, steps_per_epoch = 100, epochs = 25, callbacks=[cp_callback])


model.save('tumor_classification.h5')

print(vgghist.history.keys())
acc = vgghist.history['acc']
loss = vgghist.history['loss']
val_acc = vgghist.history['val_acc']
val_loss = vgghist.history['val_loss']

plt.figure()
plt.plot(acc, label='Training Accuracy')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.savefig('training accuracy.png')

plt.figure()
plt.plot(val_acc, label='Validation Accuracy')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.savefig('validation accuracy.png')

plt.figure()
plt.plot(val_loss, label='Validation loss')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.savefig('validation loss.png')

plt.figure()
plt.plot(loss, label='Training Loss')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.xlabel('epoch')
plt.savefig('training_loss')
plt.show()

print(dict(vgghist.history))
