import tensorflow as tf
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from YoutubeDownload import Download, Get_Frame
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt

batch_size = 32
img_size = 	224
    # Uploading images dataset from direction.
data = tf.keras.utils.image_dataset_from_directory('Weather',
                                                                 shuffle=True,
                                                                 batch_size=batch_size,
                                                                 image_size=(img_size,img_size))
# Defining number and names of classes
NUM_CLASSES=len(data.class_names)
class_names=data.class_names

# Spliting train 80% and validation 20% img
train_size=int(len(data)*.7)
eval_size=int(len(data)*.2)
test_size=int(len(data)*.1)

train=data.take(train_size)
eval=data.skip(train_size).take(eval_size)
test=train.take(train_size+eval_size).take(test_size)


print('Number of train batches: %d' % tf.data.experimental.cardinality(train))
print('Number of val batches: %d' % tf.data.experimental.cardinality(eval))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test))

# Defining image augmentation layer
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomContrast(factor=0.2),
    ],
    name="img_augmentation",
)

# Automatic decouple the time of data is produced from the time when data is consumed.
AUTOTUNE = tf.data.AUTOTUNE
ds_train = train.prefetch(buffer_size=AUTOTUNE)
ds_val = eval.prefetch(buffer_size=AUTOTUNE)

# Defining pretraining model
base_model = EfficientNetB0(include_top=False, weights='imagenet')
base_model.trainable = False

# Converting images to the required dimension
image_batch, label_batch = next(iter(ds_train))
feature_batch = base_model(image_batch)

# Using a global average pooling layer, thanks to witch I have single 1280 element vector per filter.
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)

# Combining all layers to model
inputs = tf.keras.Input(shape=(img_size,img_size,3))
x = img_augmentation(inputs)
x = base_model(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()

# Compiling model
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"]
    )
# Training model
epochs = 10
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=1)
evaluate=model.evaluate(test)

# Accuracy plot depending on epochs
def plot_hist(hist):
    plt.figure(figsize=(5, 5))
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.figure(figsize=(5, 5))
    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()
    plt.show()

plot_hist(hist)

# Saving model
model.save('modelB0.h5')

# Comparison of models

# Loading models
model_B0=load_model("modelB0.h5")
model_B1=load_model("modelB1.h5")
model_B2=load_model("modelB2.h5")
model_B3=load_model("modelB3.h5")
model_B4=load_model("modelB4.h5")

# Evaluating each models over test data
test = test.map(lambda x, y: (tf.image.resize(x,(224,224)), y))
B0=model_B0.evaluate(test)
test = test.map(lambda x, y: (tf.image.resize(x,(240,240)), y))
B1=model_B1.evaluate(test)
test = test.map(lambda x, y: (tf.image.resize(x,(260,260)), y))
B2=model_B2.evaluate(test)
test = test.map(lambda x, y: (tf.image.resize(x,(300,300)), y))
B3=model_B3.evaluate(test)
test = test.map(lambda x, y: (tf.image.resize(x,(380,380)), y))
B4=model_B4.evaluate(test)

# Printing accuracy and loss for each models
print('EfficientNetB0 accuracy is {1}, and loss is {0}'.format(B0[0], B0[1]))
print('EfficientNetB1 accuracy is {1}, and loss is {0}'.format(B1[0], B1[1]))
print('EfficientNetB2 accuracy is {1}, and loss is {0}'.format(B2[0], B2[1]))
print('EfficientNetB3 accuracy is {1}, and loss is {0}'.format(B3[0], B3[1]))
print('EfficientNetB4 accuracy is {1}, and loss is {0}'.format(B4[0], B4[1]))