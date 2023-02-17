import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.utils import load_img, img_to_array
from keras.models import load_model
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
import matplotlib.pyplot as plt
from PIL import Image

# Defining batch size end img_size.
# EfficientNetB0 requires 224x224 img size
batch_size = 32
img_size= 224

# Uploading images dataset from direction.
train = tf.keras.utils.image_dataset_from_directory(r'Weather Classification\Weather',
                                                                 shuffle=True,
                                                                 batch_size=batch_size,
                                                                 image_size=(img_size,img_size))
# Uploading  test images.
def image_load(path,img_size):
    image = tf.keras.utils.load_img(path,target_size=(img_size,img_size))
    image = tf.keras.utils.img_to_array(image)
    image = np.array([image])
    return image

rain_img_path=r'Weather Classification\Test img\rain.jpg'
sunshine_img_path=r'Weather Classification\Test img\sunshine.jpg'
cloudy_img_path=r'Weather Classification\Test img\cloudy.jpg'
sunrise_img_path=r'Weather Classification\Test img\sunrise.jpg'
test_rain_img=image_load(rain_img_path,img_size)
test_sunshine_img=image_load(sunshine_img_path,img_size)
test_cloudy_img=image_load(cloudy_img_path,img_size)
test_sunrise_img=image_load(sunrise_img_path,img_size)

# Defining number and names of classes
NUM_CLASSES=len(train.class_names)
class_names=train.class_names

# Spliting train 80% and validation 20% img
train_batches = tf.data.experimental.cardinality(train)
val = train.take(train_batches // 5)
train = train.skip(train_batches // 5)

print('Number of train batches: %d' % tf.data.experimental.cardinality(train))
print('Number of val batches: %d' % tf.data.experimental.cardinality(val))

# Defining image augmentation layer
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1)
    ],
    name="img_augmentation",
)

# Automatic decouple the time of data is produced from the time when data is consumed.
AUTOTUNE = tf.data.AUTOTUNE
ds_train = train.prefetch(buffer_size=AUTOTUNE)
ds_val = val.prefetch(buffer_size=AUTOTUNE)

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
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=2)

# Accuracy plot depending on epochs
def plot_hist(hist):
    plt.plot(hist.history["accuracy"])
    plt.plot(hist.history["val_accuracy"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()
plot_hist(hist)

# Saving model
model.save('model1.h5')
new_model = tf.keras.models.load_model('model1.h5')


def prediction(img):
    prediction = model.predict(img)
    score = tf.nn.softmax(prediction)
    return print(
        "The weather in the picture looks like: {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

# Predicting weather phenomena in images.
prediction(test_rain_img)
prediction(test_cloudy_img)
prediction(test_sunrise_img)
prediction(test_sunshine_img)
