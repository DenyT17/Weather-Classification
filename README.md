# Weather Classification ⛅🌧☁

## Technologies 💡
![Tensor Flow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-FF0000?style=for-the-badge&logo=keras&logoColor=white)
![PyCharm](https://img.shields.io/badge/pycharm-143?style=for-the-badge&logo=pycharm&logoColor=black&color=black&labelColor=green)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Description ❔❓

In this project I use images depicting weather phenomena to image classification. I use TensorFlow and EfficientNet(B0 - B4) to  pretrained models. 

When add few new weather and evaluate accuracy will be satisfactory phenomene I will want to use webcam from diferent countries as images to prediction. 

For the time being I predict this weather phenomenon :
- Rain
- Cloudy
- Sunrise
- Shine
- Snow


## Dataset📁
Images used in this project you can find [here](https://data.mendeley.com/datasets/4drtyfjtfy/1)

This datasets contain images depicting four weather phenomena such as:
* Rain
* Cloudy
* Sunrise
* Sunshine

I separated the photos into appropriate directories which I named accordingly. In futur i want extend the dataset with other weather phenomena.

## The method of project implementation ✅

A detailed description of the implementation of the project can be found in the Weather Classification.py file

#### 1️⃣ First I load and spli data: 
```python
train = tf.keras.utils.image_dataset_from_directory(r'Weather Classification\Weather',
                                                                 shuffle=True,
                                                                 batch_size=batch_size,
                                                                 image_size=(img_size,img_size))
                                                                 
train_batches = tf.data.experimental.cardinality(train)
val = train.take(train_batches // 5)
train = train.skip(train_batches // 5)
```
#### 2️⃣ Because my dataset is pretty small, I must augmentation it:
```python
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1)
    ],
    name="img_augmentation",
)
```

#### 3️⃣ Now I define pretrained model and global average pooling layer: 
``` python
image_batch, label_batch = next(iter(ds_train))
feature_batch = base_model(image_batch)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
```

#### 4️⃣ In next step I can combine all layers to model: 
``` python
inputs = tf.keras.Input(shape=(img_size,img_size,3))
x = img_augmentation(inputs)
x = base_model(x)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
model = tf.keras.Model(inputs, outputs)
model.summary()
``` 

Model summary:

![image](https://user-images.githubusercontent.com/122997699/219765405-9f8cd4f7-d327-46fd-8ff3-287c452378d6.png)

#### 5️⃣ Now I must compile,train and save my model: 
``` python
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"]
    )
epochs = 10
hist = model.fit(ds_train, epochs=epochs, validation_data=ds_val, verbose=2)  
``` 

Training results for 10 epochs: 
#### EfficientNetB0
![EN0 Accuracy](https://user-images.githubusercontent.com/122997699/220388523-d3b8e93a-a051-4e51-85fd-cb73328e3028.png) 
![EN0 Loss](https://user-images.githubusercontent.com/122997699/220388662-291be01c-0d86-432c-a355-66d39960252d.png)

#### EfficientNetB1

![EN1 Accuracy](https://user-images.githubusercontent.com/122997699/220388958-38d8f065-0b7b-4dc0-9359-028fae2d41e3.png)
![EN1 Loss](https://user-images.githubusercontent.com/122997699/220388964-847abb88-dd42-4480-89a6-8b92a993e59a.png)

#### EfficientNetB2
![EN2 Accuracy](https://user-images.githubusercontent.com/122997699/220392380-3dd7f318-191b-4e3e-a93e-2213c9aa4adc.png)
![EN2 Loss](https://user-images.githubusercontent.com/122997699/220392389-0a572268-46b6-4df4-8b54-d2887cd81d88.png)



## Next goals 🏆⌛
#### * Added new weather phenomena.
#### * Accuracy check for other pretrained models. 
#### * Prediction weather phenomene to images from webcam from different countries.
