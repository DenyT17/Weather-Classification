import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from YoutubeDownload import Download, Get_Frame
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
img_size = 260
# Declaration of video parameters and paths
link=r"https://www.youtube.com/watch?v=o_kgdCGisso&t=21s"
directory=r"Videos"
Download(link,directory)
link=r"https://www.youtube.com/watch?v=4x1bqPdUxqA"
directory=r"Videos"
Download(link,directory)
link=r"https://www.youtube.com/watch?v=3wFvFvVZkwY"
directory=r"Videos"
Download(link,directory)
link=r"https://www.youtube.com/watch?v=mXMTzrGtyck"
directory=r"Videos"
Download(link,directory)
Download(link,directory)
link=r"https://www.youtube.com/watch?v=edB9dalDDDQ&t=25s"
directory=r"Videos"
Download(link,directory)

video_path=r"Videos\Beautiful Sunrise Time lapse  Unedited  No Copyright Video  Hamilton New Zealand.mp4"
Get_Frame(video_path,'Sunrise',0.009)
video_path=r"Videos\4K Winter - scenic views  nature  joy  relaxing music  forests  snow  60fps.mp4"
Get_Frame(video_path,'Winter',0.009)
video_path=r"Videos\Beautiful Rain Raining Scenery & Neture vedio  Beautifull Rain Video  HD  Nature Video.mp4"
Get_Frame(video_path,'Rain',0.009)
video_path=r"Videos\Top 6 Sunny Day Sky Video Background  Sunny Sky Overlay -Tech Online.mp4"
Get_Frame(video_path,'Shine',0.009)
video_path=r"Videos\4K Floating CLOUDS Relaxing Nature - Fly in the sky Calming Peaceful Blue Sky.mp4"
Get_Frame(video_path,'Cloudy',0.009)

# Uploading training and test data

data_from_video=tf.keras.utils.image_dataset_from_directory('test_img_YT',
                                                                 shuffle=True,
                                                                 batch_size=batch_size,
                                                                 image_size=(img_size,img_size))
data = tf.keras.utils.image_dataset_from_directory(r'C:\Users\Dtopa\OneDrive\Pulpit\Image Classification\Weather Classification\Weather',
                                                                 shuffle=True,
                                                                 batch_size=batch_size,
                                                                 image_size=(img_size,img_size))

# Prediciton
new_model=load_model("modelB2.h5")
class_names=data.class_names
pred=new_model.predict(data_from_video)
img_count=len(data_from_video.file_paths)

for i in range(img_count):
    score = pred[i]
    print("The weather in the picture looks like: {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)))
    plt.figure(figsize=(5, 5))
    for images, labels in data_from_video.take(1):
        ax = plt.subplot()
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[np.argmax(score)])
plt.show()