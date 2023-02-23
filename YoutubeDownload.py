from pytube import YouTube
import cv2
import os

# Function for downloading videos from YouTube
def Download(link,path):
    youtubeObject = YouTube(link)
    youtubeObject = youtubeObject.streams.get_highest_resolution()
    youtubeObject.download(path)
    print("Download is completed successfully, video : {0} has been saved in {1}"
          .format(youtubeObject.title,path))
# Function for extract image from video
def Get_Frame(path,img_name,img_per_s):

    cam = cv2.VideoCapture(path)
    # Calculation of video duration
    fps = cam.get(cv2.CAP_PROP_FPS)
    frame_count = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    # Creating new folder from images, if it doesn't exist
    if not os.path.exists('test_img_YT/'+img_name):
        os.makedirs('test_img_YT/'+img_name)

    currentframe = 1
    img_number=0

    # Saving chosen number of images per second
    while (True):
        ret, frame = cam.read()

        if ret:
            if currentframe % (30/img_per_s) ==0:
                name = 'test_img_YT/'+img_name+'/'+ img_name + str(img_number) + '.jpg'
                print('Creating...' + name)
                cv2.imwrite(name, frame)
                img_number+=1
            currentframe += 1
        else:
            break
    cam.release()
    cv2.destroyAllWindows()


