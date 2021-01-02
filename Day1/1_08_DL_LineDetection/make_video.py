import os
import cv2
from natsort import natsorted
import numpy as np

fps = 20

base_path = './data/test_set/clips/'
video_list = os.listdir(base_path)

for video_folder in video_list:
    # video_path = os.path.join(video_list, video_folder)
    video_path = base_path+video_folder
    video_path = video_path + '/'
    frame_list = os.listdir(video_path)

    frame_list = natsorted(frame_list)
    print(frame_list)
    count = 0
    total_frame = []

    out = cv2.VideoWriter("./data/video/test/" + video_folder + '_fps:' + str(fps) + '.mp4',
                          cv2.VideoWriter_fourcc(*'DIVX'), fps, (1280, 720))
    for frames in frame_list:
        count+=1
        print(frames)
        data_path = os.path.join(video_path, frames)
        data_path = data_path + '/'
        # data_path = os.path.join(path2, frames)
        # data_path = data_path + '/'
        # image_list = os.listdir(data_path)

        print(data_path)
        image_list = ['1.jpg', '2.jpg', '3.jpg', '4.jpg', '5.jpg', '6.jpg',
                      '7.jpg', '8.jpg', '9.jpg', '10.jpg', '11.jpg', '12.jpg',
                      '13.jpg', '14.jpg', '15.jpg', '16.jpg', '17.jpg',
                      '18.jpg', '19.jpg', '20.jpg']

        # frame = []
        for image in image_list:
            image_path = os.path.join(data_path, image)

            img = cv2.imread(image_path)
            height, width, layers = img.shape
            size = (width, height)

            # frame.append(img)
            # total_frame.append(img)
            out.write(img)
        # for i in range(len(frame)):
        #     out.write(frame[i])
        # out.release()

    # out2 = cv2.VideoWriter("./data/vide/train" +'_fps:' + str(fps) + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    # for i in range(len(total_frame)):
    #     out2.write(total_frame[i])
    # out2.release()
    out.release()