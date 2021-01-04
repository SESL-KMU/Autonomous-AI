import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
import time
slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
gpu_options = tf.GPUOptions(allow_growth=False)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()

with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)
ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
ssd_anchors = ssd_net.anchors(net_shape)

def process_image(img, select_threshold=0.3, nms_threshold=.45, net_shape=(400, 400)):
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img], feed_dict={img_input: img})
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select( rpredictions, rlocalisations, ssd_anchors, select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)
    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes

capture = cv2.VideoCapture('0015_fps 20.mp4')
while (capture.isOpened()):
    ret, img = capture.read() # ret is true or false (if video is playing then its true)
    rclasses, rscores, rbboxes = process_image(img)
    visualization.plt_bboxes2(img, rclasses, rscores, rbboxes)
    if cv2.waitKey(1) & 0xFF == ord('q'): #if we hit the "Q" key it will go to next line
        break

capture.release()
cv2.destroyAllWindows()
