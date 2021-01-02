#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 18-5-23 上午11:33
# @Author  : MaybeShewill-CV
# @Site    : https://github.com/MaybeShewill-CV/lanenet-lane-detection
# @File    : test_lanenet.py
# @IDE: PyCharm Community Edition
"""
test LaneNet model on single image
"""
import argparse

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages') # in order to import cv2 under python3
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy

import numpy as np
import tensorflow as tf

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger


CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str, help='The model weights path')
    parser.add_argument('--video_path', type=str, help='The input video path')

    return parser.parse_args()

def minmax_scale(input_arr):
    min_val = np.min(input_arr)
    max_val = np.max(input_arr)

    output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

    return output_arr


def test_lanenet(weights_path, video_path):
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

    net = lanenet.LaneNet(phase='test', cfg=CFG)
    binary_seg_ret, instance_seg_ret = net.inference(input_tensor=input_tensor, name='LaneNet')

    postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

    # Set sess configuration
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
    sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
    sess_config.gpu_options.allocator_type = 'BFC'

    sess = tf.Session(config=sess_config)

    # define moving average version of the learned variables for eval
    with tf.variable_scope(name_or_scope='moving_avg'):
        variable_averages = tf.train.ExponentialMovingAverage(
            CFG.SOLVER.MOVING_AVE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()

    # define saver
    saver = tf.train.Saver(variables_to_restore)

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('output_test_0530.mp4', fourcc, 25.0, (640, 480))
    with sess.as_default():
        saver.restore(sess=sess, save_path=weights_path)

        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                image = frame
                image_vis = image
                image = cv2.resize(image, (512, 256), interpolation=cv2.INTER_LINEAR)
                image = image / 127.5 - 1.0

                loop_times = 1 # 500
                for i in range(loop_times):
                    binary_seg_image, instance_seg_image = sess.run(
                        [binary_seg_ret, instance_seg_ret],
                        feed_dict={input_tensor: [image]}
                    )

                postprocess_result = postprocessor.postprocess(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    source_image=image_vis
                )
                mask_image = postprocess_result['mask_image']

                for i in range(CFG.MODEL.EMBEDDING_FEATS_DIMS):
                    instance_seg_image[0][:, :, i] = minmax_scale(instance_seg_image[0][:, :, i])
                embedding_image = np.array(instance_seg_image[0], np.uint8)

                image_vis = cv2.resize(image_vis,(640, 480))
                # cv2.imshow('mask_image', mask_image)
                cv2.imshow('src_image', image_vis)
                # cv2.imshow('instance_image', embedding_image)
                # cv2.imshow('binary_image', binary_seg_image[0] * 255)

                # image_vis = cv2.resize(image_vis, (512, 256), interpolation=cv2.INTER_LINEAR)
                # mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
                # _, mask = cv2.threshold(mask_gray, 10, 255, cv2.THRESH_BINARY)
                # mask_inv = cv2.bitwise_not(mask)
                # mask_image = cv2.bitwise_and(mask_image, mask_image, mask=mask)
                # image_vis = cv2.bitwise_and(image_vis, image_vis, mask=mask_inv)
                # result = cv2.add(mask_image, image_vis)
                # cv2.imshow('result', result)

                out.write(image_vis)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break

    sess.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    return


if __name__ == '__main__':
    args = init_args()

    test_lanenet(args.weights_path, args.video_path)
    # test_lanenet("/model/tusimple_lanenet/tusimple_lanenet.ckpt", "./data/tusimple_test_image/0.jpg")

# python tools/test_lanenet.py --weights_path /PATH/TO/YOUT/CKPT_FILE_PATH  --image_path ./data/tusimple_test_image/0.jpg

# python test_video_lanenet.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --image_path ./data/tusimple_test_image/0.jpg --video_path ./input_video/0070_fps_20.mp4
# python test_video_lanenet.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --image_path ./data/tusimple_test_image/0.jpg --video_path ./data/video/train/0601_fps:20.mp4
# python test_video_lanenet.py --weights_path ./model/tusimple_lanenet/tusimple_lanenet.ckpt --video_path ./data/video/test/0530_fps:20.mp4
