# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Loads a sample RGB video & flow video and classifies using a trained Kinetics checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import i3d
import time
import argparse

import random
import re
import os
import tempfile
import cv2


_IMAGE_SIZE = 224

_SAMPLE_VIDEO_FRAMES = 79
_SAMPLE_PATHS = {
    'rgb': 'data/v_CricketShot_g04_c01_rgb.npy',
    'flow': 'data/v_CricketShot_g04_c01_flow.npy',
}

_CHECKPOINT_PATHS = {
    'rgb': 'data/checkpoints/rgb_scratch/model.ckpt',
    'rgb600': 'data/checkpoints/rgb_scratch_kin600/model.ckpt',
    'flow': 'data/checkpoints/flow_scratch/model.ckpt',
    'rgb_imagenet': 'data/checkpoints/rgb_imagenet/model.ckpt',
    'flow_imagenet': 'data/checkpoints/flow_imagenet/model.ckpt',
}

_LABEL_MAP_PATH = 'data/label_map.txt'
_LABEL_MAP_PATH_600 = 'data/label_map_600.txt'

# FLAGS = tf.flags.FLAGS

# tf.flags.DEFINE_string('eval_type', 'joint', 'rgb, rgb600, flow, or joint')
tf.flags.DEFINE_boolean('imagenet_pretrained', True, '')

def crop_center_square(frame):
  y, x = frame.shape[0:2]
  min_dim = min(y, x)
  start_x = (x // 2) - (min_dim // 2)
  start_y = (y // 2) - (min_dim // 2)
  return frame[start_y:start_y+min_dim,start_x:start_x+min_dim]

def load_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1, 0]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0

def load_flow_video(path, max_frames=0, resize=(224, 224)):
  cap = cv2.VideoCapture(path)
  frames = []
  try:
    while True:
      ret, frame = cap.read()
      if not ret:
        break
      frame = crop_center_square(frame)
      frame = cv2.resize(frame, resize)
      frame = frame[:, :, [2, 1]]
      frames.append(frame)
      
      if len(frames) == max_frames:
        break
  finally:
    cap.release()
  return np.array(frames) / 255.0


def print_video(path,acc_text,act_text,max_frames=0):
  cap = cv2.VideoCapture(path)
  if cap.isOpened() is False:
        print("Error opening video stream or file")
  while cap.isOpened():
    ret_val, image = cap.read()

    if(ret_val==True):
      image=cv2.resize(image,(500,500))

      cv2.putText(image, "Action: %s" %act_text, (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      cv2.putText(image, "Accuracy: %f" %acc_text, (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
      
      # cv2.namedWindow('Action_Recognition',cv2.WINOW_NORMAL)
      # cv2.resizeWindow('Action_Recognition', 320, 240)
      cv2.imshow('Action_Recognition', image)


      cv2.waitKey(100)
        # break
      if cv2.waitKey(1) == 27:
          break
    elif(ret_val==False):
      break

  
 # cv2.namedWindow(MAZE_NAME, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(MAZE_NAME, 900,900)
    # cv2.imshow(MAZE_NAME,image)
    # cv2.moveWindow(MAZE_NAME,100,100)


# def main(unused_argv):


if __name__ == '__main__':
  sttime = time.time()
  parser= argparse.ArgumentParser(description='I3d_action_recognition')

  parser.add_argument('--eval_type',type=str,default='rgb',help='Input the required evluation type(rgb, flow, or joint')
  parser.add_argument('--rgbvideo',type=str,default='')
  parser.add_argument('--flowvideo',type=str,default='')
  
  parser.add_argument('--imagenet_pretrained',type=bool,default='True')
  
  args=parser.parse_args()


  
 
  tf.logging.set_verbosity(tf.logging.INFO)
  eval_type = args.eval_type

  imagenet_pretrained = args.imagenet_pretrained

  NUM_CLASSES = 400
  # if eval_type == 'rgb600':
  #   NUM_CLASSES = 600

  if eval_type not in ['rgb', 'rgb600', 'flow', 'joint']:
    raise ValueError('Bad `eval_type`, must be one of rgb, rgb600, flow, joint')

  if eval_type == 'rgb600':
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH_600)]
  else:
    kinetics_classes = [x.strip() for x in open(_LABEL_MAP_PATH)]

  if eval_type in ['rgb', 'rgb600', 'joint']:
    # RGB input has 3 channels.

    rgbinput = load_video(args.rgbvideo)
    print('RGB video shape :',rgbinput.shape)
    print('RGB Number of frames :',rgbinput.shape[0])
    # print(rgbinput)
    rgb_model_input = np.expand_dims(rgbinput, axis=0)

    rgb_input = tf.placeholder(
        tf.float32,
        shape=(None, None, _IMAGE_SIZE, _IMAGE_SIZE, 3))


    with tf.variable_scope('RGB'):
      rgb_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      rgb_logits, _ = rgb_model(
          rgb_input, is_training=False, dropout_keep_prob=1.0)


    rgb_variable_map = {}
    for variable in tf.global_variables():

      if variable.name.split('/')[0] == 'RGB':
        if eval_type == 'rgb600':
          rgb_variable_map[variable.name.replace(':0', '')[len('RGB/inception_i3d/'):]] = variable
        else:
          rgb_variable_map[variable.name.replace(':0', '')] = variable

    rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)

  if eval_type in ['flow', 'joint']:
    # Flow input has only 2 channels.
    flowinput=load_flow_video(args.flowvideo)
    print('flow video shape :', flowinput.shape)
    print('flow number of frames :', flowinput.shape[0])
    flow_model_input = np.expand_dims(flowinput,axis=0)
    

    flow_input = tf.placeholder(
        tf.float32,
        shape=(None, None, _IMAGE_SIZE, _IMAGE_SIZE, 2))
    with tf.variable_scope('Flow'):
      flow_model = i3d.InceptionI3d(
          NUM_CLASSES, spatial_squeeze=True, final_endpoint='Logits')
      flow_logits, _ = flow_model(
          flow_input, is_training=False, dropout_keep_prob=1.0)
    flow_variable_map = {}
    for variable in tf.global_variables():
      if variable.name.split('/')[0] == 'Flow':
        flow_variable_map[variable.name.replace(':0', '')] = variable
    flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)

  if eval_type == 'rgb' or eval_type == 'rgb600':
    model_logits = rgb_logits
  elif eval_type == 'flow':
    model_logits = flow_logits
  else:
    model_logits = rgb_logits + flow_logits
  model_predictions = tf.nn.softmax(model_logits)

  with tf.Session() as sess:
    
    feed_dict = {}
    if eval_type in ['rgb', 'rgb600', 'joint']:
      if imagenet_pretrained:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS['rgb_imagenet'])
      else:
        rgb_saver.restore(sess, _CHECKPOINT_PATHS[eval_type])
      tf.logging.info('RGB checkpoint restored')
      rgb_sample = rgb_model_input
      tf.logging.info('RGB data loaded, shape=%s', str(rgb_sample.shape))
      feed_dict[rgb_input] = rgb_sample

    if eval_type in ['flow', 'joint']:
      if imagenet_pretrained:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow_imagenet'])
      else:
        flow_saver.restore(sess, _CHECKPOINT_PATHS['flow'])
      tf.logging.info('Flow checkpoint restored')
      flow_sample = flow_model_input
      tf.logging.info('Flow data loaded, shape=%s', str(flow_sample.shape))
      feed_dict[flow_input] = flow_sample

    out_logits, out_predictions = sess.run(
        [model_logits, model_predictions],
        feed_dict=feed_dict)
    # print('logits', out_logits)
    # print('predictions', out_predictions)

    out_logits = out_logits[0]
    # print(out_logits)
    out_predictions = out_predictions[0]
    # print(out_predictions)
    sorted_indices = np.argsort(out_predictions)[::-1]
    entime=time.time()
    print('Norm of logits: %f' % np.linalg.norm(out_logits))
    print('\nTop classes and probabilities')
    topaction=sorted_indices[0]
    # print('\n',topaction,'\n',sorted_indices.shape)
    print_video(args.rgbvideo,out_predictions[topaction]*100,kinetics_classes[topaction])
    # cv2.imshow() 
    for index in sorted_indices[:10]:
      print(index,' | ',out_predictions[index] * 100,'  | ', out_logits[index],'  | ', kinetics_classes[index])
  
  

  print("\n calculation time taken :",entime-sttime)
