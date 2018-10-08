# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
sys.path.insert(0,'C:\\Users\\admin\\Desktop\\inception')
sys.path.insert(0,'C:\\Users\\admin\\Desktop\\inception\\models\\research\\inception')

print(sys.path)

import tensorflow as tf
import datetime
import shutil
import glob, os
import io
import numpy as np


from inception.data import build_image_data
from inception import image_processing
from inception import inception_model as inception
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

checkpoint_dir = "C:\\Users\\admin\\Desktop\\inception\\inception-v3\\"
batch_size = 100

my_image_path = "C:\\Users\\admin\\Desktop\\images\\def\\"
img_file_list = [f for f in listdir(my_image_path)  if (f.rfind('png') > -1)]
file_size = len(img_file_list)
print(file_size)


def inference_on_image(img_path):
  with tf.Graph().as_default():
    num_classes = 1001

    coder = build_image_data.ImageCoder()

    image_buffer, _, _ =  build_image_data._process_image(img_path, coder)

    image = image_processing.image_preprocessing(image_buffer, 0, False) # image -> (299, 299, 3)
    image = tf.expand_dims(image,0) # (299, 299,3) -> (1, 299, 299, 3)

    logits, _ = inception.inference(image, num_classes, for_training=False, restore_logits=True)

      
    saver = tf.train.Saver()
    with tf.Session() as tf_session:
      ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
      if ckpt and ckpt.model_checkpoint_path:
        if os.path.isabs(ckpt.model_checkpoint_path):
          # Restores from checkpoint with absolute path.
          saver.restore(tf_session, ckpt.model_checkpoint_path)
          # saver.restore(tf_session, "/tmp/train_back/model.ckpt-640")
        else:
          # Restores from checkpoint with relative path.
          saver.restore(tf_session, os.path.join(FLAGS.checkpoint_dir,
                                           ckpt.model_checkpoint_path))
      l = tf_session.run([logits])
      return l

def inference_on_multi_image():
  print("total image size {} ".format(file_size) )
  
  total_batch_size = int(file_size / batch_size + 1)
  logit_list = []

  for n in range(total_batch_size):
      print("step :{} / {}".format(n + 1, total_batch_size))
      mini_batch = img_file_list[n * batch_size: (n + 1) * batch_size]
      mini_adarr = np.ndarray(shape=(0, 299,299,3))
        
      with tf.Graph().as_default():
        num_classes = 1001

        coder = build_image_data.ImageCoder()
        for i, image in enumerate(mini_batch):
          image_buffer, _, _ =  build_image_data._process_image(my_image_path + image, coder)
          image = image_processing.image_preprocessing(image_buffer, 0, False) # image -> (299, 299, 3)
          image = tf.expand_dims(image, 0) # (299, 299,3) -> (1, 299, 299, 3)
          mini_adarr = tf.concat([mini_adarr, image], 0) 

        logits, _ = inception.inference(mini_adarr, num_classes, for_training=False, restore_logits=True)

        saver = tf.train.Saver()
        with tf.Session() as tf_session:

          ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
          if ckpt and ckpt.model_checkpoint_path:
            if os.path.isabs(ckpt.model_checkpoint_path):
              # Restores from checkpoint with absolute path.
              saver.restore(tf_session, ckpt.model_checkpoint_path)
              # saver.restore(tf_session, "/tmp/train_back/model.ckpt-640")
            else:
              # Restores from checkpoint with relative path.
              saver.restore(tf_session, os.path.join(FLAGS.checkpoint_dir,
                                               ckpt.model_checkpoint_path))

          l = tf_session.run([logits])
          for ll in l[0]:
            logit_list.append(ll)
                
  return logit_list


def show_image(predictions):
    for i in predictions:
        print(my_image_path + img_file_list[i])
        print_image(my_image_path + img_file_list[i])

def print_image(path):
    plt.figure()
    im = mpimg.imread(path)
    plt.imshow(im)

#999번째 사진 출력
print_image(my_image_path+img_file_list[3])

#모든 이미지 학습, 결과를 2차원배열 형식으로 저장
logit_list = inference_on_multi_image()

#얻어낸 2차원 list를 통해서 skcit-learn의 knn거리 계산
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(n_neighbors=10)
knn.fit(logit_list)

#999번째 data의 data->2차원 형변환
re_list=np.reshape(logit_list[999],(1,-1))

#999번째 data와 유사한 사진 검색
predict=knn.kneighbors(re_list,return_distance=False)
print(predict)


#유사 사진 출력
show_image(predict[0])



image_path_one ="C:\\Users\\admin\\Desktop\\images\\def\\CAM11 ID[0066] X=02 Y=04 DEF P=03806,01680 S=53.44 본드패드.png"
logit_one = inference_on_image(image_path_one)
print_image(image_path_one)
predict_once = knn.kneighbors(logit_one[0], return_distance=False)
show_image(predict_once[0])
