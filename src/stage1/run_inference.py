# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415, E0401,R0914,E1129,E0611
# pylint: disable=E0401, C0103, W0614, W0401, E0602, W0621, E1101, C0200, R0913, C0303, I1101

# noqa: E502

"""
Run inference with benchmarks on Tensorflow native models.

"""

import time
import argparse
from glob import glob
import numpy as np
import cv2
import tensorflow as tf
from model.unet import Unet
import matplotlib.pyplot as plt
from util.load_cfg import train_cfg, test_cfg, dataset_cfg, sample_cfg
from data.test.image_patch import padding_images, img2patch_list, patchlist2image, load_test_data
from data.preprocess import preprocess



testmodel = Unet(sample_cfg["patch_size"])
ckpts = tf.train.Checkpoint(model=testmodel)
ckpts.restore(tf.train.latest_checkpoint(train_cfg["checkpoint_dir"])).expect_partial()
# testmodel.save('./models/test_saved_model')

def test_function(image_path, test_mask_dir, test_save_dir, batch_size, inf_model, totaltime_l):
    """Perform the inference on the test images

    Args:
        image_path: path of the test retinal fundus images
        test_mask_dir: path of the test mask images
        test_save_dir: path to save generated images
        batch_size: batch size for the patches of the image

    Returns:
        None
    """
    image_name = image_path.split("/")[-1].split("_")[0]
    # load and process test images 
    image = plt.imread(image_path)
    original_shape = image.shape
    ##mask = plt.imread(test_mask_dir + image_name + "_test_mask.gif")
    mask = plt.imread(test_mask_dir + image_name + "_h_mask.tif")
    mask = np.where(mask > 0, 1, 0)
    # image to patches 
    image, pad_mask = padding_images(image, mask, test_cfg["stride"])
    image = preprocess(image, pad_mask)
    test_patch_list = img2patch_list(image, test_cfg["stride"])

    # test dataloader
    test_dataset = tf.data.Dataset.from_tensor_slices(test_patch_list)
    test_dataset = test_dataset.map(load_test_data)
    test_dataset = test_dataset.batch(batch_size)
    pred_result = []
    # test process 
    #print("testing image:", int(image_name))
    binference_time = 0
    total_batches = 0

    fl = None
    for _, patch in enumerate(test_dataset):
        #patch.shape = TensorShape([64, 48, 48, 3])
        total_batches += 1
        #warmup
        if not fl:
            for _ in range(10):
                if FLAGS.inc == 1:
                    pred = inf_model.signatures["serving_default"](patch).get('Identity_1')
                else:
                    _, pred = inf_model(patch, training=False)
        fl = "Done"

        stime = time.time()
        if FLAGS.inc == 1:
            pred = inf_model.signatures["serving_default"](patch).get('Identity_1')
        else:
            _, pred = inf_model(patch, training=False)
        etime = time.time() - stime

        binference_time += etime
        #print ("Batch {} Inference time {}:".format(batch, etime))
        pred = pred.numpy()
        pred_result.append(pred)


    #print ("Average Inference time {}:".format(binference_time/total_batches))
    t = (binference_time/total_batches)
    totaltime_l.append(t)
    pred_result = np.concatenate(pred_result, axis=0)
    # patches to image
    #print("post processing:", image_name)
    pred_image = patchlist2image(pred_result, test_cfg["stride"], image.shape)
    pred_image = pred_image[:original_shape[0], :original_shape[1]]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.erode(mask.astype(np.uint8), kernel)
    pred_image = pred_image * mask
    pred_image = np.where(pred_image > test_cfg["threshold"], 1, 0)
    # visualize the test result
    plt.figure(figsize=(8, 8))
    plt.title(image_name + "-(" + str(image.shape[0]) + "," + str(image.shape[1]) + ")")
    plt.imshow(pred_image, cmap=plt.cm.gray)
    #plt.show()
    plt.imsave(test_save_dir + str(int(image_name)) + ".png", pred_image, cmap=plt.cm.gray)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-l',
                        '--load_model_path',
                        default="./models/test_saved_model/",
                        type=str,
                        required=False,
                        help="directory to load model from"
                        )

    parser.add_argument('-b',
                        '--batch_size',
                        default=64,
                        required=False,
                        type=int,
                        help="inference batch size"
                        )

    parser.add_argument('-bf16',
                        '--bf16',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 to enable bf16 capabilities, \
                                        default is 0')

    parser.add_argument('-inc',
                        '--inc',
                        default=0,
                        type=int,
                        required=False,
                        help="Enable this flag for inc model inference. Default is 0"
                        )


    FLAGS = parser.parse_args()
    batch_size = FLAGS.batch_size
    load_model_path = FLAGS.load_model_path
    if FLAGS.bf16 == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision_mkl": True})
        print("Mixed Precision Enabled!")

    test_dir = dataset_cfg["dataset_path"] + dataset_cfg["test_dir"]
    test_image_dir = test_dir + dataset_cfg["test_image_dir"]
    test_mask_dir = test_dir + dataset_cfg["test_mask_dir"]
    test_groundtruth_dir = test_dir + dataset_cfg["test_groundtruth_dir"]
    test_save_dir = test_dir + dataset_cfg["test_save_dir"]

    test_image_path_list = sorted(glob(test_image_dir + "*.jpg"))

    if load_model_path is not None:
        inf_model = tf.saved_model.load(load_model_path)
    else:
        inf_model = testmodel

    totaltime_l = []

    for i in range(len(test_image_path_list)):
        image_path = test_image_path_list[i]
        test_function(image_path, test_mask_dir, test_save_dir,
                      FLAGS.batch_size, inf_model, totaltime_l)
    s = f"""
    {'-' * 40}
    # Model Inference details:
    # Average batch inference:
    # Total Average Time (in seconds): {(sum(totaltime_l)/len(totaltime_l))}
    #   Batch size: {batch_size}
    {'-' * 40}
    """
    print(s)
    print("FINISH")
