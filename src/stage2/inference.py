# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=E0401, I1101, C0103, C0200, E1129, W0622

##stage2##

"""Benchmarking for all the models"""

import time
import argparse
import os
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from imageio import imwrite
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2



def calculate_fid(images1, images2):
    """calculates fid score"""
    # calculate mean and covariance statistics
    mean1, sigma1 = images1.mean(axis=0), cov(images1, rowvar=False)
    mean2, sigma2 = images2.mean(axis=0), cov(images2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = np.sum((mean1 - mean2) ** 2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid


def discriminator_on_generator_loss(y_tr, y_prd):
    """Cross entropy loss function used"""
    return K.mean(K.binary_crossentropy(y_prd, y_tr), axis=(1, 2, 3))


def generator_l1_loss(y_tr, y_prd):
    """Loss is calculated by computing the difference between true and pred images"""
    return K.mean(K.abs(y_prd - y_tr), axis=(1, 2, 3))


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',
                        '--model_path',
                        required=False,
                        type=str,
                        default="models/Stage2_Models/pix2pix_g_epoch_100.h5",
                        help="Give the path of the Generator model to be used, it \
                        can either be keras model(.h5) or it can be frozen graph model (.pb)."
                             "Note to set \"-nfrz\" if not using frozen graph models ")
    parser.add_argument('-b',
                        '--batch_size',
                        default=8,
                        required=False,
                        type=int,
                        help="inference batch size")
    parser.add_argument('-nfrz',
                        '--not_frozen_graph',
                        action='store_true',
                        help='Sets True if model passed is not frozen graph ie.. ".pb"')
    parser.add_argument('-bf16',
                        '--bf16',
                        type=int,
                        required=False,
                        default=0,
                        help='use 1 to enable bf16 capabilities, \
                                    default is 0')

    FLAGS = parser.parse_args()
    model_path = FLAGS.model_path
    batch_size = FLAGS.batch_size
    not_frozen_graph = FLAGS.not_frozen_graph

    if FLAGS.bf16 == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision_mkl": True})
        print("Mixed Precision Enabled!")

    # Loading the segmented and ground_truth images' as numpy arrays
    original = np.load('./src/stage2/original_test.npy', allow_pickle=True)
    color_image = np.load('./src/stage2/output_test.npy', allow_pickle=True)  # Ground Truth image

    original = original.astype('float32')
    color_image = color_image.astype('float32')

    # Defining input & output nodes of model
    INPUTS, OUTPUTS = 'input_1', 'Identity'
    sess = None
    generator = None
    l_output, l_input = None, None
    fid_metric = 0
    avg_inf_btime, count = 0, 0
    avg_time = 0

    # loading model
    if not_frozen_graph:
        generator = load_model(model_path, compile=False)
    else:
        # Load frozen graph using TensorFlow 1.x functions
        with tf.Graph().as_default() as graph:
            # Load the graph in graph_def
            with tf.io.gfile.GFile(model_path, "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                loaded = graph_def.ParseFromString(f.read())
                config = tf.compat.v1.ConfigProto()
                sess = tf.compat.v1.Session(graph=graph, config=config)
                sess.graph.as_default()
                tf.import_graph_def(graph_def, input_map=None,
                                    return_elements=None,
                                    name="",
                                    op_dict=None,
                                    producer_op_list=None)
                l_input = graph.get_tensor_by_name('input_1:0')  # Input Tensor
                l_output = graph.get_tensor_by_name('Identity:0')  # Output Tensor
                # initialize_all_variables
                tf.compat.v1.global_variables_initializer()

    gen_img, seg_img, grd_img = None, None, None

    #WARMUP
    avgtime_list = []
    # y_pred_list = []

    # { warm up
    if not not_frozen_graph:
        for i in range(10):
            inp = (original[0] - 127.5) / 127.5
            input = np.expand_dims(inp, axis=0)
            input = np.repeat(input, batch_size, axis=0)
            y_pred = sess.run(l_output, feed_dict={l_input: input})
    # }

    for i in range(10):
        for indx in range(len(original)):
            count += 1
            y_pred = None
            grd_tr = color_image[indx]
            y_true = (grd_tr * 127.5) + 127.5
            y_true = cv2.cvtColor(y_true, cv2.COLOR_BGR2RGB)
            grd_img = y_true
            # avg_time += avg_inf_btime / batch_size
            avg_time += avg_inf_btime

            input = np.expand_dims(original[indx], axis=0)
            input = np.repeat(input, batch_size, axis=0)
            input = (input - 127.5) / 127.5

            start_time = time.time()

            if generator:
                y_pred = generator.predict(input)
            else:
                y_pred = sess.run(l_output, feed_dict={l_input: input})

            end_time = time.time() - start_time

            y_pred = y_pred[0]
            gen_img = cv2.cvtColor(((y_pred[0] * 127.5) + 127.5), cv2.COLOR_BGR2RGB)
            avg_inf_btime += end_time
            seg_img = cv2.cvtColor(((original[indx] * 127.5) + 127.5), cv2.COLOR_BGR2RGB)
            y_true = (y_true - 127.5) / 127.5
            y_pred = (y_pred - 127.5) / 127.5
            y_true = cv2.cvtColor(y_true, cv2.COLOR_BGR2GRAY)
            y_pred = cv2.cvtColor(y_pred, cv2.COLOR_BGR2GRAY)

        avgtime_list.append(avg_time/count)


    o_path = './data/Inference_results'
    os.makedirs(o_path, exist_ok=True)

    imwrite('./data/Inference_results/Ground_Truth.png', grd_img)
    imwrite('./data/Inference_results/segmented_IMAGE.png', seg_img)
    imwrite('./data/Inference_results/GENERATED_IMAGE.png', gen_img)
    fid_metric += calculate_fid(y_pred, y_true)

    print("Average 'FID' Score of the model ---> ", fid_metric)
    s = f"""
    {'-' * 40}
    # Model Inference details:
    # Average batch inference:
    # Total Average Time (in seconds): {(sum(avgtime_list)/len(avgtime_list))}
    #   Batch size: {batch_size}
    {'-' * 40}
    """
    print(s)
