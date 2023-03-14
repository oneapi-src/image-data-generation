# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=E0401, I1101, C0103

"""
Quantifying the frozen models using neural compressor
"""


import os
import argparse
import numpy as np
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
import tensorflow as tf
from keras import backend as K
from tensorflow.python.tools.optimize_for_inference_lib import optimize_for_inference
from tensorflow.python.framework import dtypes
import cv2

from neural_compressor.data import DataLoader
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig



def calculate_fid(images1, images2):
    """Calculates fid score"""
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


def discriminator_on_generator_loss(y_true, y_pred):
    """Cross entropy loss function used"""
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=(1, 2, 3))


def generator_l1_loss(y_true, y_pred):
    """generator l1 loss"""
    return K.mean(K.abs(y_pred - y_true), axis=(1, 2, 3))


def generate_original_inference(gen, orig):
    """Uses the generator method to synthetically generate the images"""
    orig = (orig - 127.5) / 127.5
    output = gen.predict(orig)
    output = np.squeeze(output, axis=0)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return output


class Dataset:
    """Creating Dataset class for getting Image and labels"""

    def __init__(self, org, out):
        self.original = org
        self.target = out

    def __getitem__(self, index):
        return self.original[index], self.target[index]

    def __len__(self):
        return len(self.original)

    def eval_func(self, model):
        """ eval_func """
        fid_metric = 0
        # Defining input & output nodes of model
        INPUTS, OUTPUTS = 'input_1', 'Identity'
        output_graph = optimize_for_inference(model.as_graph_def(), [INPUTS], [OUTPUTS],
                                              dtypes.float32.as_datatype_enum, False)
        # Initializing session
        tf.import_graph_def(output_graph, name="")
        l_input = model.get_tensor_by_name('input_1:0')  # Input Tensor
        l_output = model.get_tensor_by_name('Identity:0')  # Output Tensor
        config1 = tf.compat.v1.ConfigProto()
        sess = tf.compat.v1.Session(graph=model, config=config1)

        for i in range(len(self.original)):
            y_true = (self.target[i] * 127.5) + 127.5
            y_true = cv2.cvtColor(y_true, cv2.COLOR_BGR2RGB)
            Session_out = sess.run(l_output,
                                   feed_dict={l_input: np.expand_dims(self.original[i], axis=0)})
            y_true = (y_true - 127.5) / 127.5
            y_pred = (Session_out - 127.5) / 127.5
            y_true = cv2.cvtColor(y_true, cv2.COLOR_BGR2GRAY)
            y_pred = cv2.cvtColor(np.squeeze(y_pred, axis=0), cv2.COLOR_BGR2GRAY)
            fid_metric += calculate_fid(y_pred, y_true)

        return fid_metric / len(self.original)


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',
                        '--model_path',
                        required=True,
                        type=str,
                        default="./models/frozen/saved_model.pb",
                        help='Give the path of the Generator model to be used, \
                        which is a frozen model (.pb)')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        required=False,
                        default='./models/inc',
                        help="Output quantized model will be save as\
                         'quantized_SyncImgGen.pb'.")
    parser.add_argument('-d',
                        '--data_path',
                        type=str,
                        required=False,
                        default='./',
                        help='Absolute path to the dataset folder containing npy files,\
                             created while preprocessing.')

    FLAGS = parser.parse_args()
    model_path = FLAGS.model_path
    data_path = FLAGS.data_path
    out_path = FLAGS.out_path

    # checking and creating the save_path and out_path
    os.makedirs(out_path, exist_ok=True)

    # Loading the original and ground truth images' tensors
    original = np.load(data_path + './src/stage2/original_test.npy',
                       allow_pickle=True)  # Segmented Image as input to saved generator model
    color_image = np.load(data_path + './src/stage2/output_test.npy',
                          allow_pickle=True)  # Ground Truth image
    original = original.astype('float32')
    color_image = color_image.astype('float32')

    #quantization
    dataset = Dataset(original, color_image)
    config = PostTrainingQuantConfig()
    eval_func = dataset.eval_func
    quantized_model = fit(
        model=model_path,
        conf=config,
        calib_dataloader=DataLoader(framework='tensorflow', dataset=dataset)
        , eval_func=eval_func)
    quantized_model.save(out_path+'/quantized_SyncImgGen')
    print("Finish")

