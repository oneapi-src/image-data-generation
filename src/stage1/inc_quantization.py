# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=E0401, E0611, E0602, W0612, C0103, W0611, W0401

'''
inc stage 1 quantization
'''


import argparse
import numpy as np
import tensorflow as tf
from util.dice import *
from util.load_cfg import train_cfg, test_cfg, dataset_cfg, sample_cfg
from data.train import dataloader

from neural_compressor.data import Datasets
from neural_compressor.data import DataLoader
from neural_compressor.quantization import fit
from neural_compressor.config import PostTrainingQuantConfig



def dice_l(patch, groundtruth, model):
    """
    caluclates dice_loss
    """
    pred_seg = model.signatures["serving_default"](patch).get('Identity_1')
    losses = dice_loss(groundtruth, pred_seg)  # losses.numpy() to get value
    return losses.numpy()

class Dataset:
    """Creating Dataset class for getting Image and labels"""

    def __init__(self, train_dataset):
        self.patch_lst = []
        self.target_lst = []
        for _, (patch, groundtruth) in enumerate(train_dataset):
            self.patch_lst.append(patch)
            self.target_lst.append(groundtruth)

    def __getitem__(self, index):
        return self.patch_lst[index], self.target_lst[index]

    def __len__(self):
        return len(self.patch_lst)

    def run_evaluation(self, model):
        """ eval_func """
        for index in range(0, len(self.patch_lst)):
            # loss = val_result(tf.squeeze(self.patch_lst[index]), self.target_lst[index], model)
            loss = dice_l(tf.expand_dims(self.patch_lst[index], axis=0),
                          self.target_lst[index], model)
        return loss

def get_model_inputs_outputs():
    """ Get model inputs and outputs """
    saved_model_path = model_path
    model = tf.saved_model.load(saved_model_path)
    func = model.signatures["serving_default"]
    input_dict = func.structured_input_signature[1]
    output_dict = func.structured_outputs
    return list(input_dict.keys()), list(output_dict.keys())

def test_inc(trn):
    """ runs quantization """
    dataset = Dataset(trn)
    inputs, outputs = get_model_inputs_outputs()
    config = PostTrainingQuantConfig(
        inputs=['args_0'],  #inputs,
        outputs=['Identity', 'Identity_1'],  # outputs,
        calibration_sampling_size=[20],
        excluded_precisions=["bf16"])
    eval_func = dataset.run_evaluation
    quantized_model = fit(
        model=model_path,
        conf=config,
        calib_dataloader=DataLoader(framework='tensorflow',
                                    dataset=dataset,
                                    batch_size=train_cfg["batch_size"]), eval_func=eval_func)

    quantized_model.save(out_path)
    print('Finish!')


def main():
    """main"""
    train_dataset, _ = dataloader.get_dataset(
        dataset_cfg,
        regenerate=True)

    test_inc(train_dataset)


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model',
                        '--model_path',
                        required=True,
                        type=str,
                        default="./models/test_saved_model/",
                        help='Give the path of the Generator model to be used, \
                            which is a saved model (.pb)')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        required=False,
                        default="./models/stage1_quantized_model",
                        help="Output quantized model will be save as\
                             'saved_model.pb'in stage1_quantized_model.")

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

    main()
