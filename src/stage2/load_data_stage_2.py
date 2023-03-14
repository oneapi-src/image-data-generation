# Start
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

import os
from PIL import Image
import argparse
import numpy as np
import cv2
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)



def load_data_training(trn_num, tst_num):


    original = []
    target = []
    file_names = []
    file_extn = ('.jpg', '.JPG', '.png', '.PNG')

    # Read the Segmented images. Make sure to give the appropriate image path and extensions (viz. jpg or png)
    for file in sorted(os.listdir(os.path.join(directory + 'pred_result'))):

        if file.endswith(file_extn):
            img = cv2.imread(directory + '/pred_result/' + file)
            file_names.append(file)
            original.append(img)

    for file in file_names:
        img = cv2.imread(directory + "/images/" + file[:-4] + "_h.jpg")
        target.append(img)

    orig = original[:trn_num]
    torig = original[len(original) - tst_num:]
    targ = target[:trn_num]
    ttarg = target[len(target) - tst_num:]

    np.save('original.npy', orig)
    np.save('output.npy', targ)
    np.save('original_test.npy', torig)
    np.save('output_test.npy', ttarg)
    return np.asarray(original), np.asarray(target)


if __name__ == "__main__":
    # Arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-count',
                        '--datacount',
                        default=1,
                        type=int,
                        required=True,
                        help="Total number of images"
                        )

    parser.add_argument('-path',
                        '--datapath',
                        required=True,
                        type=str,
                        default="'../MEDNETPAIRS/'",
                        help='Give the path of the data')

    parser.add_argument('-split',
                        '--splitpercent',
                        default=0.8,
                        required=False,
                        type=float,
                        help='Train-Test split percentage')


    FLAGS = parser.parse_args()
    data_count = FLAGS.datacount
    directory = FLAGS.datapath
    split = FLAGS.splitpercent

    train_num = int(split*data_count)
    test_num = data_count - train_num

    load_data_training(train_num, test_num)

