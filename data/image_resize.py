# !/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# pylint: disable=C0415, E0401,R0914,E1129,E0611
# pylint: disable=E0401, C0103, W0614, W0401, E0602

# noqa: E502

"""Functions to resize the image
"""

import cv2
from PIL import Image
from glob import glob

directory = "./training"

train_image_path_list = glob(directory + "/images/*.jpg")
train_segmentedmaskimage_path_list = glob(directory + "/1st_manual/*_h.tif")
train_maskimage_path_list = glob(directory + "/mask/*_h_mask.tif")

for fname, sname, mname in zip(train_image_path_list, train_segmentedmaskimage_path_list, train_maskimage_path_list):
    image = Image.open(fname)
    segimage = Image.open(sname)
    maskimage = Image.open(mname)

    print(f"Original size : {image.size}") # 5464x3640

    image_resized = image.resize((512, 512))
    fnamelist = fname.split('/')
    newname = fnamelist[-1].split('.')
    image_resized.save(directory+"/images/"+newname[0]+".jpg")

    image_resized = segimage.resize((512, 512))
    snamelist = sname.split('/')
    newname = snamelist[-1].split('.')
    image_resized.save(directory+"/1st_manual/"+newname[0]+".tif")

    image_resized = maskimage.resize((512, 512))
    mnamelist = mname.split('/')
    newname = mnamelist[-1].split('.')
    image_resized.save(directory+"/mask/"+newname[0]+".tif")

for fname in train_maskimage_path_list:
    ii = cv2.imread(fname)
    gray_image = cv2.cvtColor(ii, cv2.COLOR_BGR2GRAY)
    fnamelist = fname.split('/')
    newname = fnamelist[-1].split('.')
    cv2.imwrite(directory+"/mask/"+newname[0]+".tif", gray_image)
