# !/usr/bin/bash
# -*- coding: utf-8 -*-

# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause

# Copy the retinal fundus images to images directory
cd images
for f in [0]*_h.jpg; do mv -v "$f" "${f:1}"; done
cd ../
cp images/*_h.jpg training/images

# Copy the segmentation images to 1st_manual directtory
cd manual1
for f in [0]*_h.tif; do mv -v "$f" "${f:1}"; done
cd ../
cp manual1/*_h.tif training/1st_manual

# Copy the image masks to the mask directory
cd mask
for f in [0]*_h_mask.tif; do mv -v "$f" "${f:1}"; done
cd ../
cp mask/*_h_mask.tif training/mask
