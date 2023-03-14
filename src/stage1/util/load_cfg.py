# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
import json

cfg_file=open("./data/cfg.json","rb")
cfgs=json.load(cfg_file)


dataset_cfg=cfgs["dataset"]
sample_cfg=cfgs["sample"]
train_cfg=cfgs["training"]
test_cfg=cfgs["test"]