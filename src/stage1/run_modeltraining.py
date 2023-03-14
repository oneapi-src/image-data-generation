# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
"""Data Anomaly Detection
"""

# !/usr/bin/env python
# coding: utf-8

# pylint: disable=E0401, C0103, W0614, W0401, E0602
# noqa: E902

import os
import time
import datetime
import tensorflow as tf
from util.dice import *
from model.unet import Unet
from data.train import dataloader
from util.load_cfg import train_cfg, dataset_cfg, sample_cfg

checkpoint_dir=train_cfg["checkpoint_dir"]
checkpoint_path=train_cfg["checkpoint_dir"]
log_dir=train_cfg["log_dir"]

if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

model=Unet(sample_cfg["patch_size"])

# Learning rate and optimizer 
cosine_decay = tf.keras.experimental.CosineDecayRestarts(
                    initial_learning_rate=train_cfg["init_lr"], 
                    first_decay_steps=12000,
                    t_mul=1000,
                    m_mul=0.5,
                    alpha=1e-5)
optimizer=tf.keras.optimizers.Adam(learning_rate=cosine_decay)

# loss function 
#loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)

# metric record
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc=tf.keras.metrics.Mean(name='train_acc')
current_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_acc=tf.keras.metrics.Mean(name='val_acc')
val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

# checkpoint 
ckpt = tf.train.Checkpoint(model=model)
ckpt.restore(tf.train.latest_checkpoint(checkpoint_path)).expect_partial()

# tensorboard writer 
log_dir=log_dir+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_writer = tf.summary.create_file_writer(log_dir)


def train_step(step, patch, groundtruth):
    """Perform the training step

    Args:
        patch: patch of training image to train
        groundtruth: groundtruth patch image

    Returns:
        None
    """
    with tf.GradientTape() as tape:
        linear, pred_seg = model(patch, training=True)
        losses = dice_loss(groundtruth, pred_seg)

    # calculate the gradient
    grads = tape.gradient(losses, model.trainable_variables)
    # bp 
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # record the training loss and accuracy 
    train_loss.update_state(losses)
    train_acc.update_state(dice(groundtruth, pred_seg))


def val_step(step, patch, groundtruth):
    """Perform the validation step

    Args:
        patch: patch of training image to train
        groundtruth: groundtruth patch image

    Returns:
        None
    """
    linear, pred_seg = model(patch, training=False)
    losses = dice_loss(groundtruth, pred_seg)

    # record the val loss and accuracy 
    val_loss.update_state(losses)
    val_acc.update_state(dice(groundtruth, pred_seg))

    tf.summary.image("image", patch, step=step)
    tf.summary.image("image transform", linear, step=step)
    tf.summary.image("groundtruth", groundtruth * 255, step=step)
    tf.summary.image("pred", pred_seg, step=step)
    log_writer.flush()

def train_function(train_dataset,val_dataset):
    """Perform the training and validation step

    Args:
        train_dataset: training images 
        val_dataset: validation images

    Returns:
        None
    """
    lr_step = 0
    last_val_loss = 2e10
    EPOCHS=train_cfg["epoch"]
    VAL_TIME=train_cfg["val_time"]

    with log_writer.as_default():
        total_training_time = 0
        for epoch in range(EPOCHS):
            # renew the recorder 
            train_loss.reset_states()
            train_acc.reset_states()
            val_loss.reset_states()
            val_acc.reset_states()

            # training 
            btraining_time = 0
            etraining_time = 0
            for tstep, (patch, groundtruth) in enumerate(train_dataset):
                stime = time.time()
                train_step(lr_step, patch, groundtruth)
                etime = time.time() - stime

                btraining_time += etime
                #tf.summary.scalar("learning_rate", 
                #optimizer.learning_rate(tf.float32).numpy(), 
                #step=lr_step)
                print('\repoch {}, batch {}, loss:{:.4f}, dice:{:.4f}'.format(
                                                            epoch + 1,
                                                            tstep,
                                                            train_loss.result(),
                                                            train_acc.result()), 
                                                            end="")
                lr_step += 1

            bvalidation_time = 0
            if (epoch + 1) % VAL_TIME == 0:
                # valid 
                for vstep, (patch, groundtruth) in enumerate(val_dataset):
                    stime = time.time()
                    val_step(lr_step, patch, groundtruth)
                    etime = time.time() - stime
                    bvalidation_time += etime
                #print('\repoch {}, batch {}, train_loss:{:.4f}, 
                #train_dice:{:.4f}, val_loss:{:.4f}, val_dice:{:.4f}'.format(
                #epoch + 1, vstep, train_loss.result(), train_acc.result(), 
                #val_loss.result(), val_acc.result()),end="")
                tf.summary.scalar("val_loss", val_loss.result(), step=epoch)
                tf.summary.scalar("val_dice", val_acc.result(), step=epoch)

                if val_loss.result() < last_val_loss:
                    ckpt.save(checkpoint_dir)
                    last_val_loss = val_loss.result()

            etraining_time = btraining_time + bvalidation_time
            print ("epoch {} training time in seconds {}".format(
                                                    epoch+1, 
                                                    etraining_time))

            total_training_time += etraining_time
            print ("epoch {} total training time in seconds {}".format(
                                                        epoch+1, 
                                                        total_training_time))

            ckpt.save(checkpoint_dir)
            last_val_loss = val_loss.result()

            print("")
            tf.summary.scalar("train_loss", train_loss.result(), step=epoch)
            tf.summary.scalar("train_dice", train_acc.result(), step=epoch)
            log_writer.flush()

def saved_m():
    testmodel = Unet(sample_cfg["patch_size"])
    ckpts = tf.train.Checkpoint(model=testmodel)
    ckpts.restore(tf.train.latest_checkpoint(train_cfg["checkpoint_dir"])).expect_partial()

    dataset1 = tf.random.uniform(shape=(1,48,48,3), minval=1, maxval=5, dtype=tf.float32)
    for _ in range(10):
        _, pred = testmodel(dataset1, training=False)
    testmodel.save('./models/test_saved_model')
    print("---------saved_model.pb created----------")

if __name__ == "__main__":

    train_dataset,val_dataset=dataloader.get_dataset(
                                            dataset_cfg,
                                            regenerate=True)

    train_dataset = train_dataset.shuffle(buffer_size=1300).prefetch(
                        train_cfg["batch_size"]).batch(train_cfg["batch_size"])
    val_dataset = val_dataset.shuffle(buffer_size=1300).prefetch(
                        train_cfg["batch_size"]).batch(train_cfg["batch_size"])

    train_function(train_dataset,val_dataset)
    saved_m()
