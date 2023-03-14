# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=E0611, W0603, E0401, C0103, R0914, R0915

"""Training the model"""

import argparse
import time
import numpy as np
from imageio import imwrite
from tqdm import tqdm
from keras import backend as K
from keras.models import Model
from keras.layers import Input, LeakyReLU, ZeroPadding2D
from keras.layers import Conv2D, Activation, Dropout
from keras.layers import Concatenate, BatchNormalization
from keras.layers import Conv2DTranspose as Deconvolution2D


def convolution(inputs, filters, step, stride=2, normal=True):
    """convolution"""
    encoder = ZeroPadding2D(padding=(1, 1))(inputs)
    encoder = Conv2D(filters, [4, 4], strides=(stride, stride), name='conv_%d' % step)(encoder)
    if normal:
        encoder = BatchNormalization(name='CBat_%d' % step)(encoder)
    encoder = LeakyReLU(alpha=0.2, name='CLRelu_%d' % step)(encoder)
    return encoder


def deconvolution(inputs, filters, step, dropout):
    """Deconvolution"""
    # _, height, width, _ = (inputs.get_shape()).as_list()
    decoder = Deconvolution2D(filters, [4, 4],
                              strides=(2, 2),
                              padding='same',
                              name='Deconv_%d' % (8 - step))(inputs)
    decoder = BatchNormalization(name='DBat_%d' % (8 - step))(decoder)
    if step == 8:
        decoder = Activation(activation='tanh')(decoder)
    else:
        decoder = LeakyReLU(alpha=0.2, name='DLRelu_%d' % (8 - step))(decoder)
    if dropout[step - 1] > 0:
        decoder = Dropout(dropout[step - 1])(decoder)
    return decoder


def generator_model():
    """Generator"""
    # Dimensions of image
    img_x, img_y = img_size, img_size
    g_inputs = Input(shape=(img_x, img_y, 3))
    encoder_filter = [64, 128, 256, 512, 512, 512, 512]
    Encoder = []

    nb_layer = len(encoder_filter)
    encoder, decoder = None, None  # added

    decoder_filter = encoder_filter[::-1]
    dropout = [0.5, 0.5, 0.5, 0, 0, 0, 0, 0]

    for i in range(nb_layer):
        if i == 0:
            encoder = convolution(g_inputs, encoder_filter[i], i + 1)
        else:
            encoder = convolution(encoder, encoder_filter[i], i + 1)
        Encoder.append(encoder)

        # Middle layer...
    middle = convolution(Encoder[-1], 512, 8)

    # Buliding decoder layers...

    for j in range(nb_layer):
        if j == 0:
            decoder = deconvolution(middle, decoder_filter[j], j + 1, dropout)
        else:
            decoder = Concatenate(axis=-1)([decoder, Encoder[nb_layer - j]])
            decoder = deconvolution(decoder, decoder_filter[j], j + 1, dropout)

    g_output = Concatenate(axis=-1)([decoder, Encoder[0]])
    g_output = deconvolution(g_output, 3, 8, dropout)

    model = Model(g_inputs, g_output)
    return model


def discriminator_model():
    """Discriminator"""
    # Dimensions of image
    img_cols, img_rows = img_size, img_size
    channels = 3
    inputs = Input(shape=(img_cols, img_rows, channels * 2))

    d = ZeroPadding2D(padding=(1, 1))(inputs)
    d = Conv2D(64, [4, 4], strides=(2, 2))(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ZeroPadding2D(padding=(1, 1))(d)
    d = Conv2D(128, [4, 4], strides=(2, 2))(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ZeroPadding2D(padding=(1, 1))(d)
    d = Conv2D(256, [4, 4], strides=(2, 2))(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ZeroPadding2D(padding=(1, 1))(d)
    d = Conv2D(512, [4, 4], strides=(1, 1))(d)
    d = LeakyReLU(alpha=0.2)(d)

    # added
    d = ZeroPadding2D(padding=(1, 1))(d)
    d = Conv2D(512, [4, 4], strides=(2, 2))(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = ZeroPadding2D(padding=(1, 1))(d)
    # Sigmoid activation
    d = Conv2D(1, [4, 4], strides=(1, 1), activation='sigmoid')(d)

    model = Model(inputs, d)
    return model


def generator_containing_discriminator(generator, discriminator):
    """ merging Generator and Discriminator"""
    img_cols, img_rows = img_size, img_size
    channels = 3
    inputs = Input((img_cols, img_rows, channels))
    x_generator = generator(inputs)

    merged = Concatenate(axis=-1)([inputs, x_generator])

    discriminator.trainable = False
    x_discriminator = discriminator(merged)

    model = Model(inputs, [x_generator, x_discriminator])

    return model


def discriminator_on_generator_loss(y_true, y_pred):
    """Cross entropy loss function used"""
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=(1, 2, 3))


def generator_l1_loss(y_true, y_pred):
    """Loss is caclulated by computing the difference between true and pred images"""
    return K.mean(K.abs(y_pred - y_true), axis=(1, 2, 3))

def generate_original(generator, original, e):
    """ prediction"""
    output = generator.predict(original)
    output = np.squeeze(output, axis=0)
    original = np.squeeze(original, axis=0)
    output = (output * 127.5) + 127.5
    original = (original * 127.5) + 127.5
    imwrite('output_%d.png' % e, output)
    imwrite('original_%d.png' % e, original)

def train(epochs, batchsize, save_path):
    """ Training"""
    # # Loads images from .npy files

    output = np.load('original.npy', allow_pickle=True)  # segmented images
    original = np.load('output.npy', allow_pickle=True)  # color eye images

    original = original.astype('float32')
    output = output.astype('float32')
    # Processes image as [0,1]
    original = (original - 127.5) / 127.5
    output = (output - 127.5) / 127.5
    print("Shape of image", original.shape[0])
    batchCount = original.shape[0] if (original.shape[0] / batchsize) < 1 \
        else int(original.shape[0] / batchsize)
    print('Epochs', epochs)
    print('Bathc_size', batchsize)
    print('Batches per epoch', batchCount)

    generator = generator_model()
    discriminator = discriminator_model()
    # # ipex optimize

    gan = generator_containing_discriminator(generator, discriminator)
    generator.compile(loss=generator_l1_loss, optimizer='RMSprop')
    gan.compile(loss=[generator_l1_loss, discriminator_on_generator_loss], optimizer='RMSprop')
    discriminator.trainable = True
    discriminator.compile(loss=discriminator_on_generator_loss, optimizer='RMSprop')
    G_loss = []
    gloss, dloss = 0, 0
    D_loss = []

    training_time = 0
    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)
        epoch_time = 0
        for _ in tqdm(range(int(batchCount))):
            random_number = np.random.randint(1, original.shape[0], size=batchsize)
            batch_original = original[random_number]
            batch_output = output[random_number]
            batch_output2 = np.tile(batch_output, (2, 1, 1, 1))
            y_dis = np.zeros((2 * batchsize, 30, 30, 1))
            y_dis[:batchsize] = 1.0
            generated_original = generator.predict(batch_output)
            concat_original = np.concatenate((batch_original, generated_original))

            dis_input = np.concatenate((concat_original, batch_output2), axis=-1)
            stime = time.time()
            dloss = discriminator.train_on_batch(dis_input, y_dis)
            print(f"dloss for epoch {e}:", dloss)
            dtime = (time.time() - stime)
            print("Discrimation training time", dtime)
            random_number = np.random.randint(1, original.shape[0], size=batchsize)
            train_output = output[random_number]
            batch_original = original[random_number]
            y_gener = np.ones((batchsize, 30, 30, 1))
            discriminator.trainable = False
            stime = time.time()
            gloss = gan.train_on_batch(train_output, [batch_original, y_gener])
            print(f"gloss for epoch {e}:", gloss)
            gtime = time.time()-stime
            print("GAN training is :", gtime)
            discriminator.trainable = True
            epoch_time = epoch_time + gtime + dtime
        print(f"Training time for epoch {e}: ", epoch_time)
        G_loss.append(gloss)
        D_loss.append(dloss)
        training_time = training_time + epoch_time
        if e in (1, 100, 500, epochs/2, epochs):
            generate_original(generator, output[0:1], e)
            # Saves weights in h5 file
            generator.save(save_path + '/pix2pix_g_epoch_%d.h5' % e)
            discriminator.save(save_path + '/pix2pix_d_epoch_%d.h5' % e)
            gan.save(save_path + '/pix2pix_gan_epoch_%d.h5' % e)
    print("Overall training Time is :", training_time)

    D_loss = np.array(D_loss)
    G_loss = np.array(G_loss)
    np.save(save_path + '/dloss.npy', D_loss)
    np.save(save_path + '/gloss.npy', G_loss)


def main():
    """ Main"""
    # Arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs',
                        '--epochs',
                        default=100,
                        type=int,
                        required=True,
                        help="Define the number of epochs"
                        )

    parser.add_argument('-bs',
                        '--batchsize',
                        required=False,
                        type=int,
                        default=8,
                        help='Define the batchsize for training')

    parser.add_argument('-sz',
                        '--size',
                        required=False,
                        type=int,
                        default=512,
                        help='Dimension of image i.e.512')
    parser.add_argument('-mp',
                        '--model_path',
                        required=False,
                        type=str,
                        default='models/Stage2_Models',
                        help='Path to save the trained models')

    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batchsize
    global img_size
    img_size = args.size
    model_save_path = args.model_path
    train(epochs, batch_size, model_save_path)


if __name__ == "__main__":
    img_size = None
    main()
