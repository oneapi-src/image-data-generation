# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# pylint: disable=E0401

"""Creates frozen graph model from keras model"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def create_frozen_graph(model, sav_path):
    """Convert Keras model to ConcreteFunction"""
    full_model = tf.function(model)
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype, name="input_1"))
    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)
    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=sav_path,
                      name="saved_model.pb",
                      as_text=False)


def main():
    """
    Main Function
    """
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--save_path',
                        type=str,
                        required=False,
                        default='./models/frozen',
                        help='Absolute path to the folder to save the frozen graph model')

    parser.add_argument('-model',
                        '--model_path',
                        required=False,
                        type=str,
                        default="./models/Stage2_Models/pix2pix_g_epoch_500.h5",
                        help='Give the path of the Generator model to be used, \
                             which is in (.h5) file format.')

    args = parser.parse_args()
    save_path = args.save_path
    model_path = args.model_path

    # checking and creating the save_path and out_path
    os.makedirs(save_path, exist_ok=True)

    # load keras model
    generator = load_model(model_path, compile=False)

    # create frozen graph
    create_frozen_graph(generator, save_path)


if __name__ == "__main__":
    main()
