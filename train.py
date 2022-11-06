import os
import shutil
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from glob import glob

from dataset import *
from utils import *
from model import *
from losses import *
from callbacks import *

settings = get_settings()
globals().update(settings)

os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

set_memory_growth()

img_size = (im_size, im_size)
input_shape = (im_size, im_size, 3)

epochs = cycle_epoch * n_cycle
print('epochs:', epochs)

seedEverything(seed)
print('BATCH_SIZE:', BATCH_SIZE)

route_dataset = path_join(route, 'unzip', 'dataset')
print('route_dataset:', route_dataset)

X_train, Y_train, X_valid, Y_valid = auto_split_data(route_dataset, valid_ratio, test_ratio, seed)

train_n_images = len(Y_train)
train_dataset = build_dataset_from_X_Y(X_train, Y_train, train_with_labels, img_size,
                                        BATCH_SIZE, train_repeat, train_shuffle, train_augment, im_size_before_crop)

valid_n_images = len(Y_valid)
valid_dataset = build_dataset_from_X_Y(X_valid, Y_valid, valid_with_labels, img_size,
                                        BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

print('train_n_images', train_n_images)
print('valid_n_images', valid_n_images)

tf.keras.backend.clear_session()
tf.compat.v1.reset_default_graph()

strategy = auto_select_accelerator()

with strategy.scope():
    base = get_base_model(base_name, input_shape)
    model = create_model(input_shape, base, n_landmark, final_dropout)
    model.summary()

    losses = tf.keras.losses.MeanSquaredError()
    metrics = [tf.keras.metrics.MeanAbsoluteError()]

model.compile(optimizer=Adam(learning_rate=1e-3),
              loss=losses,
              metrics=metrics)

if pretrained is not None:
    try:
        model.load_weights(pretrained)
        print('Loaded pretrain from', pretrained)
    except:
        print('Failed to load pretrain from', pretrained)

save_path = f'best_model_facepose_{base_name}_{im_size}_{n_landmark}.h5'

callbacks = get_callbacks(monitor, mode, save_path, max_lr, min_lr, cycle_epoch, save_weights_only)

his = model.fit(train_dataset, 
                validation_data=valid_dataset,
                steps_per_epoch = train_n_images//BATCH_SIZE,
                epochs=epochs,
                verbose=1,
                callbacks=callbacks)

# metric = 'loss'
# visual_save_metric(his, metric)

# metric = 'cate_output_categorical_accuracy:'
# visual_save_metric(his, metric)

# # EVALUATE

# valid_eval = model.evaluate(valid_dataset)

# print("valid_eval", valid_eval)

# with open("valid_eval.txt", mode='w') as f:
#     for item in valid_eval:
#         f.write(str(item) + " ")




