import os
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf

from utils import *
from augment import *

def get_data_facepose(route):
    X_path = sorted(glob(path_join(route, 'images') + '/*'))
    Y_path = sorted(glob(path_join(route, 'marks') + '/*'))
    return X_path, Y_path

def auto_split_data(route, valid_ratio=0.1, test_ratio=None, seed=42):
    """
    input:
        route to the directory that its is images and marks
    output:
        X, Y
    """
    X_path, Y_path = get_data_facepose(route)
    
    df = pd.DataFrame({'image':X_path, 'label':Y_path})
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if valid_ratio is not None and test_ratio is not None:
        n_test = int(len(X_path) * test_ratio)
        n_valid = int(len(X_path) * valid_ratio)

        df_test = df[:n_test]
        df_valid = df[n_test:n_test+n_valid]
        df_train = df[n_test+n_valid:]

        X_test = df_test['image'].values
        X_valid = df_valid['image'].values
        X_train = df_train['image'].values

        Y_test = df_test['label'].values
        Y_valid = df_valid['label'].values
        Y_train = df_train['label'].values

        return X_train, Y_train, all_class, X_valid, Y_valid, X_test, Y_test

    elif valid_ratio is not None:
        n_valid = int(len(X_path) * valid_ratio)

        df_train = df[n_valid:]
        df_valid = df[:n_valid]

        X_train = df_train['image'].values
        X_valid = df_valid['image'].values

        Y_train = df_train['label'].values
        Y_valid = df_valid['label'].values

        return X_train, Y_train, X_valid, Y_valid

    else:
        df_train = df

        X_train = df_train['image'].values

        Y_train = df_train['label'].values

        return X_train, Y_train

# Load the numpy files
def load_npy(npy_path):
    feature = np.load(npy_path)
    feature = tf.cast(feature, tf.float32)
    return feature

def build_decoder(with_labels=True, target_size=(256, 256), im_size_before_crop=None):
    def decode_img_preprocess(img):
        if im_size_before_crop is None:
            img = tf.image.resize(img, target_size)
        else:
            img = tf.image.resize(img, (im_size_before_crop, im_size_before_crop))
        img = tf.cast(img, tf.float32) / 255.0
        return img

    def decode_img(path):
        """
        path to image
        """
        file_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
        img = decode_img_preprocess(img)
        return img
    
    def decode_label(label):
        """
        label: int label
        """
        label = tf.cast(label, tf.float32) / target_size[0]
        label = tf.clip_by_value(label, 0, 1)
        return label
        
    def decode_with_labels(path, label):
        return decode_img(path), decode_label(label)
    
    return decode_with_labels if with_labels else decode_img

def build_dataset(paths, labels=None, bsize=32,
                  decode_fn=None, augment=None,
                  repeat=False, shuffle=1024,
                  cache=False, cache_dir=""):
    """
    paths: paths to images
    labels: int label
    """              
    if cache_dir != "" and cache is True:
        os.makedirs(cache_dir, exist_ok=True)
    
    AUTO = tf.data.experimental.AUTOTUNE
    dataset_input = tf.data.Dataset.from_tensor_slices((paths))
    dataset_label = tf.data.Dataset.from_tensor_slices((labels))

    dset = tf.data.Dataset.zip((dataset_input, dataset_label))
    dset = dset.cache(cache_dir) if cache else dset
    dset = dset.repeat() if repeat else dset
    dset = dset.shuffle(shuffle) if shuffle else dset
    dset = dset.map(lambda x,y:(x,
                                tf.numpy_function(load_npy, [y], tf.float32)
                                ), num_parallel_calls=AUTO)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.map(lambda x,y:(augment(x),y), num_parallel_calls=AUTO) if augment is not None else dset
    dset = dset.batch(bsize)
    dset = dset.prefetch(AUTO)
    
    return dset

def build_dataset_from_X_Y(X_path, Y_int, with_labels, img_size,
                           batch_size, repeat, shuffle, augment, im_size_before_crop=None):
    decoder = build_decoder(with_labels=with_labels, 
                            target_size=img_size, im_size_before_crop=im_size_before_crop)

    augment_img = build_augment() if augment else None

    dataset = build_dataset(X_path, Y_int, bsize=batch_size, decode_fn=decoder,
                            repeat=repeat, shuffle=shuffle, augment=augment_img)

    return dataset

if __name__ == '__main__':
    import os
    from utils import *
    from multiprocess_dataset import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    settings = get_settings()
    globals().update(settings)

    # route_dataset = path_join(route, 'dataset')
    route_dataset = '/home/lap14880/hieunmt/facepose/facepose_gendata/dataset'

    img_size = (im_size, im_size)
    input_shape = (im_size, im_size, 3)

    X_train, Y_train, X_valid, Y_valid = auto_split_data(route_dataset, valid_ratio, test_ratio, seed)

    train_n_images = len(Y_train)
    train_dataset = build_dataset_from_X_Y(X_train, Y_train, train_with_labels, img_size,
                                           BATCH_SIZE, train_repeat, train_shuffle, train_augment, im_size_before_crop)

    valid_n_images = len(Y_valid)
    valid_dataset = build_dataset_from_X_Y(X_valid, Y_valid, valid_with_labels, img_size,
                                           BATCH_SIZE, valid_repeat, valid_shuffle, valid_augment)

    print(len(X_train))
    print(len(X_valid))
    print(X_train[0])
    print(Y_train[0])

    for x, y in train_dataset:
        break
    print(x)
    print(y)

    import cv2
    import numpy as np
    
    img = np.array(x[0][...,::-1]) * 255
    mark = np.int64(y[0] * im_size)
    cv2.imwrite("sample.png", img)
    draw_marks(img, mark)
    cv2.imwrite(f"sample_mark.jpg", img)

