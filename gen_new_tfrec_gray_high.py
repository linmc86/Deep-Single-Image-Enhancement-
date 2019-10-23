from __future__ import print_function
import glob, os, sys, cv2, random, time, math
import tensorflow as tf
import numpy as np
import imageio as io
from PIL import Image
from utils.utils_lap_pyramid import *
from utils.configs import *
from utils.utilities import *
import matplotlib.pyplot as plt


lev_scale = '3'


def processimg(imgs, label, lev):
    '''############################ cropping images ############################'''
    img_rand_patches, label_rand_patches = crop_random(imgs, label,
                                                           config.data.random_patch_ratio_x,
                                                           config.data.random_patch_ratio_y,
                                                           config.data.patch_size,
                                                           config.data.random_patch_per_img)
    ''' check length '''
    assert (len(img_rand_patches) == len(label_rand_patches))

    '''############################ create laplacian pyramid ############################'''
    for i in range(len(img_rand_patches)):
        img_rand_patches[i] = lpyr_gen(img_rand_patches[i], int(lev))
        label_rand_patches[i] = lpyr_gen(label_rand_patches[i], int(lev))

    for i in range(len(img_rand_patches)):
        img_rand_patches[i], _ = lpyr_enlarge_to_top_but_bottom(img_rand_patches[i])
        label_rand_patches[i], _ = lpyr_enlarge_to_top_but_bottom(label_rand_patches[i])

    for i in range(len(img_rand_patches)):
        img_rand_patches[i] = dualize(img_rand_patches[i])
        label_rand_patches[i] = dualize(label_rand_patches[i])

    ''' check shape '''
    for i in range(len(img_rand_patches)):
        assert (np.shape(img_rand_patches[i]) == np.shape(label_rand_patches[i]))

    ''' check length'''
    assert (len(img_rand_patches) == len(label_rand_patches))
    return img_rand_patches, label_rand_patches


def gen_tfrec(lev):
    global tfrecord_path
    tfrecord_path = '/media/ict419/SSD/SICE/' + 'high_freq' + '_' + lev + '_' + config.model.tfrecord_suffix

    with tf.python_io.TFRecordWriter(tfrecord_path) as tfrecord_writer:
        """####################### hdr ldr image reading ############################"""
        p = '/home/ict419/PycharmProjects/AVSSlapnet/dataset/Dataset_Part1/'
        #p = '/home/ict419/PycharmProjects/AVSSlapnet/dataset/test_train_high/'
        labelpath = p + 'Label/'

        subdirList = sorted([os.path.join(p, o) for o in os.listdir(p) if os.path.isdir(os.path.join(p, o))])
        for item in range(len(subdirList)):
            ''' folder of each scene '''
            print('Processing folder -> ' + subdirList[item], ', folder nummber %d/%d' % (item + 1, len(subdirList)))

            imgname = subdirList[item].split('/')[-1]
            if imgname == 'Label':
                print('Found label ' + subdirList[item] + ' but ignore here.')
            else:
                fileList = sorted(glob.glob(subdirList[item] + '/*.{}'.format('JPG')))

                label = io.imread(labelpath + '{}.JPG'.format(imgname)).astype('float32')
                imgs = []

                for it in fileList:
                    print('Processing Image -> ' + it)

                    img = io.imread(it).astype('float32')

                    ''' check shape '''
                    assert(np.shape(img) == np.shape(label))

                    imgs.append(img)

                h, w, _ = np.shape(imgs[0])
                ave_img = np.zeros((h, w, 3), np.float)
                for it in imgs:
                    ave_img += it

                del imgs

                ''' bring to [0,255] '''
                ave_img = norm_0_to_255(ave_img)
                label = norm_0_to_255(label)

                ''' bring to grayscale'''
                ave_gray = cv2.cvtColor(ave_img, cv2.COLOR_RGB2GRAY)
                label_gray = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)

                '''
                img_gray = cv2.cvtColor(ave_img, cv2.COLOR_RGB2GRAY)
                label_gray = cv2.cvtColor(label_ready, cv2.COLOR_RGB2GRAY)

                plt.figure(0)
                plt.subplot(221)
                plt.imshow(ave_img)
                plt.subplot(222)
                plt.imshow(label_ready)
                plt.subplot(223)
                plt.hist(img_gray, bins=1000, histtype='step')
                plt.subplot(224)
                plt.hist(label_gray, bins=1000, histtype='step')
                plt.show()
                '''
                img_rand_patches, label_rand_patches = processimg(ave_gray, label_gray, lev)

                '''#debug
                plt.figure(0)
                plt.subplot(231)
                plt.imshow(norm_0_to_1(img_h[0]))
                plt.subplot(232)
                plt.imshow(norm_0_to_1(img_h[1]))
                plt.subplot(233)
                plt.imshow(norm_0_to_1(img_h[2]))
                plt.subplot(234)
                plt.imshow(norm_0_to_1(img_h[3]))
                plt.subplot(235)
                plt.imshow(norm_0_to_1(img_h[4]))
                plt.subplot(236)
                plt.imshow(norm_0_to_1(label_h[2]))
                plt.show()
                '''
                '''############################ store in tfrecord ############################'''
                # Iterate through all the patches.
                patch_length = len(img_rand_patches)
                for i in range(0, patch_length):
                    print('\r-- processing images patches %d / %d' % (i + 1, patch_length), end='')
                    sys.stdout.flush()
                    example = pack_example(img_rand_patches[i][0], label_rand_patches[i][0])
                    tfrecord_writer.write(example.SerializeToString())


def pack_example(img_h, label_h):
    """
    img_h , label_h --->   highlayer [256or512,256or512,3]
    """
    features = {}

    ''' high layer patches 512x512'''
    img_high_patch = np.reshape(img_h, -1)
    features['train'] = tf.train.Feature(float_list=tf.train.FloatList(value=img_high_patch))

    label_high_patch = np.reshape(label_h, -1)
    features['label'] = tf.train.Feature(float_list=tf.train.FloatList(value=label_high_patch))

    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example


def complete_time_predict(dot):
    minutes = 60  # seconds
    hours = minutes * 60  # minutes
    days = hours * 24

    if dot < minutes:
        return '%.2f seconds' % dot
    elif minutes <= dot < hours:
        return '{} minutes'.format(dot // minutes)
    elif hours <= dot < days:
        return '{} hours, {} minutes'.format(dot // hours, int(dot % hours) / minutes)
    else:
        return '{} days, {} hours'.format(dot // days, int(dot % days) / hours)


def crop_random(img, label, x, y, size, N):
    """
    This method randomly crops patches from img and label, and resize them to the size of 'size'.
    The patche size is [x*h, y*w] where h and w are height and width of original image.
    x, y are the decimal in [0, 1].  N is the number of patches to crop from the original image.
    Assume img and label are in the same shape.

    :param imgs:
    :param label:
    :param x: height ratio
    :param y: weight ratio
    :param size: patch size
    :param N: the number of patches to crop
    :return: cropped img and label patches array
    """
    imgpatchs = []
    labelpatchs = []
    h, w = np.shape(img)

    for i in range(N):
        rand_coe_h = random.random() * (y - x) + x
        rand_coe_w = random.random() * (y - x) + x

        # get width and height of the patch
        rand_h = int(h * rand_coe_h)
        rand_w = int(w * rand_coe_w)

        # the random - generated coordinates are limited in
        # h -> [0, coor_h]
        # w -> [0, coor_w]
        coor_h = h - rand_h
        coor_w = w - rand_w

        # get x and y starting point of the patch
        coor_x = int(random.random() * coor_h)
        coor_y = int(random.random() * coor_w)

        # only create patches for the high layer
        img_patch = img[coor_x:coor_x + rand_h, coor_y:coor_y + rand_w]
        # resize the patch to [size, size]
        resize_img = cv2.resize(img_patch, (size, size))
        imgpatchs.append(resize_img)

        # Create patches for the label
        label_patch = label[coor_x:coor_x + rand_h, coor_y:coor_y + rand_w]
        # resize the patch to [size, size]
        resize_label = cv2.resize(label_patch, (size, size))
        labelpatchs.append(resize_label)

    return imgpatchs, labelpatchs


def dualize(py_layers):
    freq_layer = 0
    bottom_layer = py_layers[-1]
    freq_layers = py_layers[:-1]
    for item in range(0, len(freq_layers)):
        freq_layer += freq_layers[item]

    dual_layers = [freq_layer, bottom_layer]
    return dual_layers


gen_tfrec(lev_scale)