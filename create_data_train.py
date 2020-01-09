import pickle
from scipy.io import loadmat
import numpy as np
from glob import glob
import os
import cv2
from tqdm import tqdm

FOLDER = '29_11'

# ======== gen train_list.txt test_list.txt ============

train_list = os.listdir("/hdd/UBD/background-images/data_generated/{}/train/29_11_train/synth/train_images/".format(FOLDER))
if not os.path.isdir(os.path.join('datasets', FOLDER)):
    os.mkdir(os.path.join('datasets', FOLDER))
with open('./datasets/{}/train_list.txt'.format(FOLDER), 'a') as f:
    for fn in train_list:
        f.write('{}\n'.format(fn))

test_list = os.listdir("/hdd/UBD/background-images/data_generated/{}/val/29_11_val/synth/img/".format(FOLDER))
with open('./datasets/{}/test_list.txt'.format(FOLDER), 'a') as f:
    for fn in test_list:
        f.write('{}\n'.format(fn))

print('=====CREATING GTS LABEL=====')
# ======== create gts label ============================
path_save_train = './datasets/{}/train_gts'.format(FOLDER)
path_save_test = './datasets/{}/test_gts'.format(FOLDER)
if not os.path.isdir(path_save_train):
    os.mkdir(path_save_train)
if not os.path.isdir(path_save_test):
    os.mkdir(path_save_test)

# train
print('PROCESS TRAINING DATA:')

mat = loadmat('/hdd/UBD/background-images/data_generated/{}/train/29_11_train/synth/bg.mat'.format(FOLDER))
imnames = mat['imnames'][0]
wordBB = np.array(mat['wordBB'][0])


for fn, words in tqdm(zip(imnames, wordBB), total=len(imnames)):
    fn = fn[0]
    words = np.transpose(words, [2,1,0]).reshape(-1,8).astype(np.int)
    path = '/hdd/UBD/background-images/data_generated/{}/train/29_11_train/synth/train_images/{}'.format(FOLDER, fn)

    img = cv2.imread(path)
    h,w = img.shape[:2]
    words = np.where(words < 0, 0, words)

    words[:,[0,2,4,6]] = np.where(words[:,[0,2,4,6]] >= w, w-1, words[:,[0,2,4,6]])

    words[:,[1,3,5,7]] = np.where(words[:,[1,3,5,7]] >= h, h-1, words[:,[1,3,5,7]])
    
    words = np.pad(words, pad_width=((0,0), (0,1)))

    np.savetxt('{}/{}.txt'.format(path_save_train, fn), words.astype(np.int), fmt='%i', delimiter=',')
    np.savetxt('{}/{}.gt'.format(path_save_train, fn.split('.')[0]), words.astype(np.int), fmt='%i', delimiter=',')


    #with open('{}.txt'.format(fn), 'a') as f:
        

# test
print('PROCESS TESTING DATA:')
mat = loadmat('/hdd/UBD/background-images/data_generated/{}/val/29_11_val/synth/bg.mat'.format(FOLDER))
imnames = mat['imnames'][0]
wordBB = np.array(mat['wordBB'][0])


for fn, words in tqdm(zip(imnames, wordBB), total=len(imnames)):
    fn = fn[0]
    words = np.transpose(words, [2,1,0]).reshape(-1,8).astype(np.int)
    path = '/hdd/UBD/background-images/data_generated/{}/val/29_11_val/synth/img/{}'.format(FOLDER, fn)

    img = cv2.imread(path)
    h,w = img.shape[:2]
    words = np.where(words < 0, 0, words)

    words[:,[0,2,4,6]] = np.where(words[:,[0,2,4,6]] >= w, w-1, words[:,[0,2,4,6]])

    words[:,[1,3,5,7]] = np.where(words[:,[1,3,5,7]] >= h, h-1, words[:,[1,3,5,7]])
    
    words = np.pad(words, pad_width=((0,0), (0,1)))

    np.savetxt('{}/{}.txt'.format(path_save_test, fn), words.astype(np.int), fmt='%i', delimiter=',')
    np.savetxt('{}/{}.gt'.format(path_save_test, fn.split('.')[0]), words.astype(np.int), fmt='%i', delimiter=',')
# ======== create gts label ============================

