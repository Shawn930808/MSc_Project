import  h5py
import  scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from matplotlib import cm as CM
from image import *
import json

# root is the path to ShanghaiTech dataset
root='./dataset/'

part_B_train = os.path.join(root,'train_data','images')
part_B_test = os.path.join(root,'test_data','images')
path_sets = [part_B_train,part_B_test]


img_paths  = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        img_paths.append(img_path)

for img_path in img_paths:
    print(img_path)
    with open(img_path.replace('.jpg','.JSON').replace('images','ground_truth').replace('IMG_','GT_IMG_'))as fp:

        json_data = json.load(fp)
        shapes=json_data["shapes"]
        #point = shapes[0]['points']
        #print(shapes)
        #point = shapes[1]['points']
        #print(point)
        #print(shapes)
        n=len(shapes)
        #print(n)
        gt=[]
        for l in range(0,n):
            point = shapes[l]['points']
    #mat = io.loadmat(img_path.replace('.jpg','.JSON').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
            gt.append(point[0])

        print(gt)
    img = plt.imread(img_path)
    print(img.shape)
    k = np.zeros((img.shape[0], img.shape[1]))
    #print("k=",k)
    #print(gt[1][1])
    print(len(gt))
    for i in range(0, len(gt)):
        #print("gt:",gt[i][1],gt[i][0])
        if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
            #print(1)
            #print("i",i)
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter(k, 15)
    #print(k.shape)
    #k=np.reshape([768,1024])
    #print("k1=",k)
    with h5py.File(img_path.replace('.jpg', '.h5').replace('images', 'ground_truth'), 'w') as hf:
            hf['density'] = k
    #gt = mat["image_info"][0,0][0,0][0]
            groundtruth = np.asarray(hf['density'])
            print(groundtruth.shape)
            plt.figure(2)

            plt.imshow(groundtruth, cmap=CM.jet)
            plt.show()