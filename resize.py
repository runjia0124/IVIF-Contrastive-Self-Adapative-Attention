import cv2
import os
import re
from PIL import Image
import glob
import torch
import numpy as np

# convert rgb (224,224,3 ) to gray (224,224) image
def rgb2gray(rgb):
        return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]) #分别对应通道 R G B


root_A = "D:/Senior/IVIF/IVIF_data/CTest/vis"
listA = os.listdir(root_A)


for i, file in enumerate(listA):

    a = cv2.imread(root_A + '/' + file)
    # a = cv2.resize(a, (360, 270))
    # a = rgb2gray(a)
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
    # print(a.shape)
    # print(a[0])
    # print(a[1])
    # print(a[2])
    cv2.imwrite(root_A + '/' + file, a)

    # image_file = Image.open(root_A + '/' + file) # open colour image
    # image_file = image_file.convert('1') # convert image to black and white
    # image_file.save(root_A + '/'+'result.png')

