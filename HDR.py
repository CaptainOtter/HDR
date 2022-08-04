import cv2 as cv
import os, sys
import argparse
import numpy as np

hdr_list = []
parser = argparse.ArgumentParser(description = "Given a directory, generate an Excel file with the list of files and the amount of storage used")
parser.add_argument("-d",dest = "folder", action = "store", help = "the folder we want to get store usage", required = True)

args = parser.parse_args()
top_root = args.folder

for im in os.scandir(top_root+"/HDR/"):
    img = cv.imread(top_root+"/HDR/"+im.name)
    width = int(img.shape[1] * 0.5)
    height = int(img.shape[0] * 0.5)
    dim = (width, height)
    hdr_list.append(cv.resize(img,dim))
    
exposure_times = np.array([1.0/100, 1.0/125, 1.0/60, 1.0/200, 1.0/50], dtype=np.float32)
merge_debevec = cv.createMergeDebevec()

hdr_debevec = merge_debevec.process(hdr_list, times=exposure_times)

merge_robertson = cv.createMergeRobertson()
hdr_robertson = merge_robertson.process(hdr_list, times=exposure_times)

tonemap1 = cv.createTonemap(gamma=2.2)
res_debevec = tonemap1.process(hdr_debevec.copy())
res_robertson = tonemap1.process(hdr_robertson.copy())
# Exposure fusion using Mertens
merge_mertens = cv.createMergeMertens()
res_mertens = merge_mertens.process(hdr_list)

res_debevec_8bit = np.clip(res_debevec*255, 0, 255).astype('uint8')
res_robertson_8bit = np.clip(res_robertson*255, 0, 255).astype('uint8')
res_mertens_8bit = np.clip(res_mertens*255, 0, 255).astype('uint8')
print("saving images")
cv.imwrite("ldr_debevec.jpg", res_debevec_8bit)
cv.imwrite("ldr_robertson.jpg", res_robertson_8bit)
cv.imwrite("fusion_mertens.jpg", res_mertens_8bit)