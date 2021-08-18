import os
import h5py
import cv2
import numpy as np
ir_list = os.listdir('Ir')
vis_list = ir_list
mask_list = ir_list
s = 64
vis_list = ['Vis/'+i for i in vis_list]
ir_list = ['Ir/'+i for i in ir_list]
mask_list = ['Mask/'+i for i in mask_list]
print(len(ir_list))
# f = h5py.File('vis_ir_dataset64.h5','r')
# ori_data = f['data'][:]
# f = h5py.File('data.h5','w')
data = []
for i in range(len(vis_list)):
	vis = cv2.imread(vis_list[i],0)/255.0
	ir = cv2.imread(ir_list[i],0)/255.0
	mask = cv2.imread(mask_list[i],0)/255.0
	vis = np.transpose(vis)
	ir = np.transpose(ir)
	mask = np.transpose(mask)
	x1, x2 = vis.shape
	for i1 in range(x1//s):
		for i2 in range(x2//s):
			vis_crop = vis[i1*s:(i1+1)*s, i2*s: (i2+1)*s]
			ir_crop = ir[i1*s:(i1+1)*s, i2*s: (i2+1)*s]
			mask_crop = mask[i1*s:(i1+1)*s, i2*s: (i2+1)*s]
			# print(ir_crop.shape)

			crop_data = [vis_crop, ir_crop, mask_crop]

			data.append(crop_data)
data = np.array(data)
np.random.shuffle(data)
# print(len(data))
# print(data.shape)
# np.random.shuffle(ori_data)
# ori_data = ori_data[:1500,:,:,:]
# data = np.concatenate((data,ori_data), 0)
f = h5py.File('data.h5','w')
det = f.create_dataset('data', data=data,dtype="f")
f.close


