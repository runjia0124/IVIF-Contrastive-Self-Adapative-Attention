import time
import os
import sys
import h5py
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import imageio
import scipy.misc
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import  argparse
from train_tasks import train, finetune, train_noCrop, collab_multiNegative, finetune_multiNegative
from model import Unet_resize_conv, FusionNet, Unet_res18, Unet_train, Unet
# from model_attention import One4AllNet
# from model1 import One4AllNet
from utils import YCbCr2RGB,CbCrFusion
os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from torchvision import utils as vutils
argparser = argparse.ArgumentParser()
argparser.add_argument('--epoch', type=int, help='epoch number', default=5)
# argparser.add_argument('--n_way', type=int, help='n way', default=5)
argparser.add_argument('--num_task', type=int, help='k shot for support set', default=3)
# argparser.add_argument('--imgsz', type=int, help='imgsz', default=84)
# argparser.add_argument('--imgc', type=int, help='imgc', default=3)
# argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=1)
# argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--lr', type=float, help='task-level inner update learning rate', default=1e-4)
# argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
# argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
argparser.add_argument('--bs', type=int, help='batch size', default=10)
argparser.add_argument('--logdir', type=str, default='logs/')
argparser.add_argument('--train', action='store_true')
argparser.add_argument('--train_collab', action='store_true')
argparser.add_argument('--train_2', action='store_true')
argparser.add_argument('--train_UNet', action='store_true')
argparser.add_argument('--test', action='store_true')
argparser.add_argument('--testVisual', action='store_true')
argparser.add_argument('--test_attention', action='store_true')
argparser.add_argument('--max_pool',action='store_true')
argparser.add_argument('--resume',action='store_true')
argparser.add_argument('--finetune',action='store_true')
argparser.add_argument('--finetune_train',action='store_true')
argparser.add_argument('--finetune_multiNegative',action='store_true')
argparser.add_argument('--pretrain', action='store_true')
argparser.add_argument('--use_gpu', action='store_true')
argparser.add_argument('--gn', action='store_true')
argparser.add_argument('--dwa', action='store_true')
argparser.add_argument('--pc', action ='store_true')
argparser.add_argument('--w', action ='store_true')
argparser.add_argument('--fs', type=int, help='fusion strategy,0~6', default=0)
argparser.add_argument('--task', type=int,help='task 0,1,2(visir,me,mf)', default=0)
argparser.add_argument('--save_dir', type=int,help='1,2,3,4,5,6', default=0)
argparser.add_argument('--c1', type=float,help='weight grad', default=0.5)
argparser.add_argument('--c2', type=float,help='weight entropy', default=0.5)
argparser.add_argument('--contrast', type=float,help='contrastive loss weight', default=1.0)
argparser.add_argument('--w_loss', type=float,help='weight of self-adaptive loss', default=1.0)

args = argparser.parse_args()

# data1_path = 'Training_dataset/data.h5'
data1_path = '../new_ir_vis_data.h5'
data2_path = '../Training_dataset/oe_ue_Y_dataset64.h5'
data3_path = '../Training_dataset/far_near_Y_dataset64.h5'

data_paths = [data1_path, data2_path, data3_path]


def test(model, vis_path, ir_path, f, filepath, save_path, pre, logs=None):
	"""
	if args.save_dir == 1:
		save_path = 'results_1/' + args.logdir.split('/')[-1] + '/'
	elif args.save_dir == 2:
		save_path = 'results_2/' + args.logdir.split('/')[-1] + '/'
	elif args.save_dir == 3:
		save_path = 'results_3/' + args.logdir.split('/')[-1] + '/'
	elif args.save_dir == 4:
		save_path = 'results_4/' + args.logdir.split('/')[-1] + '/'
	elif args.save_dir == 5:
		save_path = 'results_5/' + args.logdir.split('/')[-1] + '/'
	elif args.save_dir == 6:
		save_path = 'results_6/' + args.logdir.split('/')[-1] + '/'


	if args.save_dir == 1:
		checkpath = './checkpoints_1/latest.pth'
	elif args.save_dir == 2:
		checkpath = './checkpoints_2/latest.pth'
	elif args.save_dir == 3:
		checkpath = './checkpoints_3/latest.pth'
	elif args.save_dir == 4:
		checkpath = './checkpoints_4/latest.pth'
	elif args.save_dir == 5:
		checkpath = './checkpoints_5/latest.pth'
	elif args.save_dir == 6:
		checkpath = './checkpoints_6/latest.pth'
	"""
	save_path = 'results/' + args.logdir.split('/')[-1] + '/'
	checkpath = './checkpoints/latest.pth'
	vis_list = [n for n in os.listdir(vis_path)]
	ir_list = vis_list
	# logs = torch.load('./checkpoints/latest.pth') # use checkpoints when testing
	logs = torch.load(checkpath) # use checkpoints when testing
	device = torch.device('cpu')
	if args.use_gpu:
		device = torch.device('cuda')
	# print(logs['model_state_dict']['MTencoder.Mean.weight'])
	# model = Unet().to(torch.device('cuda'))

	model.load_state_dict(logs['state_dict'])
	# model = torch.load(filepath+'1')
	model.to(device)
	# mdel.load_state_dict(logs['optimizer_state_dict'])
	# model.eval()
	# print('epoch', logs['epoch'])
	transform = transforms.Compose([
        transforms.ToTensor(), # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)), # 归一化
                             ])
	tail = vis_list[0].split('.')[-1]
	import time
	Time = []
	s = []
	for vis_, ir_ in zip(vis_list, ir_list):
		vis = vis_path + '/' + vis_
		ir = ir_path + '/' + ir_
		start = time.time()
		slis = False
		resi = False
		si = 2
		# vis_img_ = scipy.misc.imread(vis, mode='YCbCr')
		# vis_img = vis_img_[:,:,0]
		if f == 0:
			img1 = imageio.imread(vis).astype(np.float32)
			
			# w,h = vis_img.shape[0], vis_img.shape[1]
			# Cb, Cr = vis_img_[:,:,1].reshape([w,h,1]).astype(np.float),vis_img_[:,:,2].reshape([w,h,1]).astype(np.float)

			img2 = imageio.imread(ir).astype(np.float32)

			if resi:
				# print(img1.shape)
				w0,h0 = img1.shape
				img1 = imageio.imresize(img1, (w0//si, h0//si)).astype(np.float32)
				img2 = imageio.imresize(img2, (w0//si, h0//si)).astype(np.float32)
			img1_data = transform(img1)
			img2_data = transform(img2)
		if f == 1 or f ==2:
			img1 =imageio.imread(vis, mode='YCbCr').astype(np.float32)
			img2 =imageio.imread(ir, mode='YCbCr').astype(np.float32)

			w0,h0,c0 = img1.shape
			# if resi:
			# 	img1 = scipy.misc.imresize(img1, (w0//si, h0//si)).astype(np.float32)
			# 	img2 = scipy.misc.imresize(img2, (w0//si, h0//si)).astype(np.float32)

			Cb1,Cr1 = img1[:,:,1], img1[:,:,2]
			Cb2,Cr2= img2[:,:,1],img2[:,:,2]
			w,h = Cb1.shape[0], Cb1.shape[1]
			Cb = CbCrFusion(Cb1,Cb2,w,h).reshape([w,h,1])
			Cr = CbCrFusion(Cr1,Cr2,w,h).reshape([w,h,1])
			img1_ = img1[:,:,0]/255.0
			img2_ = img2[:,:,0]/255.0
			# print(Cb1,Cr1,img1_)
			img1_data = transform(img1_)
			img2_data = transform(img2_)
		if f ==3:
			img1 =scipy.misc.imread(vis, mode='YCbCr').astype(np.float32)
			Cb,Cr = img1[:,:,1], img1[:,:,2]
			w,h = Cb.shape[0], Cb.shape[1]
			img1_ = img1[:,:,0]/255.0
			# print(Cb1,Cr1,img1_)
			Cb = Cb.reshape([w,h,1])
			Cr = Cr.reshape([w,h,1])
			img1_data = transform(img1_)


		img1_data = torch.unsqueeze(img1_data, 0).to(device)
		img2_data = torch.unsqueeze(img2_data, 0).to(device)
		# print(vis_)
		# print('0000000',img2_data.shape)
		if slis:
			s = 84
			output = img1_data
			h0, w0 = img1_data.shape[2], img1_data.shape[3]
			b_h = h0//s
			b_w = w0//s
			for i in range(b_h):
				for j in range(b_w):
					output[:,:,i*s:min((i+1)*s,h0-1),j*s:min((j+1)*s,w0-1)] = model(img1_data[:,:,i*s:min((i+1)*s,h0-1),j*s:min((j+1)*s,w0-1)], img1_data[:,:,i*s:min((i+1)*s,h0-1),j*s:min((j+1)*s,w0-1)])
		else:
			output = model(img1_data, img2_data)
		torch.cuda.synchronize()



		output = np.transpose((torch.squeeze(output,0).cpu().detach().numpy()*127.5+127.5), (1,2,0)).astype(np.float32)
		if f==1 or f==2:
			R,G,B = YCbCr2RGB(output,Cb,Cr)
			output = np.concatenate((B,G,R),2)

			img1 =cv2.imread(vis).astype(np.float32)
			img2 =cv2.imread(ir).astype(np.float32)
		if f==3:
			R,G,B = YCbCr2RGB(output,Cb,Cr)
			output = np.concatenate((B,G,R),2)
		# output = cv2.hconcat([img1, img2, output])
		if not os.path.exists(save_path):
			os.makedirs(save_path)
		if resi:
			output = cv2.resize(output, (h0,w0))

		cv2.imwrite(save_path+pre+vis_.split('.')[0]+'.png', output)

		end = time.time()
		Time.append(end-start)


	print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))


def testVisual(model, vis_path, ir_path, f, filepath, save_path, pre, logs=None):
	vis_list = [n for n in os.listdir(vis_path)]
	ir_list = vis_list
	if logs == None:
		logs = torch.load(filepath)
	device = torch.device('cpu')
	if args.use_gpu:
		device = torch.device('cuda')
	model.load_state_dict(logs['state_dict'])
	model.to(device)
	transform = transforms.Compose([
		transforms.ToTensor(),  # 转为Tensor
		transforms.Normalize((0.5,), (0.5,)),  # 归一化
	])
	tail = vis_list[0].split('.')[-1]
	import time
	Time = []
	s = []
	for vis_, ir_ in zip(vis_list, ir_list):
		vis = vis_path + '/' + vis_
		ir = ir_path + '/' + ir_
		start = time.time()
		slis = False
		resi = False
		si = 2
		# vis_img_ = scipy.misc.imread(vis, mode='YCbCr')
		# vis_img = vis_img_[:,:,0]
		if f == 0:
			img1 = imageio.imread(vis).astype(np.float32)

			img2 = imageio.imread(ir).astype(np.float32)

			if resi:
				# print(img1.shape)
				w0, h0 = img1.shape
				img1 = imageio.imresize(img1, (w0 // si, h0 // si)).astype(np.float32)
				img2 = imageio.imresize(img2, (w0 // si, h0 // si)).astype(np.float32)
			img1_data = transform(img1)
			img2_data = transform(img2)
		if f == 1 or f == 2:
			img1 = imageio.imread(vis, mode='YCbCr').astype(np.float32)
			img2 = imageio.imread(ir, mode='YCbCr').astype(np.float32)

			w0, h0, c0 = img1.shape
			# if resi:
			# 	img1 = scipy.misc.imresize(img1, (w0//si, h0//si)).astype(np.float32)
			# 	img2 = scipy.misc.imresize(img2, (w0//si, h0//si)).astype(np.float32)

			Cb1, Cr1 = img1[:, :, 1], img1[:, :, 2]
			Cb2, Cr2 = img2[:, :, 1], img2[:, :, 2]
			w, h = Cb1.shape[0], Cb1.shape[1]
			Cb = CbCrFusion(Cb1, Cb2, w, h).reshape([w, h, 1])
			Cr = CbCrFusion(Cr1, Cr2, w, h).reshape([w, h, 1])
			img1_ = img1[:, :, 0] / 255.0
			img2_ = img2[:, :, 0] / 255.0
			# print(Cb1,Cr1,img1_)
			img1_data = transform(img1_)
			img2_data = transform(img2_)
		if f == 3:
			img1 = scipy.misc.imread(vis, mode='YCbCr').astype(np.float32)
			Cb, Cr = img1[:, :, 1], img1[:, :, 2]
			w, h = Cb.shape[0], Cb.shape[1]
			img1_ = img1[:, :, 0] / 255.0
			# print(Cb1,Cr1,img1_)
			Cb = Cb.reshape([w, h, 1])
			Cr = Cr.reshape([w, h, 1])
			img1_data = transform(img1_)

		img1_data = torch.unsqueeze(img1_data, 0).to(device)
		img2_data = torch.unsqueeze(img2_data, 0).to(device)
		# print(vis_)
		# print('0000000',img2_data.shape)
		if slis:
			s = 84
			output = img1_data
			h0, w0 = img1_data.shape[2], img1_data.shape[3]
			b_h = h0 // s
			b_w = w0 // s
			for i in range(b_h):
				for j in range(b_w):
					output[:, :, i * s:min((i + 1) * s, h0 - 1), j * s:min((j + 1) * s, w0 - 1)] = model(
						img1_data[:, :, i * s:min((i + 1) * s, h0 - 1), j * s:min((j + 1) * s, w0 - 1)],
						img1_data[:, :, i * s:min((i + 1) * s, h0 - 1), j * s:min((j + 1) * s, w0 - 1)])
		else:
			output = model(img1_data, img2_data)
		torch.cuda.synchronize()

		weights = 'placeholder'
		vgg = Vgg19_2(vgg19_weights=weights).cuda()
		outputs = vgg(output)
		for i, output in enumerate(outputs):
			print(output.shape)
			# output = np.expand_dims(output[0][0].cpu().detach().numpy(), 0)
			# output = torch.tensor(output)
			print(output[0].shape) # 64, 266, 486
			output = np.transpose((torch.squeeze(output[0], 0).cpu().detach().numpy() * 127.5 + 127.5), (1, 2, 0)).astype(
				np.float32)

			# output = cv2.hconcat([img1, img2, output])
			if not os.path.exists(save_path):
				os.makedirs(save_path)
			if resi:
				output = cv2.resize(output, (h0, w0))

			cv2.imwrite(save_path + str(i) + vis_.split('.')[0] + '.png', output[:,:,0]) # notice here

			end = time.time()
			Time.append(end - start)

	print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))




def main():
	print('cuda ', torch.cuda.is_available())
	print('training',args.train)
	
	# train_dataset_path = 'Medical_image/source_data/Train/'
	# train_dataset_path = 'RoadScene-master'
	# test_dataset_path = 'RoadScene-master'
	

	# transform = transforms.Compose([
	# 	transforms.ToTensor(), # 转为Tensor
	# 	transforms.Normalize((0.5,), (0.5,)), # 归一化
	# 	])
	if args.use_gpu:
		model = Unet_resize_conv().to(torch.device('cuda'))
	else:
		model = Unet_resize_conv().to(torch.device('cpu'))
	# print(model.state_dict().keys())
	tmp = filter(lambda x: x.requires_grad, model.parameters())
	num = sum(map(lambda x: np.prod(x.shape), tmp))
	# print(maml)
	print('Total trainable tensors:', num)
	
	optim = torch.optim.Adam(model.parameters(), lr = args.lr)
	if not os.path.exists(args.logdir):
		os.makedirs(args.logdir)

	filepath = os.path.join(args.logdir, 'checkpoint_model.pth.tar')
	filepath = './logs/latest.pth'
	if args.train:
		model.train()
		data_path = []
		dir_vis = "../TNO_/vis/"  # RoadScene
		dir_ir = "../TNO_/ir/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# crop: data1_path

		train(model, data1_path, optim, args.epoch, filepath, args)
		save_path = 'results/'+args.logdir.split('/')[-1]+'/'	


		# test(model,'../TNO_/vis','../TNO_/ir',0, filepath, save_path, 'TNO_')
		# test(model,'../test_imgs_/vis-ir/RoadScene/vis','../test_imgs_/vis-ir/RoadScene/ir',0, filepath, save_path, 'road_')
		# test(model,'../test_imgs_/vis-ir/RoadScene/vis_rgb','../test_imgs_/vis-ir/RoadScene/ir',1, filepath, save_path, 'road_rgb_')
		# test(model,'../test_imgs_/multi-ex_/dataset1/oe','../test_imgs_/multi-ex_/dataset1/ue',1,filepath, save_path,'me_1_')
		# test(model,'../test_imgs_/multi-ex_/dataset2/oe','../test_imgs_/multi-ex_/dataset2/ue',1,filepath, save_path,'me_2_')
		# test(model,'../test_imgs_/multi-focus/far','../test_imgs_/multi-focus/near',2,filepath, save_path,'mf_')
		# plot(args.logdir)
	elif args.train_collab:
		model.train()
		data_path = []
		dir_vis = "../TNO_/vis/"  # RoadScene
		dir_ir = "../TNO_/ir/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# crop: data1_path
		collab_multiNegative(model, data1_path, optim, args.epoch, filepath, args)
		save_path = 'results/'+args.logdir.split('/')[-1]+'/'


		test(model,'../TNO_/vis','../TNO_/ir',0, filepath, save_path, 'TNO_')
	elif args.train_2:
		model = Unet_train().to(torch.device('cuda'))
		optim = torch.optim.Adam(model.parameters(), lr=args.lr)
		model.train()
		data_path = []
		dir_vis = "../TNO_/vis/"  # RoadScene
		dir_ir = "../TNO_/ir/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# crop: data1_path
		train(model, data1_path, optim, args.epoch, filepath, args)
		save_path = 'results/' + args.logdir.split('/')[-1] + '/'

		test(model, '../TNO_/vis', '../TNO_/ir', 0, filepath, save_path, 'TNO_')
	elif args.train_UNet:
		model = Unet().to(torch.device('cuda'))
		optim = torch.optim.Adam(model.parameters(), lr=args.lr)
		model.train()
		data_path = []
		dir_vis = "../TNO_/vis/"  # RoadScene
		dir_ir = "../TNO_/ir/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# crop: data1_path
		train(model, data1_path, optim, args.epoch, filepath, args)
		save_path = 'results/' + args.logdir.split('/')[-1] + '/'

		# test(model, '../TNO_/vis', '../TNO_/ir', 0, filepath, save_path, 'TNO_')

	elif args.test:
		# model = Unet().to(torch.device('cuda'))
		# model = Unet_resize_conv().to(torch.device('cuda'))
		dir_vis = "../CTest/vis/" # RoadScene "../TNO_/vis/" ./dataset/vis/
		dir_ir = "../CTest/ir/"
		# model = Unet_pretrain().to(torch.device('cuda'))
		save_path = 'results/' + args.logdir.split('/')[-1] + '/'
		data_path = []
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# load from ./checkpoints folder
		# logs = torch.load('./checkpoints/latest.pth')
		test(model, dir_vis, dir_ir, 0, filepath, save_path, '')
		# train_test(model, data_path, optim, args.epoch, filepath, args)
	elif args.finetune:
		model.train()
		data_path = []
		dir_vis = "../Train_vis/"  # RoadScene ./dataset/vis/
		dir_ir = "../Train_ir/"
		dir_masks = "../Train_ir_mask/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		data_path.append(dir_masks)


		# load .pth in ./logs
		finetune(model, data_path, optim, args.epoch, filepath, args)
		if args.save_dir == 0:
			save_path = 'results/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 1:
			save_path = 'results_1/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 2:
			save_path = 'results_2/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 3:
			save_path = 'results_3/' + args.logdir.split('/')[-1] + '/'

		test(model, '../TNO_/vis', '../TNO_/ir', 0, filepath, save_path, 'TNO_')
	elif args.finetune_multiNegative:
		model.train()
		data_path = []
		dir_vis = "../Train_vis/"  # RoadScene ./dataset/vis/
		dir_ir = "../Train_ir/"
		dir_masks = "../Train_ir_mask/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		data_path.append(dir_masks)


		# load .pth in ./logs
		finetune_multiNegative(model, data_path, optim, args.epoch, filepath, args)
		if args.save_dir == 0:
			save_path = 'results/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 1:
			save_path = 'results_1/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 2:
			save_path = 'results_2/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 3:
			save_path = 'results_3/' + args.logdir.split('/')[-1] + '/'

		test(model, '../CTest/vis', '../CTest/ir', 0, filepath, save_path, '')
	elif args.finetune_train:
		model = Unet().to(torch.device('cuda'))
		optim = torch.optim.Adam(model.parameters(), lr=args.lr)
		model.train()
		data_path = []
		dir_vis = "../Train_vis/"  # RoadScene ./dataset/vis/
		dir_ir = "../Train_ir/"
		dir_masks = "../Train_ir_mask/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		data_path.append(dir_masks)


		# load .pth in ./logs
		finetune_multiNegative(model, data_path, optim, args.epoch, filepath, args)
		if args.save_dir == 0:
			save_path = 'results/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 1:
			save_path = 'results_1/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 2:
			save_path = 'results_2/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 3:
			save_path = 'results_3/' + args.logdir.split('/')[-1] + '/'

		test(model, '../CTest/vis', '../CTest/ir', 0, filepath, save_path, 'TNO_')
	elif args.testVisual:
		dir_vis = "../TNO_/vis/" # RoadScene "../TNO_/vis/" ./dataset/vis/
		dir_ir = "../TNO_/ir/"

		save_path = 'results/' + args.logdir.split('/')[-1] + '/'
		data_path = []
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		# load from ./checkpoints folder
		logs = torch.load('./checkpoints/latest.pth')
		testVisual(model, dir_vis, dir_ir, 0, filepath, save_path, 'TNO_', logs=logs)
		# train_test(model, data_path, optim, args.epoch, filepath, args)


	elif args.test_attention:
		dir_vis = "../TNO_/vis/"  # RoadScene
		dir_ir = "../TNO_/ir/"

		save_path = 'results/' + args.logdir.split('/')[-1] + '/'
		data_path = []
		data_path.append(dir_vis)
		data_path.append(dir_ir)
		filepath = './checkpoints/latest.pth'
		test_att(model, '../TNO_/vis', '../TNO_/ir', 0, filepath, save_path, '')
		# train_test(model, data_path, optim, args.epoch, filepath, args)

	elif args.finetune_noMask:
		model.train()
		data_path = []
		dir_vis = "../TNO_/vis/"  # RoadScene ./dataset/vis/
		dir_ir = "../TNO_/ir/"
		data_path.append(dir_vis)
		data_path.append(dir_ir)

		finetune_noMask(model, data_path, optim, args.epoch, filepath, args)
		if args.save_dir == 0:
			save_path = 'results/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 1:
			save_path = 'results_1/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 2:
			save_path = 'results_2/'+args.logdir.split('/')[-1]+'/'
		elif args.save_dir == 3:
			save_path = 'results_3/' + args.logdir.split('/')[-1] + '/'
		logs = torch.load('./checkpoints/latest.pth')
		test(model, '../TNO_/vis', '../TNO_/ir', 0, filepath, save_path, 'TNO_', logs=logs)

	else:
		# model.eval()
		save_path = 'results/'+args.logdir.split('/')[-1]+'/'
		print(save_path)
		test(model,'../TNO_/vis','../TNO_/ir',0, filepath, save_path, 'TNO_')
		# test(model,'../test_imgs_/vis-ir/RoadScene/vis','../test_imgs_/vis-ir/RoadScene/ir',0, filepath, save_path, 'road_')
		# test(model,'../test_imgs_/vis-ir/RoadScene/vis_rgb','../test_imgs_/vis-ir/RoadScene/ir',1, filepath, save_path, 'road_rgb_')
		# test(model,'../test_imgs_/multi-ex_/dataset1/oe','../test_imgs_/multi-ex_/dataset1/ue',1,filepath, save_path,'me_1_')
		# test(model,'../test_imgs_/multi-ex_/dataset2/oe','../test_imgs_/multi-ex_/dataset2/ue',1,filepath, save_path,'me_2_')
		# test(model,'../test_imgs_/multi-focus/far','../test_imgs_/multi-focus/near',2,filepath, save_path,'mf_')
		# test(model,'test_imgs/multi-focus/far_oe','test_imgs/multi-focus/near_ue',2,filepath, save_path,'mfme_')

		# plot(args.logdir)

def test_att(model, vis_path, ir_path, f, filepath, save_path, pre, logs=None):
    vis_list = [n for n in os.listdir(vis_path)]
    ir_list = vis_list
    if logs == None:
        logs = torch.load(filepath)
    device = torch.device('cpu')
    if args.use_gpu:
        device = torch.device('cuda')
    # print(logs['model_state_dict']['MTencoder.Mean.weight'])
    model.load_state_dict(logs['state_dict'])
    # model = torch.load(filepath+'1')
    model.to(device)
    # mdel.load_state_dict(logs['optimizer_state_dict'])
    # model.eval()
    # print('epoch', logs['epoch'])
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转为Tensor
        transforms.Normalize((0.5,), (0.5,)),  # 归一化
    ])
    tail = vis_list[0].split('.')[-1]
    import time
    Time = []
    s = []
    for vis_, ir_ in zip(vis_list, ir_list):
        vis = vis_path + '/' + vis_
        ir = ir_path + '/' + ir_
        start = time.time()
        slis = False
        resi = False
        si = 2
        if f == 0:
            img1 = imageio.imread(vis).astype(np.float32)


            img2 = imageio.imread(ir).astype(np.float32)

            if resi:
                # print(img1.shape)
                w0, h0 = img1.shape
                img1 = imageio.imresize(img1, (w0 // si, h0 // si)).astype(np.float32)
                img2 = imageio.imresize(img2, (w0 // si, h0 // si)).astype(np.float32)
            img1_data = transform(img1)
            img2_data = transform(img2)
        if f == 1 or f == 2:
            img1 = imageio.imread(vis, mode='YCbCr').astype(np.float32)
            img2 = imageio.imread(ir, mode='YCbCr').astype(np.float32)

            w0, h0, c0 = img1.shape
            # if resi:
            # 	img1 = scipy.misc.imresize(img1, (w0//si, h0//si)).astype(np.float32)
            # 	img2 = scipy.misc.imresize(img2, (w0//si, h0//si)).astype(np.float32)

            Cb1, Cr1 = img1[:, :, 1], img1[:, :, 2]
            Cb2, Cr2 = img2[:, :, 1], img2[:, :, 2]
            w, h = Cb1.shape[0], Cb1.shape[1]
            Cb = CbCrFusion(Cb1, Cb2, w, h).reshape([w, h, 1])
            Cr = CbCrFusion(Cr1, Cr2, w, h).reshape([w, h, 1])
            img1_ = img1[:, :, 0] / 255.0
            img2_ = img2[:, :, 0] / 255.0
            # print(Cb1,Cr1,img1_)
            img1_data = transform(img1_)
            img2_data = transform(img2_)
        if f == 3:
            img1 = scipy.misc.imread(vis, mode='YCbCr').astype(np.float32)
            Cb, Cr = img1[:, :, 1], img1[:, :, 2]
            w, h = Cb.shape[0], Cb.shape[1]
            img1_ = img1[:, :, 0] / 255.0
            # print(Cb1,Cr1,img1_)
            Cb = Cb.reshape([w, h, 1])
            Cr = Cr.reshape([w, h, 1])
            img1_data = transform(img1_)

        img1_data = torch.unsqueeze(img1_data, 0).to(device)
        img2_data = torch.unsqueeze(img2_data, 0).to(device)
        print(vis_)
        # print('0000000',img2_data.shape)
        if slis:
            s = 84
            output = img1_data
            h0, w0 = img1_data.shape[2], img1_data.shape[3]
            b_h = h0 // s
            b_w = w0 // s
            for i in range(b_h):
                for j in range(b_w):
                    output[:, :, i * s:min((i + 1) * s, h0 - 1), j * s:min((j + 1) * s, w0 - 1)] = model(
                        img1_data[:, :, i * s:min((i + 1) * s, h0 - 1), j * s:min((j + 1) * s, w0 - 1)],
                        img1_data[:, :, i * s:min((i + 1) * s, h0 - 1), j * s:min((j + 1) * s, w0 - 1)])
        else:
            output, maps = model(img1_data, img2_data)
        torch.cuda.synchronize()

        print(maps[0].shape)  # 1, 32, 34, 62
        print(output.shape)  # 1, 1, 266, 486

        output = np.transpose((torch.squeeze(output, 0).cpu().detach().numpy() * 127.5 + 127.5), (1, 2, 0)).astype(
            np.float32)
        maps_before = np.transpose((torch.squeeze(maps[0], 0).cpu().detach().numpy() * 127.5 + 127.5), (1, 2, 0)).astype(
            np.float32)
        maps_after = np.transpose((torch.squeeze(maps[1], 0).cpu().detach().numpy() * 127.5 + 127.5),
                                   (1, 2, 0)).astype(np.float32)
        if f == 1 or f == 2:
            R, G, B = YCbCr2RGB(output, Cb, Cr)
            output = np.concatenate((B, G, R), 2)

            img1 = cv2.imread(vis).astype(np.float32)
            img2 = cv2.imread(ir).astype(np.float32)
        if f == 3:
            R, G, B = YCbCr2RGB(output, Cb, Cr)
            output = np.concatenate((B, G, R), 2)
        # output = cv2.hconcat([img1, img2, output])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if resi:
            output = cv2.resize(output, (h0, w0))

        # cv2.imwrite(save_path+pre+vis_.split('.')[0]+'.png', output)
        # attention map
        print("MAP: ", maps_after.shape)
        before_path = os.path.join(save_path, 'before/')
        after_path = os.path.join(save_path, 'after/')
        for i in range(maps_after.shape[2]): # maps_after.shape[2]
            map = maps_after[:, :, i]
            cv2.imwrite(after_path + vis_.split('.')[0] + '_' + str(i) + '.png', map)
        for i in range(maps_before.shape[2]): # maps_before.shape[2]
            map = maps_before[:, :, i]
            cv2.imwrite(before_path + vis_.split('.')[0] + '_' + str(i) + '.png', map)
        # cv2.imwrite(save_path+pre+vis_.split('.')[0]+'.1_png', maps[1][0])
        # cv2.imwrite(save_path+pre+vis_.split('.')[0]+'.2_png', maps[2][0])
        end = time.time()
        Time.append(end - start)

    print("Time: mean:%s, std: %s" % (np.mean(Time), np.std(Time)))

if __name__ == '__main__':
	main()