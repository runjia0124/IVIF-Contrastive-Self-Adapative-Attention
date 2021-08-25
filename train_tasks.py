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
# from model_attention import One4AllNet
# from model1 import One4AllNet
from utils import YCbCr2RGB,CbCrFusion
from model import Unet_resize_conv, FusionNet, Unet_res18
# DWA = False
# GradNorm = True

from pytorch_ssim import ssim
from dataset import TrainDataSet as trainset
from dataset import BasicDataset, MaskDataset
from torch.utils.data import DataLoader
from utils import *
from Visualizer import Visualizer
import visdom
from save_image import normimage
from checkpoint import save_epoch, save_latest
from P_loss import ContrastiveLoss, ContrastiveLoss_multi, ContrastiveLoss_Highlevel, \
					ContrastiveLoss_Lowlevel, ContrastiveLoss_multiNegative




def collab_multiNegative(model, data_path, optimizer, epoch, filepath, args):
	# modify: dataset, loss
	data_path = './data.h5'
	# data_path = '../new_ir_vis_data.h5' # for low-high level feature loss

	device = torch.device('cpu')
	if args.use_gpu:
		device = torch.device('cuda')

	model.to(device)

	checkpath = './checkpoints'
	mo = 'w'
	if args.resume:
		mo = 'a'

	f = open(args.logdir + '/log.txt', mo)
	c = [0.75, 0.5, 0.5]
	# c = [0.3, 0.3, 0.3]
	device = torch.device('cuda')
	w = []
	print('total e:', epoch)
	e = 0
	if args.resume:
		logs = torch.load(filepath)
		# model.load_state_dict(logs['model_state_dict'])
		model = torch.load(filepath + '1')
		e = logs['epoch']
	# m = torch.nn.

	T = 1

	visualizer = Visualizer()
	visd = visdom.Visdom()
	ite_num = 0

	model = model.to(device)
	for i in range(e, epoch):
		data_loader = DataLoader(trainset(data_path, arg=args), batch_size=args.bs,
								 shuffle=True, drop_last=True)  # trainset(data_path, arg=args) MaskDataset(data_path[0], data_path[1], data_path[2])
		t = enumerate(iter(data_loader))
		model.train()
		for batch_idx, batch in t:

			ite_num = ite_num + 1
			total_loss = 0
			if args.dwa and (batch_idx > 0 or i > 0):
				pre_losses = losses
			losses = []
			rand = 0
			# x1, x2, mask = batch['vis'], batch['ir'], batch['mask']
			x1 = batch[:, :, :, 0].to(device)
			x2 = batch[:, :, :, 1].to(device)
			mask = batch[:, :, :, 2].to(device)
			mask = mask * 0.5 + 0.5

			n, w, h = x1.shape[0], x1.shape[1], x1.shape[2]
			x1 = x1.view([n, 1, w, h])
			x2 = x2.view([n, 1, w, h])
			mask = mask.view([n, 1, w, h])


			x1 = x1.to(device)
			x2 = x2.to(device)
			mask = mask.to(device)

			vis_1 = x1[0:10, :, :,:]
			vis_2 = x1[10:20, :, :,:]
			vis_3 = x1[20:30, :, :,:]
			ir_1 = x2[0:10, :, :,:]
			ir_2 = x2[10:20, :, :,:]
			ir_3 = x2[20:30, :, :,:]
			mask_1 = mask[0:10, :, :, :]
			if mask_1.max() == 0:
				continue
			w = measure_module1(vis_1, ir_1, args)  # c[args.task]
			y = model(vis_1, ir_1)

			# for contrastive loss
			positive_ir = ir_1 * mask_1
			negative_ir_1 = vis_1 * mask_1
			negative_ir_2 = vis_2 * mask_1
			negative_ir_3 = vis_3 * mask_1

			positive_vis = vis_1 * (1 - mask_1)
			negative_vis_1 = ir_1 * (1 - mask_1)
			negative_vis_2 = ir_2 * (1 - mask_1)
			negative_vis_3 = ir_3 * (1 - mask_1)

			vgg19_weights = '/data/lrj/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'
			# contrastive loss with mask
			c_loss = ContrastiveLoss_multiNegative(vgg19_weights).cuda()
			# contrastive loss with more levels
			# c_loss = ContrastiveLoss_multi(vgg19_weights).cuda()

			contrast_loss = c_loss(negative_ir_1, negative_ir_2, negative_ir_3, y * mask_1, positive_ir) + \
							c_loss(negative_vis_1, negative_vis_2, negative_vis_3, y * (1 - mask_1), positive_vis)

			# contrastive loss using low-high level features
			# lowlevel_loss = ContrastiveLoss_Lowlevel(vgg19_weights).cuda()
			# highlevel_loss = ContrastiveLoss_Highlevel(vgg19_weights).cuda()
			# contrast_loss = lowlevel_loss(x2, y, x1) + highlevel_loss(x1, y, x2)
			loss = args.contrast * contrast_loss + loss_fc(vis_1, ir_1, y, w)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses = collections.OrderedDict()
			losses['loss'] = loss.data.cpu()
			losses['contrast_loss'] = contrast_loss.data.cpu()
			visualizer.plot_current_losses(ite_num + 1,
										   float(i) / len(data_loader),
										   losses)

			visshow = normimage(x1, save_cfg=False)
			irshow = normimage(x2, save_cfg=False)
			outshow = normimage(y, save_cfg=False)
			pos_irshow = normimage(positive_ir, save_cfg=False)
			pos_visshow = normimage(positive_vis, save_cfg=False)

			shows = []
			# shows.append(inputs_show)
			# shows.append(gt_show)
			# shows.append(outputs_show)
			# shows.append(outputs_show1)
			shows.append(visshow)
			shows.append(irshow)
			shows.append(outshow)
			shows.append(pos_irshow)
			shows.append(pos_visshow)
			visd.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))
			# if ite_num % 100 == 0:
			# 	save_latest(netG_2, optimizer_G_2, cfg.work_dir, epoch, ite_num)
			# 	netG_2.train()
			if ite_num % 5 == 0:
				save_latest(model, optimizer, checkpath, e, ite_num)
			# positive_vis = np.transpose((torch.squeeze(positive_vis, 0).cpu().detach().numpy() * 255),
			# 					  (1, 2, 0)).astype(np.float32)
			# cv2.imwrite('./results/' + str(ite_num) + '.png', positive_vis)

			if batch_idx % 100 == 0 and batch_idx != 0:
				print('epoch: {}, batch: {}, total_loss: {}'.format(i, batch_idx, loss))
		# f.write(str(loss.cpu().detach().numpy())+'\n')

	# print(torch.sum(model.weights))
	# model.eval()
	# torch.save(model, filepath+'1')

	# torch.save({
	# 'epoch': i,
	# 'model_state_dict': model.state_dict(),
	# 'optimizer_state_dict': optimizer.state_dict(),
	# 'loss': total_loss
	# }, filepath)

	f.close()

def finetune(model, data_path, optimizer, epoch, filepath, args):
	# modify: dataset, loss
	data_path = './data.h5'
	# data_path = '../new_ir_vis_data.h5' # for low-high level feature loss
	logs = torch.load(filepath)
	device = torch.device('cpu')
	if args.use_gpu:
		device = torch.device('cuda')
	model.load_state_dict(logs['state_dict'])
	optimizer.load_state_dict(logs['optimizer'])
	# model = torch.load(filepath+'1')
	model.to(device)

	checkpath = './checkpoints'
	mo = 'w'
	if args.resume:
		mo = 'a'

	f = open(args.logdir+'/log.txt',mo)
	c = [0.75, 0.5, 0.5]
	# c = [0.3, 0.3, 0.3]
	device = torch.device('cuda')
	w = []
	print('total e:',epoch)
	e = 0
	if args.resume:
		logs = torch.load(filepath)
		# model.load_state_dict(logs['model_state_dict'])
		model = torch.load(filepath+'1')
		e = logs['epoch']
	# m = torch.nn.
	
	T = 1

	visualizer = Visualizer()
	visd = visdom.Visdom()
	ite_num = 0

	model = model.to(device)
	for i in range(e, epoch):
		data_loader = DataLoader(trainset(data_path, arg=args), batch_size = args.bs,
								 shuffle=True) #  trainset(data_path, arg=args) MaskDataset(data_path[0], data_path[1], data_path[2])
		t = enumerate(iter(data_loader))
		model.train()
		for batch_idx, batch in t:

			ite_num = ite_num + 1
			total_loss = 0
			if args.dwa and (batch_idx > 0 or i > 0):
				pre_losses = losses
			losses = []
			rand = 0
			# x1, x2, mask = batch['vis'], batch['ir'], batch['mask']
			x1 = batch[:,:,:,0].to(device)
			x2 = batch[:,:,:,1].to(device)
			mask = batch[:,:,:,2].to(device)
			mask = mask*0.5 + 0.5



			n,w,h = x1.shape[0], x1.shape[1], x1.shape[2]
			x1 = x1.view([n,1,w,h])
			x2 = x2.view([n,1,w,h])
			mask = mask.view([n,1,w,h])
			if mask.max() == 0:
				continue

			x1 = x1.to(device)
			x2 = x2.to(device)
			mask = mask.to(device)

			w = measure_module1(x1, x2, args)   # c[args.task]
			y = model(x1,x2)

			# for contrastive loss
			positive_ir = x2 * mask
			negative_ir = x1 * mask
			positive_vis = x1 * (1 - mask)
			negative_vis = x2 * (1 - mask)

			vgg19_weights = '/data/lrj/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'
			# contrastive loss with mask
			c_loss = ContrastiveLoss(vgg19_weights).cuda()
			# contrastive loss with more levels
			# c_loss = ContrastiveLoss_multi(vgg19_weights).cuda()

			contrast_loss = c_loss(negative_ir, y*mask, positive_ir) + \
							 c_loss(negative_vis, y*(1-mask), positive_vis)

			# contrastive loss using low-high level features
			# lowlevel_loss = ContrastiveLoss_Lowlevel(vgg19_weights).cuda()
			# highlevel_loss = ContrastiveLoss_Highlevel(vgg19_weights).cuda()
			# contrast_loss = lowlevel_loss(x2, y, x1) + highlevel_loss(x1, y, x2)
			loss = loss_fc(x1, x2, y, w) + args.contrast * contrast_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses = collections.OrderedDict()
			losses['loss'] = loss.data.cpu()
			losses['contrast_loss'] = contrast_loss.data.cpu()
			visualizer.plot_current_losses(ite_num + 1,
										   float(i) / len(data_loader),
										   losses)


			visshow = normimage(x1, save_cfg=False)
			irshow = normimage(x2, save_cfg=False)
			outshow = normimage(y, save_cfg=False)
			pos_irshow = normimage(positive_ir, save_cfg=False)
			pos_visshow = normimage(positive_vis, save_cfg=False)

			shows = []
			# shows.append(inputs_show)
			# shows.append(gt_show)
			# shows.append(outputs_show)
			# shows.append(outputs_show1)
			shows.append(visshow)
			shows.append(irshow)
			shows.append(outshow)
			shows.append(pos_irshow)
			shows.append(pos_visshow)
			visd.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))
			# if ite_num % 100 == 0:
			# 	save_latest(netG_2, optimizer_G_2, cfg.work_dir, epoch, ite_num)
			# 	netG_2.train()
			if ite_num % 5 == 0:
				save_latest(model, optimizer, checkpath, e, ite_num)
				# positive_vis = np.transpose((torch.squeeze(positive_vis, 0).cpu().detach().numpy() * 255),
				# 					  (1, 2, 0)).astype(np.float32)
				# cv2.imwrite('./results/' + str(ite_num) + '.png', positive_vis)


			if batch_idx % 100 == 0 and batch_idx!=0:
				print('epoch: {}, batch: {}, total_loss: {}'.format(i,batch_idx,loss))
				# f.write(str(loss.cpu().detach().numpy())+'\n')
		
		# print(torch.sum(model.weights))
		# model.eval()
		# torch.save(model, filepath+'1')

		# torch.save({
		# 'epoch': i,
		# 'model_state_dict': model.state_dict(),
		# 'optimizer_state_dict': optimizer.state_dict(),
		# 'loss': total_loss
		# }, filepath)
		
	f.close()


def finetune_multiNegative(model, data_path, optimizer, epoch, filepath, args):
	# modify: dataset, loss
	data_path = './data.h5'
	# data_path = '../new_ir_vis_data.h5' # for low-high level feature loss
	# logs = torch.load(filepath)
	filepath = './logs/latest.pth'
	"""
	if args.save_dir == 1:
		filepath = './logs_1/latest.pth'
	elif args.save_dir == 2:
		filepath = './logs_2/latest.pth'
	elif args.save_dir == 3:
		filepath = './logs_3/latest.pth'
	elif args.save_dir == 4:
		filepath = './logs_4/latest.pth'
	elif args.save_dir == 5:
		filepath = './logs_5/latest.pth'
	elif args.save_dir == 6:
		filepath = './logs_6/latest.pth'
	"""
	logs = torch.load(filepath)

	device = torch.device('cpu')
	if args.use_gpu:
		device = torch.device('cuda')
	model.load_state_dict(logs['state_dict'])
	optimizer.load_state_dict(logs['optimizer'])
	# model = torch.load(filepath+'1')
	model.to(device)

	checkpath = './checkpoints'
	"""
	if args.save_dir == 1:
		checkpath = './checkpoints_1'
	elif args.save_dir == 2:
		checkpath = './checkpoints_2'
	elif args.save_dir == 3:
		checkpath = './checkpoints_3'
	elif args.save_dir == 4:
		checkpath = './checkpoints_4'
	elif args.save_dir == 5:
		checkpath = './checkpoints_5'
	elif args.save_dir == 6:
		checkpath = './checkpoints_6'
	"""
	mo = 'w'
	if args.resume:
		mo = 'a'

	f = open(args.logdir + '/log.txt', mo)
	c = [0.75, 0.5, 0.5]
	# c = [0.3, 0.3, 0.3]
	device = torch.device('cuda')
	w = []
	print('total e:', epoch)
	e = 0
	if args.resume:
		logs = torch.load(filepath)
		# model.load_state_dict(logs['model_state_dict'])
		model = torch.load(filepath + '1')
		e = logs['epoch']
	# m = torch.nn.

	T = 1

	visualizer = Visualizer()
	visd = visdom.Visdom()
	ite_num = 0

	model = model.to(device)
	for i in range(e, epoch):
		data_loader = DataLoader(trainset(data_path, arg=args), batch_size=args.bs,
								 shuffle=True, drop_last=True)  # trainset(data_path, arg=args) MaskDataset(data_path[0], data_path[1], data_path[2])
		t = enumerate(iter(data_loader))
		model.train()
		for batch_idx, batch in t:

			ite_num = ite_num + 1
			total_loss = 0
			if args.dwa and (batch_idx > 0 or i > 0):
				pre_losses = losses
			losses = []
			rand = 0
			# x1, x2, mask = batch['vis'], batch['ir'], batch['mask']
			x1 = batch[:, :, :, 0].to(device)
			x2 = batch[:, :, :, 1].to(device)
			mask = batch[:, :, :, 2].to(device)
			mask = mask * 0.5 + 0.5

			n, w, h = x1.shape[0], x1.shape[1], x1.shape[2]
			x1 = x1.view([n, 1, w, h])
			x2 = x2.view([n, 1, w, h])
			mask = mask.view([n, 1, w, h])


			x1 = x1.to(device)
			x2 = x2.to(device)
			mask = mask.to(device)

			vis_1 = x1[0:10, :, :,:]
			vis_2 = x1[10:20, :, :,:]
			vis_3 = x1[20:30, :, :,:]
			ir_1 = x2[0:10, :, :,:]
			ir_2 = x2[10:20, :, :,:]
			ir_3 = x2[20:30, :, :,:]
			mask_1 = mask[0:10, :, :, :]
			if mask_1.max() == 0:
				continue
			w = measure_module1(vis_1, ir_1, args)  # c[args.task]
			y = model(vis_1, ir_1)

			# for contrastive loss
			positive_ir = ir_1 * mask_1
			negative_ir_1 = vis_1 * mask_1
			negative_ir_2 = vis_2 * mask_1
			negative_ir_3 = vis_3 * mask_1
			# negative_ir_1 = ir_1 * mask_1
			# negative_ir_2 = ir_2 * mask_1
			# negative_ir_3 = ir_3 * mask_1

			positive_vis = vis_1 * (1 - mask_1)
			negative_vis_1 = ir_1 * (1 - mask_1)
			negative_vis_2 = ir_2 * (1 - mask_1)
			negative_vis_3 = ir_3 * (1 - mask_1)
			# negative_vis_1 = vis_1 * (1 - mask_1)
			# negative_vis_2 = vis_2 * (1 - mask_1)
			# negative_vis_3 = vis_3 * (1 - mask_1)

			vgg19_weights = '/data/lrj/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'
			# contrastive loss with mask
			c_loss = ContrastiveLoss_multiNegative(vgg19_weights).cuda()
			# contrastive loss with more levels
			# c_loss = ContrastiveLoss_multi(vgg19_weights).cuda()

			contrast_loss = c_loss(negative_ir_1, negative_ir_2, negative_ir_3, y * mask_1, positive_ir) + \
							c_loss(negative_vis_1, negative_vis_2, negative_vis_3, y * (1 - mask_1), positive_vis)
			# contrast_loss = c_loss(negative_ir_1, negative_ir_2, negative_ir_3, y * mask_1, positive_ir)

			# contrastive loss using low-high level features
			# lowlevel_loss = ContrastiveLoss_Lowlevel(vgg19_weights).cuda()
			# highlevel_loss = ContrastiveLoss_Highlevel(vgg19_weights).cuda()
			# contrast_loss = lowlevel_loss(x2, y, x1) + highlevel_loss(x1, y, x2)
			loss = args.contrast * contrast_loss + loss_fc(args, vis_1, ir_1, y, w)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses = collections.OrderedDict()
			losses['loss'] = loss.data.cpu()
			losses['contrast_loss'] = contrast_loss.data.cpu()
			visualizer.plot_current_losses(ite_num + 1,
										   float(i) / len(data_loader),
										   losses)

			visshow = normimage(x1, save_cfg=False)
			irshow = normimage(x2, save_cfg=False)
			outshow = normimage(y, save_cfg=False)
			pos_irshow = normimage(positive_ir, save_cfg=False)
			pos_visshow = normimage(positive_vis, save_cfg=False)

			shows = []
			# shows.append(inputs_show)
			# shows.append(gt_show)
			# shows.append(outputs_show)
			# shows.append(outputs_show1)
			shows.append(visshow)
			shows.append(irshow)
			shows.append(outshow)
			shows.append(pos_irshow)
			shows.append(pos_visshow)
			visd.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))
			# if ite_num % 100 == 0:
			# 	save_latest(netG_2, optimizer_G_2, cfg.work_dir, epoch, ite_num)
			# 	netG_2.train()
			if ite_num % 5 == 0:
				save_latest(model, optimizer, checkpath, e, ite_num)
			# positive_vis = np.transpose((torch.squeeze(positive_vis, 0).cpu().detach().numpy() * 255),
			# 					  (1, 2, 0)).astype(np.float32)
			# cv2.imwrite('./results/' + str(ite_num) + '.png', positive_vis)

			if batch_idx % 100 == 0 and batch_idx != 0:
				print('epoch: {}, batch: {}, total_loss: {}'.format(i, batch_idx, loss))
		# f.write(str(loss.cpu().detach().numpy())+'\n')

	# print(torch.sum(model.weights))
	# model.eval()
	# torch.save(model, filepath+'1')

	# torch.save({
	# 'epoch': i,
	# 'model_state_dict': model.state_dict(),
	# 'optimizer_state_dict': optimizer.state_dict(),
	# 'loss': total_loss
	# }, filepath)

	f.close()


def train(model, data_path, optimizer, epoch, filepath, args):
	checkpath = './logs'
	"""
	if args.save_dir == 1:
		checkpath = './logs_1'
	elif args.save_dir == 2:
		checkpath = './logs_2'
	elif args.save_dir == 3:
		checkpath = './logs_3'
	elif args.save_dir == 4:
		checkpath = './logs_4'
	elif args.save_dir == 5:
		checkpath = './logs_5'
	elif args.save_dir == 6:
		checkpath = './logs_6'
	"""
	mo = 'w'
	if args.resume:
		mo = 'a'

	f = open(args.logdir + '/log.txt', mo)
	c = [0.75, 0.5, 0.5]
	# c = [0.3, 0.3, 0.3]
	device = torch.device('cuda')
	w = []
	print('total e:', epoch)
	e = 0
	if args.resume:
		logs = torch.load(filepath)
		# model.load_state_dict(logs['model_state_dict'])
		model = torch.load(filepath + '1')
		e = logs['epoch']
	# m = torch.nn.

	T = 1

	visualizer = Visualizer()
	visd = visdom.Visdom()
	ite_num = 0

	model = model.to(device)
	for i in range(e, epoch):
		data_loader = DataLoader(trainset(data_path, arg=args),
								 batch_size=args.bs, shuffle=True) #  trainset(data_path, arg=args) MaskDataset(data_path[0], data_path[1], data_path[2])
		t = enumerate(iter(data_loader))
		model.train()
		for batch_idx, batch in t:

			ite_num = ite_num + 1
			total_loss = 0
			if args.dwa and (batch_idx > 0 or i > 0):
				pre_losses = losses
			losses = []
			rand = 0
			# x1, x2 = batch['image'], batch['mask']
			x1 = batch[:,:,:,rand].to(device)
			x2 = batch[:,:,:,1-rand].to(device)
			# print(x1.shape)
			n, w, h = x1.shape[0], x1.shape[1], x1.shape[2]
			x1 = x1.view([n, 1, w, h])
			x2 = x2.view([n, 1, w, h])

			x1 = x1.to(device)
			x2 = x2.to(device)

			w = measure_module1(x1, x2, args)  # c[args.task]
			y = model(x1, x2)

			vgg19_weights = '/data/lrj/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'
			# contrastive loss with mask
			# c_loss = ContrastiveLoss(vgg19_weights).cuda()
			# contrast_loss = c_loss(negative_ir, y*mask, positive_ir) + \
			# 				 c_loss(negative_vis, y*(1-mask), positive_vis)

			# contrastive loss using low-high level features
			# lowlevel_loss = ContrastiveLoss_Lowlevel(vgg19_weights).cuda()
			# highlevel_loss = ContrastiveLoss_Highlevel(vgg19_weights).cuda()
			# contrast_loss = lowlevel_loss(x2, y, x1) + highlevel_loss(x1, y, x2)

			loss = loss_fc(args, x1, x2, y, w)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses = collections.OrderedDict()
			losses['loss'] = loss.data.cpu()
			losses[' '] = 0
			visualizer.plot_current_losses(ite_num + 1,
										   float(i) / len(data_loader),
										   losses)

			if ite_num % 5 == 0:
				visshow = normimage(x1, save_cfg=False)
				irshow = normimage(x2, save_cfg=False)
				outshow = normimage(y, save_cfg=False)

				shows = []
				# shows.append(inputs_show)
				# shows.append(gt_show)
				# shows.append(outputs_show)
				# shows.append(outputs_show1)
				shows.append(visshow)
				shows.append(irshow)
				shows.append(outshow)
				visd.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))
				# if ite_num % 100 == 0:
				# 	save_latest(netG_2, optimizer_G_2, cfg.work_dir, epoch, ite_num)
				# 	netG_2.train()
				save_latest(model, optimizer, checkpath, e, ite_num)

			if batch_idx % 100 == 0 and batch_idx != 0:
				print('epoch: {}, batch: {}, total_loss: {}'.format(i, batch_idx, loss))
		# f.write(str(loss.cpu().detach().numpy())+'\n')

	# print(torch.sum(model.weights))
	# model.eval()
	# torch.save(model, filepath+'1')

	# torch.save({
	# 'epoch': i,
	# 'model_state_dict': model.state_dict(),
	# 'optimizer_state_dict': optimizer.state_dict(),
	# 'loss': total_loss
	# }, filepath)

	f.close()


def train_noCrop(model, data_path, optimizer, epoch, filepath, args):
	checkpath = './logs'
	mo = 'w'
	if args.resume:
		mo = 'a'

	f = open(args.logdir + '/log.txt', mo)
	c = [0.75, 0.5, 0.5]
	# c = [0.3, 0.3, 0.3]
	device = torch.device('cuda')
	w = []
	print('total e:', epoch)
	e = 0
	if args.resume:
		logs = torch.load(filepath)
		# model.load_state_dict(logs['model_state_dict'])
		model = torch.load(filepath + '1')
		e = logs['epoch']
	# m = torch.nn.

	T = 1

	visualizer = Visualizer()
	visd = visdom.Visdom()
	ite_num = 0

	model = model.to(device)
	for i in range(e, epoch):
		data_loader = DataLoader(BasicDataset(data_path[0], data_path[1]),
								 batch_size=args.bs, shuffle=True) #  trainset(data_path, arg=args) MaskDataset(data_path[0], data_path[1], data_path[2])
		t = enumerate(iter(data_loader))
		model.train()
		for batch_idx, batch in t:

			ite_num = ite_num + 1
			total_loss = 0
			if args.dwa and (batch_idx > 0 or i > 0):
				pre_losses = losses
			losses = []
			rand = 0
			x1, x2 = batch['image'], batch['mask']
			# x1 = batch[:,:,:,rand].to(device)
			# x2 = batch[:,:,:,1-rand].to(device)
			# print(x1.shape)
			# n, w, h = x1.shape[0], x1.shape[1], x1.shape[2]
			# x1 = x1.view([n, 1, w, h])
			# x2 = x2.view([n, 1, w, h])

			x1 = x1.to(device)
			x2 = x2.to(device)

			w = measure_module1(x1, x2, args)  # c[args.task]
			y = model(x1, x2)

			vgg19_weights = '/data/lrj/.cache/torch/hub/checkpoints/vgg19-dcbb9e9d.pth'
			# contrastive loss with mask
			# c_loss = ContrastiveLoss(vgg19_weights).cuda()
			# contrast_loss = c_loss(negative_ir, y*mask, positive_ir) + \
			# 				 c_loss(negative_vis, y*(1-mask), positive_vis)

			# contrastive loss using low-high level features
			lowlevel_loss = ContrastiveLoss_Lowlevel(vgg19_weights).cuda()
			highlevel_loss = ContrastiveLoss_Highlevel(vgg19_weights).cuda()
			contrast_loss = lowlevel_loss(x2, y, x1) + highlevel_loss(x1, y, x2)
			loss = loss_fc(x1, x2, y, w) + args.contrast * contrast_loss

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			losses = collections.OrderedDict()
			losses['loss'] = loss.data.cpu()
			losses[' '] = 0
			visualizer.plot_current_losses(ite_num + 1,
										   float(i) / len(data_loader),
										   losses)

			if ite_num % 5 == 0:
				visshow = normimage(x1, save_cfg=False)
				irshow = normimage(x2, save_cfg=False)
				outshow = normimage(y, save_cfg=False)

				shows = []
				# shows.append(inputs_show)
				# shows.append(gt_show)
				# shows.append(outputs_show)
				# shows.append(outputs_show1)
				shows.append(visshow)
				shows.append(irshow)
				shows.append(outshow)
				visd.images(shows, nrow=4, padding=3, win=1, opts=dict(title='Output images'))
				# if ite_num % 100 == 0:
				# 	save_latest(netG_2, optimizer_G_2, cfg.work_dir, epoch, ite_num)
				# 	netG_2.train()
				save_latest(model, optimizer, checkpath, e, ite_num)

			if batch_idx % 100 == 0 and batch_idx != 0:
				print('epoch: {}, batch: {}, total_loss: {}'.format(i, batch_idx, loss))
		# f.write(str(loss.cpu().detach().numpy())+'\n')

	# print(torch.sum(model.weights))
	# model.eval()
	# torch.save(model, filepath+'1')

	# torch.save({
	# 'epoch': i,
	# 'model_state_dict': model.state_dict(),
	# 'optimizer_state_dict': optimizer.state_dict(),
	# 'loss': total_loss
	# }, filepath)

	f.close()




def softmax(x, T):
	x = [np.exp(i.cpu().detach()/T) for i in x]
	sum_ = sum(x)
	x = [len(x)*i/sum_ for i in x]
	return	x

def measure_module1(x1, x2, args):
	c1 = 0.5
	c2 = 0.2
	c1 = args.c1
	c2 = args.c2
	gm1,m1 = measure_info(x1)
	gm2,m2 = measure_info(x2)
	# print(gm1.shape,m1.shape)
	output = []
	with torch.no_grad():
		m = nn.Softmax(dim=0)
		sum_ = gm1 + gm2
		e = torch.stack((gm1/c1/sum_,gm2/c1/sum_),0)
		output.append(m(e))
		# a = m(e)
		# print(a[0,0,0,0,0],a[1,0,0,0,0])
		sum_ = m1 + m2
		e = torch.stack((m1/c2/sum_,m2/c2/sum_),0)
		# print(m(e))
		output.append(m(e))
	return output

def measure_module(gm1, gm2, m1, m2, c):
	output = []
	# with torch.no_grad():
	m = nn.Softmax(dim=0)
	sum_ = gm1 + gm2
	e = torch.stack((gm1/c/sum_,gm2/c/sum_),0)
	output.append(m(e))
	
	sum_ = m1 + m2
	e = torch.stack((m1/c/sum_,m2/c/sum_),0)
	# print(m(e))
	output.append(m(e))
	return output

def measure_info(x):
	with torch.no_grad():
		grad_model = Gradient_Net_iqa().to(x.device)
		grad = gradient(x, grad_model)

		grad_mean = grad.mean(dim=(1,2,3))
		# x = x + 1
		# mean = x.mean()
		en = entropy(x)
		en = torch.from_numpy(en).cuda()
		return grad_mean, en

def entropy(x):
	len = x.shape[0]
	entropies = np.zeros(shape = (len))
	grey_level = 256
	counter = np.zeros(shape = (grey_level, 1))

	for i in range(len):
		input_uint8 = (x[i, 0, :, :] * 127.5+127.5).cpu().detach().numpy().astype(np.uint8)
		# input_uint8 = input_uint8 + 1
		W = x.shape[2]
		H = x.shape[3]
		for m in range(W):
			for n in range(H):
				indexx = input_uint8[m, n]
				counter[indexx] = counter[indexx] + 1
		total = np.sum(counter)
		p = counter / total
		for k in range(grey_level):
			if p[k] != 0:
				entropies[i] = entropies[i] - p[k] * np.log2(p[k])
				
		entropies += 0.0001
	return entropies

def loss_fc(args, x1,x2,y,w=[[0.5,0.5],[0.5,0.5]]):
	L1loss = nn.L1Loss()
	MSEloss = nn.MSELoss()
	smoothloss = nn.SmoothL1Loss()

	len_ = x1.shape[0]


	mse_loss = []
	# print(w[0][0][0])
	for i in range(len_):
		loss = w[1][0][i]*MSEloss(y[i], x1[i]) + w[1][1][i]*MSEloss(y[i], x2[i])
		# loss = args.w_loss * MSEloss(y[i], x1[i]) + (1-args.w_loss) * MSEloss(y[i], x2[i])
		mse_loss.append(loss)
	mse_loss = torch.stack(mse_loss)
	mse_loss = torch.mean(mse_loss)

	"""
	l1_loss = []
	# print(w[0][0][0])
	for i in range(len_):
		loss = w[1][0][i] * L1loss(y[i], x1[i]) + w[1][1][i] * L1loss(y[i], x2[i])
		l1_loss.append(loss)
	# l1_loss = w[1][0]*MSEloss(y, x1) + w[1][1]*MSEloss(y, x2)
	l1_loss = torch.stack(l1_loss)
	l1_loss = torch.mean(l1_loss)
	"""
	x1 = x1 * 0.5 + 0.5
	x2 = x2 * 0.5 + 0.5
	y = y * 0.5 + 0.5
	# ssim_ = []
	# for i in range(len_):
	# 	loss = w[1][0][i]*(1 - ssim(x1,y,data_range=1,size_average=False))+\
	# 			w[1][1][i]*(1 - ssim(x2,y,data_range=1,size_average=False))
	# 	ssim_.append(loss)
	# ssim_loss = torch.stack(ssim_)
	# ssim_loss = torch.mean(ssim_loss)
	# print(w)
	# print(w[0][0].shape, ssim(x1,y,data_range=1,size_average=False).shape)
	ssim_1 = ssim(x1,y)
	ssim_2 = ssim(x2,y)

	# ssim_1 = torch.reshape(ssim_1, [ssim_1.shape[0], 1,1,1])
	# ssim_2 = torch.reshape(ssim_2, [ssim_2.shape[0], 1,1,1])

	ssim_loss = w[0][0]*(1 - ssim_1)+\
				w[0][1]*(1 - ssim_2)
	# ssim_loss = args.w_loss * (1 - ssim_1) + \
	# 			(1-args.w_loss) * (1 - ssim_2)
	ssim_loss = torch.mean(ssim_loss)

	return  20*ssim_loss + mse_loss

