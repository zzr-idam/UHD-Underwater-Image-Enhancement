import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
# import network
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from torchvision.utils import save_image
import network, network_wochannel, network_wo1d, network_doublegrid, network_doubleNAF, network_kong, network_pin
import datetime
import cv2
import os
import math
from torchvision.transforms import InterpolationMode


import torch
import torch.optim
from torchvision import transforms
from PIL import Image
import torch 
from torchvision.utils import save_image
import network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
my_model = network.Pyramid().to(device)
my_model.eval()
my_model.to(device)
my_model.load_state_dict(torch.load("model/test_our_e_440.pth", map_location=torch.device('cpu'))) 
to_pil_image = transforms.ToPILImage()
tfs_full = transforms.Compose([
            transforms.ToTensor()
        ])
i = 0
for idx in range(1):
     image_in = Image.open('input_2.jpg').convert('RGB')
     full = tfs_full(image_in).unsqueeze(0).to(device)
     output = my_model(full)
     save_image(output[0], '{}.jpg'.format('2_test'))
'''
def psnr1(img1, img2):
    mse = torch.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def psnr2(img1, img2):
    mse = torch.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
from skimage.metrics import structural_similarity
def ssim(img1,img2):
    return structural_similarity(np.squeeze(img1.cpu().detach().numpy().transpose(0,2,3,1), 0), np.squeeze(img2.cpu().detach().numpy().transpose(0,2,3,1), 0), multichannel=True)
# 指定使用0,1,2三块卡
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_model = network.Pyramid()
my_model.cuda()
my_model.eval()
my_model.load_state_dict(torch.load("model/our_deblur_430.pth"))  #905
# GAN.load_state_dict(torch.load("/home/dell/IJCAI/JBL/JBPSC/model/model_g_epoch69.pth"))
to_pil_image = transforms.ToPILImage()

tfs_full = transforms.Compose([
    #transforms.Resize((2000,2000),InterpolationMode.BICUBIC),
    transforms.ToTensor()
])
tfs_full1 = transforms.Compose([
    #transforms.Resize((2000,2000),Image.BICUBIC),
    transforms.ToTensor()
])
tfs_full2 = transforms.Compose([
    #transforms.Resize((2160,3840),Image.BICUBIC),
])

def load_simple_list(src_path):
    name_list = list()
    for name in os.listdir(src_path):
        path = os.path.join(src_path, name)
        name_list.append(path)
    name_list = [name for name in name_list if '.jpg' in name]
    name_list.sort()
    return name_list
list_t = '/home/wenwen/dataset/underwater_dataset/train/input/'
list_g= '/home/wenwen/dataset/underwater_dataset/train/gt/'
#list_t = '/home/wenwen/dataset/underwater_dataset/train/input/'
#list_g= '/home/wenwen/dataset/underwater_dataset/train/gt/'
#list_t = '/home/wenwen/dataset/underwater/underwater/'
#list_g = '/home/wenwen/dataset/underwater/gt/'
#list_s = '/home/wenwen/dataset/IO-HAZE/testi_haze/'
'''
'''
name = '87.jpg'

#image_haze = Image.open('/home/wenwen/dataset/underwater/underwater/' + name).convert('RGB')
#image_gt = Image.open('/home/wenwen/dataset/underwater/gt/' + name).convert('RGB')
image_haze = Image.open('/home/wenwen/dataset/underwater_dataset/cycle_test/input/' + name).convert('RGB')
image_gt = Image.open('/home/wenwen/dataset/underwater_dataset/cycle_test/input/' + name).convert('RGB')
full_haze = tfs_full(image_haze).unsqueeze(0).cuda()
full_gt = tfs_full(image_gt).unsqueeze(0).cuda()
x_f = torch.fft.fft2(full_haze)
x_f = torch.fft.fftshift(x_f)
temp = transforms.GaussianBlur(kernel_size=501)(full_haze)
real = x_f.real
imag = x_f.imag
save_image(torch.log(torch.abs(imag[:,1,:,:])), 'test_result/imag.jpg', normalize=True)
save_image(torch.log(torch.abs(real[:,0,:,:])), 'test_result/real.jpg', normalize=True)
save_image(temp, 'test_result/gs.jpg', normalize=True)
for i in range(5):
  start = time.time()
  output = my_model(full_haze)
  end = time.time()
  print("TIME:",end - start)
  print("PSNR:",psnr1(full_gt*255,output*255))
  s = ssim(full_gt, output)
  print("SSIM:",s)

  save_image(output, 'test_result/' + name)
  '''
'''
def test_model(list_haze, list_gt, my_model):
    PSNR = 0
    i = 0
    times = 0
    SSIM = 0
    for name in os.listdir(list_haze):

        image_haze = Image.open(list_haze + '/' + name).convert('RGB')
        image_gt = Image.open(list_gt + '/' + name).convert('RGB')
        #image_haze = Image.open('vedio//' + name).convert('RGB')
        #image_gt = Image.open('vedio//' + name).convert('RGB')
        # content = cv2.imread('/6T/home/dell/PyTorch-Image-Dehazing-master/O-HAZY/hazy/35_outdoor_hazy.jpg')

        # content = content.transpose((2, 0, 1))/255.0
        # content = torch.tensor(content).unsqueeze(0).float().cuda()

        full_haze = tfs_full(image_haze).unsqueeze(0).cuda()
        full_gt = tfs_full1(image_gt).unsqueeze(0).cuda()
        # hg, wg = torch.meshgrid([torch.arange(0, full.shape[2]), torch.arange(0, full.shape[3])]) # [0,511] HxW

        # hg = hg.to(device)
        # wg = wg.to(device)

        start = time.time()
        output = my_model(full_haze)
        end = time.time()
        t = end - start
        print(end - start)
        if i>0:
            times += t
        #output = tfs_full2(full_haze)
        output = tfs_full2(output)
        p = psnr1(full_gt*255,output*255)
        PSNR += p
        s = ssim(full_gt, output)
        SSIM += s
        #print('TIME:',t)
        print('PSNR:',p)
        print('SSIM:',s)
        save_image(output, 'test_result/' + name)
        #save_image(output, 'test_real/' + name)
        i += 1
    print(i)
    print('PSNR_mean:', PSNR/i)
    print('SSIM_mean:', SSIM/i)
    print('Time_mean:', times/(i-1))
    return PSNR/i
test_model(list_t, list_g, my_model)
'''