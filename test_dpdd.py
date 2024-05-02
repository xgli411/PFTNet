import os
import argparse
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data

from PFTNet import DeepRFT as mynet

from skimage import img_as_ubyte
from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as simm_loss
from sklearn.metrics import mean_absolute_error
import cv2

#################3

from util import *
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx
#################33

from sklearn.metrics import mean_absolute_error
from skimage.metrics import structural_similarity
from utils_dpdd import *
from data_loader.utils import load_file_list, read_frame, refine_image
import lpips
alex = lpips.LPIPS(net='alex').cuda()

parser = argparse.ArgumentParser(description='Image Deblurring')
parser.add_argument('--input_dir', default='./Datasets/blur/', type=str, help='Directory of validation images')
#parser.add_argument('--input_dir', default='./Datasets/RealBlur/RealBlur_J/test/blur', type=str, help='Directory of validation images')
parser.add_argument('--target_dir', default='./Datasets/sharp/', type=str, help='Directory of validation images')

parser.add_argument('--output_dir', default='./results/DPDD(YUAN)/', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='./checkpoints/DeepRFT/model_DPDD.pth', type=str, help='Path to weights')

parser.add_argument('--get_psnr', default=True, type=bool, help='PSNR')
parser.add_argument('--gpus', default='0,1,2', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--save_result', default=True, type=bool, help='save result')
parser.add_argument('--win_size', default=512, type=int, help='window size, [GoPro, HIDE, RealBlur]=256, [DPDD]=512')
args = parser.parse_args()
result_dir = args.output_dir
win = args.win_size
get_psnr = args.get_psnr
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
# model_restoration = mynet()
model_restoration = mynet(inference=True)
# print number of model
get_parameter_number(model_restoration)
# utils.load_checkpoint(model_restoration, args.weights)
utils.load_checkpoint_compress_doconv(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# dataset = args.dataset
rgb_dir_test = args.input_dir
test_dataset = get_test_data(rgb_dir_test, img_options={})
test_loader  = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=8, drop_last=False, pin_memory=True)

def mae(img1, img2):
    mae_0=mean_absolute_error(img1[:,:,0], img2[:,:,0],
                              multioutput='uniform_average')
    mae_1=mean_absolute_error(img1[:,:,1], img2[:,:,1],
                              multioutput='uniform_average')
    mae_2=mean_absolute_error(img1[:,:,2], img2[:,:,2],
                              multioutput='uniform_average')
    return np.mean([mae_0,mae_1,mae_2])


def psnr(img1, img2, PIXEL_MAX = 1.0):
    mse_ = np.mean( (img1 - img2) ** 2 )
    return 10 * math.log10(PIXEL_MAX / mse_)

    ##
time_norm = 0
total_itr_time = 0

PSNR_dpdd = 0
SSIM  = 0
MAE = 0
LPIPs = 0
PSNR_mean = 0.
SSIM_mean = 0.
MAE_mean = 0.
LPIPS_mean = 0.
#LPIPSN = LPIPS.PerceptualLoss(model='net-lin',net='alex', use_gpu=args.gpus)
##


psnr_val_rgb = []
simm_val_rgb = []
MAE_val_rgb = []
alex_val_rgb = []

psnr = 0

utils.mkdir(result_dir)

with torch.no_grad():
    psnr_list = []
    ssim_list = []
    alex_list = []
    #MAE_list = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):

        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()
        input_    = data_test[0].cuda()
        filenames = data_test[1]
        _, _, Hx, Wx = input_.shape
        filenames = data_test[1]
        input_re, batch_list = window_partitionx(input_, win)

        restored = model_restoration(input_re)
        restored = window_reversex(restored, win, Hx, Wx, batch_list)

        restored = torch.clamp(restored, 0, 1)
        restored1 = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        #LPIPS_mean += 1
        #print("LPIPS_mean",LPIPS_mean)

        
        for batch in range(len(restored1)):
                restored_img = restored1[batch]
                restored_img = img_as_ubyte(restored1[batch])
                    #print("restored_img",len(restored_img))
                if get_psnr:
                   
                        
        
                    rgb_gt = cv2.imread(os.path.join(args.target_dir, filenames[batch]+'.png'))
                    rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)
                    rgb_gt = np.float32(load_img(rgb_gt))/255.
                        
                    patchC = torch.from_numpy(rgb_gt).unsqueeze(0).permute(0,3,1,2).cuda()
    
                    psnr_val_rgb.append(psnr_loss(restored_img, rgb_gt))
                    simm_val_rgb.append(simm_loss(restored_img, rgb_gt,multichannel=True))
                    MAE_val_rgb.append(mae(restored_img, rgb_gt))
                    alex_val_rgb.append(alex(patchC, restored,normalize=True).item())
    #                
                        #MAE = mae(restored_img, rgb_gt)
                    # print("psnr_val_rgb",psnr_val_rgb)
                    # print("simm_val_rgb",simm_val_rgb)
                    # print("MAE_val_rgb",MAE_val_rgb)
                    # print("alex_val_rgb",alex_val_rgb)
                    #     

                if args.save_result:
                    utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)
    
               # PSNR_mean = PSNR_mean / len(test_dataset)


if get_psnr:
    psnr = sum(psnr_val_rgb) / len(test_dataset)
    ssim = sum(simm_val_rgb) / len(test_dataset)
    mae = sum(MAE_val_rgb) / len(MAE_val_rgb)
    alex = sum(alex_val_rgb) / len(alex_val_rgb)
    print("PSNR: %f" % psnr)
    print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(psnr, ssim, mae, alex))

#    print("Overall: PSNR :%f SSIM: %f MAE: %f} LPIPS: f}"%PSNR_mean %SSIM_mean %MAE_mean %LPIPS_mean)

