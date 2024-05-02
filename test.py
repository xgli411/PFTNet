import os
import argparse
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data

from PFTNet import DeepRFT as mynet

from skimage import img_as_ubyte
from get_parameter_number import get_parameter_number
from tqdm import tqdm
from layers import *
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
import cv2


parser = argparse.ArgumentParser(description='Image Deblurring')
parser.add_argument('--input_dir', default='./Datasets/blur/', type=str, help='Directory of validation images')
parser.add_argument('--target_dir', default='./Datasets/sharp/', type=str, help='Directory of validation images')
parser.add_argument('--output_dir', default='./results/', type=str, help='Directory of validation images')
parser.add_argument('--weights', default='./model_DPDD.pth', type=str, help='Path to weights')
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
psnr_val_rgb = []
psnr = 0

utils.mkdir(result_dir)

with torch.no_grad():
    psnr_list = []
    ssim_list = []
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
        restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()
        for batch in range(len(restored)):
            restored_img = restored[batch]
            restored_img = img_as_ubyte(restored[batch])
            if get_psnr:
               # with open(001.png,'rb') as f:
               #      im_read = f.read()
               #  data = np.frombuffer(im_read,dtype='uint8')
               #  rgb_gt = cv2.imdecode(data,1)
                rgb_gt = cv2.imread(os.path.join(args.target_dir, filenames[batch]+'.png'))
                rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)
                psnr_val_rgb.append(psnr_loss(restored_img, rgb_gt))
            if args.save_result:
                utils.save_img((os.path.join(result_dir, filenames[batch]+'.png')), restored_img)

if get_psnr:
    psnr = sum(psnr_val_rgb) / len(test_dataset)
    print("PSNR: %f" % psnr)
