import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
import time 
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--data_path", type=str, default="input", help='path to training data')
parser.add_argument("--save_path", type=str, default="test_results", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
# parser.add_argument("--test_model", type=str,default='')
# parser.add_argument("--prefix",type=str,default='')
opt = parser.parse_args()

# device = torch.device('cuda:{}'.format(opt.gpu_id[0])) if opt.gpu_id else torch.device('cpu')
device = torch.device('cpu')


# def main():

os.makedirs(os.path.join(opt.save_path, 'test_x'), exist_ok=True)

# Build model
print('Loading model ...\n')
model = PReNet(opt.recurrent_iter, opt.use_GPU, opt=opt)
print_network(model)
if opt.use_GPU:
    model = model.to(device)
model.load_state_dict(torch.load('H_net_epoch_100.pth'))
model.eval()

time_test = 0
count = 0
sum_psnr = 0
sum_ssim = 0
# for img_name in os.listdir(opt.data_path):
# if is_image(img_name):
img_name = 
img_path = os.path.join(opt.data_path, img_name)
# target_path = "RainTestH/norain/"+"no"+img_name
# target_path = "RainDisstilationH/rain/"+img_name
# target = cv2.imread(target_path)
# input image
y = cv2.imread(img_path)
b, g, r = cv2.split(y)
y = cv2.merge([r, g, b])
#y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

y = normalize(np.float32(y))
y = np.expand_dims(y.transpose(2, 0, 1), 0)
y = Variable(torch.Tensor(y))

if opt.use_GPU:
    y = y.to(device)

with torch.no_grad(): #
    if opt.use_GPU:
        torch.cuda.synchronize()
    start_time = time.time()

    out, _ = model(y)
    out = torch.clamp(out, 0., 1.)

    if opt.use_GPU:
        torch.cuda.synchronize()
    end_time = time.time()
    dur_time = end_time - start_time
    time_test += dur_time

    print(img_name, ': ', dur_time)

if opt.use_GPU:
    save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
else:
    save_out = np.uint8(255 * out.data.numpy().squeeze())

save_out = save_out.transpose(1, 2, 0)
b, g, r = cv2.split(save_out)
save_out = cv2.merge([r, g, b])

cv2.imwrite(os.path.join(opt.save_path, 'test_x', 'no'+img_name), save_out)
            # count += 1
            # psnr = compare_psnr(save_out, target)
            # ssim = compare_ssim(save_out, target, multichannel=True)
            # sum_psnr += psnr
            # sum_ssim += ssim
    # avg_psnr = sum_psnr/100
    # avg_ssim = sum_ssim/100
    # print('Avg. time:', time_test/count)
    # print('Avg. psnr:',avg_psnr)
    # print('Avg. ssim:',avg_ssim)

# if __name__ == "__main__":
#     main()

