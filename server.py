from websocket_server import WebsocketServer
import argparse
from pathlib import Path
import os
import torch
import torch.nn as nn
from PIL import Image
from os.path import basename
from os.path import splitext
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import time
from bs4 import BeautifulSoup
from html.parser import HTMLParser
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
import base64

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--data_path", type=str, default="input", help='path to training data')
parser.add_argument("--save_path", type=str, default="output", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=False, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
# parser.add_argument("--test_model", type=str,default='')
# parser.add_argument("--prefix",type=str,default='')
opt = parser.parse_args()

# device = torch.device('cuda:{}'.format(opt.gpu_id[0])) if opt.gpu_id else torch.device('cpu')
device = torch.device('cpu')


def getByte(path):
    with open(path, 'rb') as f:
        img_byte = base64.b64encode(f.read())
    img_str = img_byte.decode('ascii')
    return img_str

def D_BASE64(origStr):
    #base64 decode should meet the padding rules
    if(len(origStr)%3 == 1): 
        origStr += "=="
    elif(len(origStr)%3 == 2): 
        origStr += "=" 

    origStr = bytes(origStr, encoding='utf8')
    dStr = base64.b64decode(origStr).decode()
    print("BASE64 Decode result is: \n" + dStr)
    return dStr

def new_client(client, server):
        print("New client connected and was given id %d" % client['id'])


# Called for every client disconnecting
def client_left(client, server):
        print("Client(%d) disconnected" % client['id'])


# Called when a client sends a message
def message_received(client, server, message):
        if len(message) > 200:
                message = message[:200]+'..'
        # print("Client(%d)_address%s said: %s" % (client['id'],client['address'], message))
        # message_list = message.split('#')
        
        os.makedirs(opt.save_path, exist_ok=True)

        # Build model
        print('Loading model ...\n')
        model = PReNet(opt.recurrent_iter, opt.use_GPU, opt=opt)
        print_network(model)
        if opt.use_GPU:
                model = model.to(device)
        model.load_state_dict(torch.load(os.path.join('H_net_epoch_100.pth'),map_location='cpu'))
        model.eval()

        time_test = 0
        count = 0
        sum_psnr = 0
        sum_ssim = 0
        # for img_name in os.listdir(opt.data_path):


        # if is_image(img_name):
        # img_name = message
        # img_path = os.path.join(opt.data_path, img_name)


        # target_path = "RainTestH/norain/"+"no"+img_name
        # target_path = "RainDisstilationH/rain/"+img_name
        # target = cv2.imread(target_path)
        # input image

        # print("message:",message)
        img_message = message.replace("data:image/png;base64,","")
        img_message += "=="
        print("img_message:",type(img_message),img_message)
        # with open ("1.txt",'wb') as f:
        #         f.write(str(img_message))

        img = base64.b64decode(img_message)
        # print(img)
        file = open('input/rain.png','wb')
        file.write(img)

        y = cv2.imread('input/rain.png')        

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

                # print(img_name, ': ', dur_time)

        if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
        else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())

        save_out = save_out.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out)
        save_out = cv2.merge([r, g, b])

        cv2.imwrite(os.path.join(opt.save_path,'norain.png'), save_out)
        print("Derain done!")

        img_str = getByte(os.path.join(opt.save_path,'norain.png'))
        server.send_message(client,img_str)
        
        # server.send_message(client,'用户编号'+str(client['id'])+':'+message)


server = WebsocketServer(host='localhost',port=5678)
server.set_fn_new_client(new_client)
server.set_fn_client_left(client_left)
server.set_fn_message_received(message_received)
server.run_forever()
