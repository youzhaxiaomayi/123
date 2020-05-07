#-*-coding:utf-8-*-
import torch.nn.functional as F
import torch
import torch.nn as nn
import torchvision
from models import Darknet
import cv2
from utils.utils import letterbox 
from utils.utils import non_max_suppression 
from utils.utils import scale_coords
import numpy as np
import random


def detect():
    path = './data/bus.png'
    im0 = cv2.imread(path)  # BGR
    assert im0 is not None, 'Image Not Found ' + path
    # img = letterbox(im0, (608,608 ), )[0]
    img = cv2.resize(im0, (608, 608))
    # img = im0
    draw_img = img
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img_size = (608, 608) 
    device = torch.device('cpu')
    cfg = './cfg/yolov4.cfg'
    model = Darknet(cfg, img_size)
    weights = './weights/yolov4.pt'
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    model.to(device).eval()
    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)

    # Apply NMS
    pred[:,:,:4] *= torch.Tensor(img_size*2) 
    pred = non_max_suppression(pred) 

    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det:
                if conf > 0.7:
                    c1 = (int(xyxy[0].item()), int(xyxy[1].item()))
                    c2 = (int(xyxy[2].item()), int(xyxy[3].item()))
                    # color = tuple(np.random.randint(0,255,3))
                    # import ipdb;ipdb.set_trace()
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    
                    cv2.rectangle(draw_img, c1, c2, color) 
                    print(conf.item(), cls.item())

    cv2.imshow("123", draw_img)
    cv2.waitKey(10000)
if __name__ == '__main__':
    detect()
    # path = './data/bus.png'
    # im0 = cv2.imread(path)  # BGR
    # img = cv2.resize(im0, (608, 608))
    # cv2.imwrite("data/bus608.png", img)
