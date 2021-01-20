"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from PIL import Image

from scipy import ndimage

import cv2
from skimage import io
import numpy as np
import craft_utils
import imgproc
import file_utils
import json
import zipfile

from craft import CRAFT

from collections import OrderedDict

HEIGH_FITTER = 1.4


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


parser = argparse.ArgumentParser(description='CRAFT Text Detection')
parser.add_argument('--trained_model', default='weights/craft_mlt_25k.pth', type=str, help='pretrained model')
parser.add_argument('--text_threshold', default=0.7, type=float, help='text confidence threshold')
parser.add_argument('--low_text', default=0.3, type=float, help='text low-bound score')
parser.add_argument('--link_threshold', default=0.4, type=float, help='link confidence threshold')
parser.add_argument('--cuda', default=True, type=str2bool, help='Use cuda for inference')
parser.add_argument('--canvas_size', default=1280, type=int, help='image size for inference')
parser.add_argument('--mag_ratio', default=1.5, type=float, help='image magnification ratio')
parser.add_argument('--poly', default=False, action='store_true', help='enable polygon type')
parser.add_argument('--show_time', default=False, action='store_true', help='show processing time')
parser.add_argument('--test_folder', default='/data/', type=str, help='folder path to input images')
parser.add_argument('--refine', default=False, action='store_true', help='enable link refiner')
parser.add_argument('--refiner_model', default='weights/craft_refiner_CTW1500.pth', type=str,
                    help='pretrained refiner model')

args = parser.parse_args()

""" For test images in a folder """
image_list, _, _ = file_utils.get_files(args.test_folder)

result_folder = './result/'
if not os.path.isdir(result_folder):
    os.mkdir(result_folder)

import math


class Utils:

    @staticmethod
    def dot_product(v1, v2):
        return sum((a * b) for a, b in zip(v1, v2))

    @staticmethod
    def length(v):
        return math.sqrt(Utils.dot_product(v, v))

    @staticmethod
    def angle(v1, v2):
        _ = Utils.dot_product(v1, v2) / (Utils.length(v1) * Utils.length(v2))
        if abs(_) > 1:
            _ = abs(_) / _
        return math.acos(_)

    @staticmethod
    def rotate(point, angle, type="deg"):
        px, py = point

        if type == "deg":
            angle = angle * math.pi / 180

        qx = math.cos(angle) * px - math.sin(angle) * py
        qy = math.sin(angle) * px + math.cos(angle) * py
        return np.array([qx, qy])

    @staticmethod
    def draw(target, point, width, color):
        #
        # param
        #   color = [255, 255, 255]
        #   target : image (m,n,3)
        #   width : px
        #   point : [x, y]
        #
        target[
        int(point[1]) - int(width / 2):int(point[1]) + width,
        int(point[0]) - int(width / 2):int(point[0]) + width
        ] = color


def test_net(network, input_image, text_threshold, link_threshold, low_text, cuda, poly, refine_network=None):
    t0 = time.time()

    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(input_image, args.canvas_size,
                                                                          interpolation=cv2.INTER_LINEAR,
                                                                          mag_ratio=args.mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    with torch.no_grad():
        y, feature = network(x)

    # make score and link map
    score_text = y[0, :, :, 0].cpu().data.numpy()
    score_link = y[0, :, :, 1].cpu().data.numpy()

    # refine link
    if refine_network is not None:
        with torch.no_grad():
            y_refiner = refine_network(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().data.numpy()

    t0 = time.time() - t0
    t1 = time.time()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    t1 = time.time() - t1

    # render results (optional)
    # render_img = score_text.copy()
    # render_img = np.hstack((render_img, score_link))
    # ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    if args.show_time: print("\ninfer/postproc time : {:.3f}/{:.3f}".format(t0, t1))

    return boxes


def auto_crop(image_dir, box_list, result_dir):
    filename, file_ext = os.path.splitext(os.path.basename(image_dir))
    img = cv2.imread(image_dir)

    img_shape = img.shape
    img_w = img_shape[1]  # width
    img_h = img_shape[0]  # height
    img_center = np.array([img_w / 2, img_h / 2])  # center point
    # print("shape ", img_shape)
    for i, bbox in enumerate(box_list):
        box = {
            "a": np.array(bbox[0]),
            "b": np.array(bbox[1]),
            "c": np.array(bbox[2]),
            "d": np.array(bbox[3])}
        _3edges = {
            "ab": np.linalg.norm(box["b"] - box["a"]),
            "ac": np.linalg.norm(box["c"] - box["a"]),
            "ad": np.linalg.norm(box["d"] - box["a"])
        }

        _3edges_sorted = dict(sorted(_3edges.items(), key=lambda item: item[1]))

        box_w = list(_3edges_sorted.values())[1]  # width  of the text box: 'ab' => AB
        box_h = list(_3edges_sorted.values())[0]  # height of the text box: 'ad' => AD
        if box_h < 3 or box_w < 3:
            continue
        box_ab = list(_3edges_sorted.keys())[1]

        #
        #    A-------B
        #    |       |
        #    D-------C
        #

        point_a, point_b = box[box_ab[0]], box[box_ab[1]]  # A ,  B

        box_center = sum(box.values()) / 4
        box_center_o = box_center - img_center
        vector_ab = point_a - point_b  # vector _x to _y
        radian = Utils.angle(vector_ab, np.array([1, 0]))  # angle (Rad)
        degree = radian * 180 / math.pi  # angle (Deg)

        rotation_dir = -1  # -1 : counter clockwise;  1: clockwise
        if Utils.dot_product(vector_ab, [0, 1]) < 0:
            rotation_dir = 1

        rotation = rotation_dir * ((180 if degree > 90 else 0) - degree)  # rotation (Deg)

        # print("---" * 10)
        # print("box ", bbox)
        # print("edge ", _3edges_sorted)
        # print("width ", box_w, " = ", vector_ab)
        # print("rad ", radian, " => ", degree, "(deg)")
        # print("rotation~", rotation)

        # image = cv2.imread("result/res_" + filename + ".jpg")

        red = [0, 0, 255]
        # Utils.draw(img, point_a, 10, red)
        # Utils.draw(img, point_b, 10, red)
        # Utils.draw(img, img_center, 10, [0, 255, 0])

        rotated_img = ndimage.rotate(img, rotation)

        rotated_img_shape = rotated_img.shape
        rotated_img_w = rotated_img_shape[1]  # width
        rotated_img_h = rotated_img_shape[0]  # height
        rotated_img_center = np.array([rotated_img_w / 2, rotated_img_h / 2])  # center point

        box_cc = {k: (Utils.rotate(box[k] - img_center, -rotation) + rotated_img_center) for k in box}

        rotated_box_center = sum(box_cc.values()) / 4

        h, w = int(box_h * HEIGH_FITTER), int(box_w)
        x, y = int(rotated_box_center[0] - w / 2), int(rotated_box_center[1] - h / 2)

        # for point in box_cc.values():
        #     draw(rotated_img, point, 10, [255, 255, 255])

        # print(rotated_img_shape, x, y, h, w)
        crop_img = rotated_img[y:y + h, x:x + w]
        try:
            cv2.imwrite(os.path.join(result_dir, filename + "_" + str(i) + ".jpg"), crop_img)
        except:
            """saved failed"""


if __name__ == '__main__':
    # load net
    net = CRAFT()  # initialize

    print('Loading weights from checkpoint (' + args.trained_model + ')')
    if args.cuda:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model)))
    else:
        net.load_state_dict(copyStateDict(torch.load(args.trained_model, map_location='cpu')))

    if args.cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    if args.refine:
        from refinenet import RefineNet

        refine_net = RefineNet()
        print('Loading weights of refiner from checkpoint (' + args.refiner_model + ')')
        if args.cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(args.refiner_model, map_location='cpu')))

        refine_net.eval()
        args.poly = True

    t = time.time()

    # load data
    for k, image_path in enumerate(image_list):
        print("Test image {:d}/{:d}: {:s}".format(k + 1, len(image_list), image_path), end='\r')
        image = imgproc.loadImage(image_path)

        boxes = test_net(net, image, args.text_threshold, args.link_threshold, args.low_text,
                         args.cuda, args.poly, refine_net)

        # save score text
        # mask_file = result_folder + "/res_" + filename + '_mask.jpg'
        # cv2.imwrite(mask_file, score_text)
        # file_utils.saveResult(image_path, image[:, :, ::-1], polys, dirname=result_folder)
        try:
            auto_crop(image_path, boxes, result_folder)
        except:
            """error"""
            continue

    # rotation angle in degree
    print("elapsed time : {}s".format(time.time() - t))
