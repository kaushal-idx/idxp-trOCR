import os 
import cv2
import math
from pathlib import Path 
import numpy as np 
from collections import OrderedDict
from typing import Optional, Union
from ..utils import torch_utils as torch_utils
from ..utils import util as file_utils


CRAFT_GDRIVE_URL = "https://drive.google.com/uc?id=1bupFXqT-VU6Jjeul13XP7yx2Sg5IHr4J"
REFINENET_GDRIVE_URL = (
    "https://drive.google.com/uc?id=1xcE9qpJXp4ofINwXWVhhQIh9S8Z7cuGj"
)

# unwarp corodinates
def warp_coord(Minv, pt):
    out = np.matmul(Minv, (pt[0], pt[1], 1))
    return np.array([out[0] / out[2], out[1] / out[2]])


def copy_state_dict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_craftnet_model(
        cuda: bool = False,
        weight_path: Optional[Union[str, Path]] = None
):
    # get craft net path
    if weight_path is None:
        home_path = str(Path.home())
        weight_path = Path(
            home_path,
            ".idxp_trOCR",
            "artefacts",
            "craft_mlt_25k.pth"
        )
    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    # load craft net
    from .craft_net import CraftNet

    craft_net = CraftNet()  # initialize

    # check if weights are already downloaded, if not download
    url = CRAFT_GDRIVE_URL
    if not os.path.isfile(weight_path):
        print("Craft text detector weight will be downloaded to {}".format(weight_path))

        file_utils.download(url=url, save_path=weight_path)

    # arange device
    if cuda:
        craft_net.load_state_dict(copy_state_dict(torch_utils.load(weight_path)))

        craft_net = craft_net.cuda()
        craft_net = torch_utils.DataParallel(craft_net)
        torch_utils.cudnn_benchmark = False
    else:
        craft_net.load_state_dict(
            copy_state_dict(torch_utils.load(weight_path, map_location="cpu"))
        )
    craft_net.eval()
    return craft_net


def load_refinenet_model(
        cuda: bool = False,
        weight_path: Optional[Union[str, Path]] = None
):
    # get refine net path
    if weight_path is None:
        home_path = Path.home()
        weight_path = Path(
            home_path,
            ".idxp_trOCR",
            "artefacts",
            "craft_refiner_CTW1500.pth"
        )
    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    # load refine net
    from .refine_net import RefineNet

    refine_net = RefineNet()  # initialize

    # check if weights are already downloaded, if not download
    url = REFINENET_GDRIVE_URL
    if not os.path.isfile(weight_path):
        print("Craft text refiner weight will be downloaded to {}".format(weight_path))

        file_utils.download(url=url, save_path=weight_path)

    # arange device
    if cuda:
        refine_net.load_state_dict(copy_state_dict(torch_utils.load(weight_path)))

        refine_net = refine_net.cuda()
        refine_net = torch_utils.DataParallel(refine_net)
        torch_utils.cudnn_benchmark = False
    else:
        refine_net.load_state_dict(
            copy_state_dict(torch_utils.load(weight_path, map_location="cpu"))
        )
    refine_net.eval()
    return refine_net

def _get_bbox(textmap, linkmap, text_threshold=0.7, link_threshold=0.4, low_text=0.4):
    # prepare data
    linkmap = linkmap.copy()
    textmap = textmap.copy()
    img_h, img_w = textmap.shape

    """ labeling method """
    _, text_score = cv2.threshold(textmap, low_text, 1, 0)
    _, link_score = cv2.threshold(linkmap, link_threshold, 1, 0)

    text_score_comb = np.clip(text_score + link_score, 0, 1)
    
    nLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb.astype(np.uint8), connectivity=4
    )

    det = []

    for k in range(1, nLabels):
        # size filtering
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # thresholding
        if np.max(textmap[labels == k]) < text_threshold:
            continue

        # make segmentation map
        segmap = np.zeros(textmap.shape, dtype=np.uint8)
        segmap[labels == k] = 255

        # remove link area
        segmap[np.logical_and(link_score == 1, text_score == 0)] = 0

        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
        niter = int(math.sqrt(size * min(w, h) / (w * h)) * 2)
        sx, ex, sy, ey = (x - niter, x + w + niter + 1, y - niter, y + h + niter + 1)

        # boundary check
        if sx < 0:
            sx = 0
        if sy < 0:
            sy = 0
        if ex >= img_w:
            ex = img_w
        if ey >= img_h:
            ey = img_h
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter))
        segmap[sy:ey, sx:ex] = cv2.dilate(segmap[sy:ey, sx:ex], kernel)

        # make box
        np_temp = np.roll(np.array(np.where(segmap != 0)), 1, axis=0)
        np_contours = np_temp.transpose().reshape(-1, 2)
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # boundary check due to minAreaRect may have out of range values 
        # (see https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga3d476a3417130ae5154aea421ca7ead9)
        for p in box:
            if p[0] < 0:
                p[0] = 0
            if p[1] < 0:
                p[1] = 0
            if p[0] >= img_w:
                p[0] = img_w
            if p[1] >= img_h:
                p[1] = img_h

        # align diamond-shape
        w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w, h) / (min(w, h) + 1e-5)
        if abs(1 - box_ratio) <= 0.1:
            l, r = min(np_contours[:, 0]), max(np_contours[:, 0])
            t, b = min(np_contours[:, 1]), max(np_contours[:, 1])
            box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)

        # make clock-wise order
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)
        box = np.array(box)

        det.append(box)
        # mapper.append(k)

    return det, labels

def rescale_coordinates(polys, ratio_w=1, ratio_h=1, ratio_net=2):
    if len(polys) > 0:
        polys = np.array(polys)
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] *= (ratio_w * ratio_net, ratio_h * ratio_net)
    return polys

def get_det_boxes(textmap, linkmap, idx):
    det, labels = _get_bbox(textmap, linkmap)
    det = rescale_coordinates(det)
    return (det, idx.item())