from ..detection import get_det_boxes
from ..data import DetectionDataset
from torch.utils.data import DataLoader
import multiprocessing
import numpy as np

import logging
logger = logging.getLogger('__file__')


def create_det_dataset(image_paths, is_roi=True, config={}):
    # build tensors out of images / preprocess n parallel
    det_dataset = DetectionDataset(image_paths, roi_flag=is_roi)
    det_dataloader = DataLoader(
        det_dataset, batch_size=config["det_batch_size_roi"], num_workers=config.get("num_workers"))
    return det_dataloader

def batch_detect(craft_net, refine_net, det_dataloader, config):
    det_batch = next(iter(det_dataloader))
    images = det_batch["images"]
    idxs = det_batch["idxs"]
    y, features = craft_net(images)
    y_refiner = refine_net(y, features)
    score_texts = [y[i, :, :, 0].cpu().data.numpy() for i in idxs]
    score_links = [y_refiner[i, :, :, 0].cpu().data.numpy()
                    for i in idxs]

    # get bounding boxes
    with multiprocessing.Pool(processes=config.get("num_workes")) as p:
        det_op = p.starmap(get_det_boxes, list(zip(score_texts, score_links, idxs)))
    # try:
    #     det_op = [get_det_boxes(x,y,z) for x,y,z in list(zip(score_texts, score_links, idxs))]
    # except Exception:
    #     print("detection failed")
    #     try:
    #         det_op = [get_det_boxes(x,y,z) for x,y,z in list(zip(score_texts, score_links, idxs))]
    #     except Exception:
    #         print("detection failed again")
    return det_op, images

def main(craft_net, refine_net, image_paths, is_roi, config):
    det_dataloader = create_det_dataset(image_paths, is_roi, config)
    det_op, images = batch_detect(craft_net, refine_net, det_dataloader, config)
    input_for_rec_dataset = []
    for bboxes, idx in det_op:
        for i,box in enumerate(bboxes):
            inp = (images[idx], np.array(box), idx, i)
            input_for_rec_dataset.append(inp)
    return input_for_rec_dataset