import os
from pathlib import Path
from urllib import response
from idxp_trocr.prediction.detection_inf import main as do_detection
from idxp_trocr.prediction.recognition_inf import main as do_recognition
from idxp_trocr.recognition import load_trocr_model, load_trocr_processor
from idxp_trocr.detection import load_craftnet_model, load_refinenet_model
from idxp_trocr.utils import read_images_in_dir, build_response

# import logging
# logger = logging.getLogger(__file__)
# logging

# config

cuda = False
config = {
    "cuda":cuda,
    "trocr_model_weights": '/home/n13452/saved_models/trocr_printed_model',
    "trocr_processor_weights": '/home/n13452/saved_models/trocr_printed_processor',
    "det_batch_size_roi": 64,
    "num_workers":8,
    "rec_batch_size":80,
}

craft_net = load_craftnet_model(cuda=config["cuda"], weight_path=config.get("craft_net_weights", None))
refine_net = load_refinenet_model(cuda=config["cuda"], weight_path=config.get("refine_net_weights", None))
trocr_model = load_trocr_model(cuda=config["cuda"], weight_path=config.get("trocr_model_weights", None))
trocr_processor = load_trocr_processor(weight_path=config.get("trocr_processor_weights", None))


# get images
dir_path = "/home/n13452/github/idxp-trOCR/tests/fixtures"
is_roi = True
inp_path = dir_path

def main():
    if os.path.isdir(inp_path):
        image_paths = read_images_in_dir(inp_path)
        inp_for_recognition = do_detection(craft_net, refine_net, image_paths, is_roi, config)
        rec_op_collection = do_recognition(inp_for_recognition, trocr_processor, trocr_model, config)
        response= build_response(image_paths, rec_op_collection)
        return response
    else:
        return {"error": "invalid_request, check if path passed exist or is accessible"}

if __name__ == "__main__":
    jsn = main()
    import json
    with open("res.json","w") as f:
        f.writelines(json.dumps(jsn))


# import pickle
# f = open("/home/n13452/github/idxp-trOCR/idxp_trocr/experiments/input_for_rec_dataset.pkl", 'rb')
# inp_for_recognition = pickle.load(f)

# rec_op_collection = [{'word_num': [0, 1, 2, 0, 0, 0, 1, 0], 'left': [742, 0, 738, 43, 12, 257, 130, 0], 'right': [1026, 505, 1026, 953, 625, 852, 212, 913], 'top': [0, 67, 140, 79, 25, 66, 126, 113], 'bottom': [92, 169, 234, 271, 164, 243, 258, 217], 'text': ['VALU', 'EM VALUE', 'ATION', 'VALUE $ 150,000', '1B9U120247B089041', '35,000', '$', 'POLICY PREMIUM'], 'conf': [0.5173636674880981, 0.5128017067909241, 0.6104872226715088, 0.7209446430206299, 0.9430790543556213, 0.5575735569000244, 0.5030555725097656, 0.8403480052947998], 'idxs': [0, 0, 0, 1, 2, 3, 3, 4]}]