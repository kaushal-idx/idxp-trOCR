from fastapi import FastAPI
from pathlib import Path
import os
import sys
from idxp_trocr import (load_craftnet_model,
                        load_refinenet_model,
                        load_trocr_model, 
                        load_trocr_processor,
                        read_images_in_dir,
                        do_detection,
                        do_recognition,
                        build_response)


config = {

}

craft_net = load_craftnet_model(cuda=config["cuda"], weight_path=config.get("craft_net_weights", None))
refine_net = load_refinenet_model(cuda=config["cuda"], weight_path=config.get("refine_net_weights", None))
trocr_model = load_trocr_model(cuda=config["cuda"], weight_path=config.get("trocr_model_weights", None))
trocr_processor = load_trocr_processor(weight_path=config.get("trocr_processor_weights", None))


app = FastAPI()

@app.get("/")
def check():
    return {"message": "all goood"}

@app.get("/ocr/{dir_path}/")
def do_ocr(dir_path, is_roi=True):
    inp_path = Path(dir_path)
    if os.path.isdir(inp_path):
        image_paths = read_images_in_dir(inp_path)
        inp_for_recognition = do_detection(craft_net, refine_net, image_paths, is_roi, config)
        rec_op_collection = do_recognition(inp_for_recognition)
        response= build_response(rec_op_collection)
        return response
    else:
        return {"error": "invalid_request, check if path passed exist or is accessible"}
