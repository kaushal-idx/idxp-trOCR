import os
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__file__)

def load_trocr_processor(weight_path: Optional[Union[str, Path]] = None, checkpoint=None):  
    if weight_path is None:
        home_path = str(Path.home())
        weight_path = Path(
            home_path,
            ".idxp_trOCR",
            "artefacts",
            "trocr_processor"
        )

    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    from transformers import TrOCRProcessor
    
    if not os.path.isdir(weight_path):
        logger.info(f"trocr processor weight will be downloaded to {weight_path}")
        if checkpoint:
            trocr_processor = TrOCRProcessor.from_pretrained(checkpoint)
            trocr_processor.save_pretrained(weight_path)
        else:
            raise ValueError("please provide downloaded weights or checkpoint from HF")        
    else:
        trocr_processor = TrOCRProcessor.from_pretrained(weight_path)
    return trocr_processor


def load_trocr_model(cuda, weight_path: Optional[Union[str, Path]] = None, checkpoint = None):  
    if weight_path is None:
        home_path = str(Path.home())
        weight_path = Path(
            home_path,
            ".idxp_trOCR",
            "artefacts",
            "trocr_model"
        )

    weight_path = Path(weight_path).resolve()
    weight_path.parent.mkdir(exist_ok=True, parents=True)
    weight_path = str(weight_path)

    from transformers import VisionEncoderDecoderModel
    
    if not os.path.isdir(weight_path):
        logger.info("trocr model weights will be downloaded to {}".format(weight_path))
        if checkpoint:
            trocr_model = VisionEncoderDecoderModel.from_pretrained(checkpoint)
            trocr_model.save_pretrained(weight_path)
        else:
            raise ValueError("please provide downloaded weights or checkpoint from HF")
    else:
        trocr_model = VisionEncoderDecoderModel.from_pretrained(weight_path)
    
    if cuda:
        trocr_model.to("cuda")
    else:
        trocr_model.to("cpu")
        
    trocr_model.config.eos_token_id = 2
    return trocr_model