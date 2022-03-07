from .util import Coordinates
import logging
import bson
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def _get_bounding_box(data, coords):
    if coords == Coordinates.FOUR:
        box_data = {
            "Left": data["left"],
            "Top": data["top"],
            "Width": data["width"],
            "Height": data["height"]
        }

    elif coords == Coordinates.EIGHT:
        box_data = {
            "x1": data["left"],
            "y1": data["top"],
            "x2": data["right"],
            "y2": data["top"],
            "x3": data["right"],
            "y3": data["bottom"],
            "x4": data["left"],
            "y4": data["bottom"],
        }
    else:
        raise ValueError(f"invalid coordinates: {coords}")
    return box_data


def _get_rotation_resolution(image):
    """
    to be implemented
    """
    height, width = image.shape[:2]
    return 0, {
        "Height": height,
        "Width": width
    }

def _find(arr, id, value):
    return next((x for x in arr if x[id] == value), None)

def _get_uuid():
    return str(bson.objectid.ObjectId())

def df_to_native_jsn(df, image, coords):
    """
    converts tesseract output dataframe to native json
    Args
        df: dataframe from tesseract, pytesseract_agent.detect output
        coord: Coordinates system
    """
    assert isinstance(df,pd.DataFrame), "please pass a dataframe"
    assert isinstance(image,np.ndarray), f"works only with numpy array {image}"
    assert isinstance(coords, Coordinates)

    blocks = []
    orientation, resolution = _get_rotation_resolution(image)
    page_data = {
        "Orientation": orientation,
        "Resolution": resolution,
    }

    try:
        for index, row in df.iterrows():
            block = _find(blocks, "Block_No", row["block_num"])
            if block is None:
                block = dict(
                    {"Id": _get_uuid(), "Block_No": row["block_num"], "Type": "block", "Lines": []})
                blocks.append(block)
            line = _find(block.get("Lines"),
                                "Line_no", row["line_num"])
            if line is None:
                line = dict(
                    {"Id": _get_uuid(), "Line_no": row["line_num"], "Type": "Line", "Words": []})
                block.get("Lines").append(line)
            line.get("Words").append(dict({
                "Id": _get_uuid(),
                "Type": "Word",
                "Text": row["text"],
                "Confidence": row["conf"],
                "BoundingBox": _get_bounding_box(row, coords)
            }))
    except Exception as e:
        logger.exception(str(e))

    page_data["Blocks"] = blocks
    return page_data