import json
from urllib import response
import pandas as pd

from idxp_trocr.utils.util import Coordinates
from .df_to_json import df_to_native_jsn
from .util import read_image
import numpy as np


def response_collection_to_df(result):
    df = pd.concat([pd.DataFrame.from_dict(x) for x in result])
    df["block_num"] = df["word_num"]
    df["line_num"] = df["word_num"]
    df_groups = df.groupby("idxs")
    for idx, group in df_groups:
        yield (idx, group)


def build_response(image_paths, result):
    response = []
    for i, df in response_collection_to_df(result):
        image = np.array(read_image(image_paths[i]))
        jsn = df_to_native_jsn(df, image, coords=Coordinates.EIGHT)
        response.append(
            {
                "jsn": jsn,
                "image_path": image_paths[i]
            }
        )
    return response
