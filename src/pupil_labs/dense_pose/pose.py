import glob
import logging
import os
from enum import Enum
from typing import Dict, Type

import cv2
import densepose
import numpy as np
import torch
from densepose import add_densepose_config
from densepose.structures import (
    DensePoseChartPredictorOutput,
    DensePoseEmbeddingPredictorOutput,
)
from densepose.utils.logger import verbosity_to_level
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.densepose_outputs_vertex import (
    DensePoseOutputsTextureVisualizer,
    DensePoseOutputsVertexVisualizer,
    get_texture_atlases,
)
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.densepose_results_textures import (
    DensePoseResultsVisualizerWithTexture,
    get_texture_atlas,
)
from densepose.vis.extractor import (
    CompoundExtractor,
    DensePoseOutputsExtractor,
    DensePoseResultExtractor,
    create_extractor,
)
from detectron2.config import CfgNode, get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.structures.instances import Instances

# Set my own logger
logger = logging.getLogger("pl-densepose-pose")
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


class PartsDefinition(Enum):
    """Parts definition for DensePose"""

    BACKGROUND = 0
    TORSO_BACK = 1
    TORSO_FRONT = 2
    RIGHT_HAND = 3
    LEFT_HAND = 4
    LEFT_FOOT = 5
    RIGHT_FOOT = 6
    UPPER_LEG_RIGHT_BACK = 7
    UPPER_LEG_LEFT_BACK = 8
    UPPER_LEG_RIGHT_FRONT = 9
    UPPER_LEG_LEFT_FRONT = 10
    LOWER_LEG_RIGHT_BACK = 11
    LOWER_LEG_LEFT_BACK = 12
    LOWER_LEG_RIGHT_FRONT = 13
    LOWER_LEG_LEFT_FRONT = 14
    UPPER_ARM_LEFT_INSIDE = 15
    UPPER_ARM_RIGHT_INSIDE = 16
    UPPER_ARM_LEFT_OUTSIDE = 17
    UPPER_ARM_RIGHT_OUTSIDE = 18
    LOWER_ARM_LEFT_INSIDE = 19
    LOWER_ARM_RIGHT_INSIDE = 20
    LOWER_ARM_LEFT_OUTSIDE = 21
    LOWER_ARM_RIGHT_OUTSIDE = 22
    HEAD_RIGHT = 23
    HEAD_LEFT = 24


def setup_config(min_score=0.7):
    logging.info("Loading config...")
    weights = "https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_DL_s1x/165712097/model_final_0ed407.pkl"
    opts = []
    opts.append("MODEL.ROI_HEADS.SCORE_THRESH_TEST")
    opts.append(min_score)
    cfg = get_cfg()
    add_densepose_config(cfg)
    dir = os.path.dirname(__file__)
    config_file = glob.glob(os.path.join(dir, "config", "[!Base_]*.yaml"))[0]
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    logging.info(f"Loading model from {weights}")
    cfg.MODEL.WEIGHTS = weights

    cfg.MODEL.DEVICE = "cpu"
    cfg.freeze()
    predictor = DefaultPredictor(cfg)

    VISUALIZERS: Type[Dict[str, object]] = {
        "dp_contour": DensePoseResultsContourVisualizer,
        "dp_segm": DensePoseResultsFineSegmentationVisualizer,
        "dp_u": DensePoseResultsUVisualizer,
        "dp_v": DensePoseResultsVVisualizer,
        "dp_iuv_texture": DensePoseResultsVisualizerWithTexture,
        "dp_cse_texture": DensePoseOutputsTextureVisualizer,
        "dp_vertex": DensePoseOutputsVertexVisualizer,
        "bbox": ScoredBoundingBoxVisualizer,
    }
    vis_specs = ["bbox", "dp_contour"]
    visualizers = []
    extractors = []
    for vis_spec in vis_specs:
        vis = VISUALIZERS[vis_spec]()
        visualizers.append(vis)
        extractor = create_extractor(vis)
        extractors.append(extractor)
    visualizer = CompoundVisualizer(visualizers)
    extractor = CompoundExtractor(extractors)

    context = {"extractor": extractor, "visualizer": visualizer}
    visualizer = context["visualizer"]
    extractor = context["extractor"]

    return predictor, visualizer, extractor, cfg


def get_densepose(
    frame, predictor, visualizer, extractor, cfg, xy, frameid=0, labels_onimg=True
):
    with torch.no_grad():
        outputs = predictor(frame)["instances"]
    result = {}
    extractor_r = extractor
    if outputs.has("scores"):
        result["scores"] = outputs.get("scores").cpu()
    if outputs.has("pred_boxes"):
        result["pred_boxes_XYXY"] = outputs.get("pred_boxes").tensor.cpu()
        if outputs.has("pred_densepose"):
            if isinstance(outputs.pred_densepose, DensePoseChartPredictorOutput):
                extractor_r = DensePoseResultExtractor()
            elif isinstance(outputs.pred_densepose, DensePoseEmbeddingPredictorOutput):
                extractor_r = DensePoseOutputsExtractor()
            result["pred_densepose"] = extractor_r(outputs)[0]
    logging.debug(f"DensePose result: {result}")
    # execute on outputs
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.tile(frame[:, :, np.newaxis], (1, 1, 3)) / 255
    data = extractor(outputs)
    id_part = []
    # As of now, it checks the gaze point for labels from densepose.
    # TODO: Take a small area around the gaze point and take the mode of the labels
    if xy is not None and len(result["pred_boxes_XYXY"]) > 0:
        for i, box in enumerate(result["pred_boxes_XYXY"]):
            if xy[0] > box[0] and xy[0] < box[2] and xy[1] > box[1] and xy[1] < box[3]:
                id_part.append(int(result["pred_densepose"][i].labels.max().numpy()))
            else:
                id_part.append(0)
    else:
        id_part.append(0)
    frame_vis = visualizer.visualize(frame, data)
    # Get id name
    id_part = list(set(id_part))
    id_name = []
    for i in range(len(id_part)):
        if id_part != 0:
            id_name.append(PartsDefinition(id_part[i]).name)
    text_id_name = ", ".join(id_name)
    logging.debug(f"DensePose frame {frameid} - looking at part {text_id_name}")

    # write part in the bottom left corner of the image
    if labels_onimg:
        cv2.putText(
            frame_vis,
            text_id_name,
            (10, 1000),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            lineType=1,
        )

    return frame_vis, result, text_id_name
