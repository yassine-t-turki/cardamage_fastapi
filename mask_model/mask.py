from config import metadata_cardmg
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import cv2
from PIL import Image
import os
classes = ['Cracked', 'Scratch', 'Flaking', 'Broken part', 'Corrosion', 'Dent','Paint chip','Missing part']


def run_mask(threshold_mask: float, file_path: str):
    cfg = get_cfg()
    cfg.merge_from_file(os.path.join("mask_model", "cfg_file.yaml"))  # Update this path
    cfg.MODEL.WEIGHTS = os.path.join("mask_model", "maskrcnn.pth")  # Update this path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_mask
    predictor = DefaultPredictor(cfg)
    im = cv2.imread(file_path)
    output = predictor(im)
    v = Visualizer(im[:, :, ::-1], metadata=metadata_cardmg, scale=1.5)
    out = v.draw_instance_predictions(output["instances"].to("cpu"))
    annotated_image = Image.fromarray(out.get_image())
    class_indices = output["instances"].pred_classes
    mapped_classes = [classes[index.item()] for index in class_indices]
    return annotated_image, mapped_classes