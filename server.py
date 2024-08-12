from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, StreamingResponse
from typing import List
import os
from PIL import Image
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import supervision as sv
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from detectron2.data import MetadataCatalog

# Initialize FastAPI
app = FastAPI()

# Initialize inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="mMBVXsXJjIB3RSeH0SRP"
)

# Define directories
UPLOAD_DIR = "uploads"
ANNOTATED_DIR = "uploads/annotated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)

# Initialize Jinja2 template renderer
templates = Jinja2Templates(directory="templates")

# Access the metadata catalog and register metadata for the dataset
metadata_cardmg = MetadataCatalog.get('car_data')
metadata_cardmg.set(
    json_file='coco_annotations_damage.json',
    image_root='uploads/',
    evaluator_type='coco',
    thing_classes=['Cracked', 'Scratch', 'Flaking', 'Broken part', 'Corrosion', 'Dent', 'Paint chip', 'Missing part'],
    thing_dataset_id_to_contiguous_id={
        1: 0,
        2: 1,
        3: 2,
        4: 3,
        5: 4,
        6: 5,
        7: 6,
        8: 7
    }
)

# Helper function to get list of uploaded files
def get_uploaded_files() -> List[str]:
    return [f for f in os.listdir(UPLOAD_DIR) if os.path.isfile(os.path.join(UPLOAD_DIR, f))]

# Helper function to get sv.Position from string
def get_position_from_string(position: str) -> sv.Position:
    positions = {
        "CENTER": sv.Position.CENTER,
        "CENTER_LEFT": sv.Position.CENTER_LEFT,
        "CENTER_RIGHT": sv.Position.CENTER_RIGHT,
        "TOP_CENTER": sv.Position.TOP_CENTER,
        "TOP_LEFT": sv.Position.TOP_LEFT,
        "TOP_RIGHT": sv.Position.TOP_RIGHT,
        "BOTTOM_LEFT": sv.Position.BOTTOM_LEFT,
        "BOTTOM_CENTER": sv.Position.BOTTOM_CENTER,
        "BOTTOM_RIGHT": sv.Position.BOTTOM_RIGHT
    }
    return positions.get(position.upper(), sv.Position.TOP_RIGHT)  # Default to TOP_RIGHT

def resize_image(image, base_height=None, base_width=None):
    """
    Resize the image while keeping the aspect ratio.
    """
    if base_height:
        aspect_ratio = base_height / float(image.height)
        new_width = int(aspect_ratio * image.width)
        new_size = (new_width, base_height)
    elif base_width:
        aspect_ratio = base_width / float(image.width)
        new_height = int(aspect_ratio * image.height)
        new_size = (base_width, new_height)
    else:
        return image

    return image.resize(new_size)

def combine_images(image1, image2, align='horizontal'):
    """
    Combine two images either horizontally or vertically.
    """
    if align == 'horizontal':
        # Resize images to the same height
        height1, height2 = image1.height, image2.height
        max_height = max(height1, height2)

        image1_resized = resize_image(image1, base_height=max_height)
        image2_resized = resize_image(image2, base_height=max_height)

        # Create a new image with combined width
        combined_width = image1_resized.width + image2_resized.width
        combined_image = Image.new('RGB', (combined_width, max_height))

        # Paste the images into the combined image
        combined_image.paste(image1_resized, (0, 0))
        combined_image.paste(image2_resized, (image1_resized.width, 0))

    elif align == 'vertical':
        # Resize images to the same width
        width1, width2 = image1.width, image2.width
        max_width = max(width1, width2)

        image1_resized = resize_image(image1, base_width=max_width)
        image2_resized = resize_image(image2, base_width=max_width)

        # Create a new image with combined height
        combined_height = image1_resized.height + image2_resized.height
        combined_image = Image.new('RGB', (max_width, combined_height))

        # Paste the images into the combined image
        combined_image.paste(image1_resized, (0, 0))
        combined_image.paste(image2_resized, (0, image1_resized.height))

    return combined_image

# Route for the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, image_display: str = '', label_position: str = "TOP_RIGHT", threshold_bbox: float = 0.4, threshold_mask: float = 0.4, file_name: str = '', chosen_model: str = 'bounding_box'):
    images = get_uploaded_files()
    image_options = "".join(f'<option value="{img}">{img}</option>' for img in images)
    label_positions = [
        "CENTER", "CENTER_LEFT", "CENTER_RIGHT", "TOP_CENTER", "TOP_LEFT", "TOP_RIGHT", 
        "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"
    ]
    label_position_options = "".join(f'<option value="{pos}" {"selected" if pos == label_position else ""}>{pos}</option>' for pos in label_positions)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_options": image_options,
        "image_display": image_display,
        "label_position_options": label_position_options,
        "selected_label_position": label_position,
        "selected_threshold_mask": threshold_mask,
        "selected_threshold_bbox": threshold_bbox,
        "file_name": file_name,
        "selected_model": chosen_model
    })

# Route to handle image upload
@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...),
                          threshold_mask: float = Form(default=0.4),
                          threshold_bbox: float = Form(default=0.4),
                          label_position: str = Form(default="TOP_RIGHT"),
                          chosen_model: str = Form(default="bounding_box")):
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    annotated_image = None

    if chosen_model == "bounding_box":
        print("Bounding box model selected")  # Debugging line
        custom_configuration = InferenceConfiguration(confidence_threshold=threshold_bbox)
        CLIENT.configure(custom_configuration)
        results = CLIENT.infer(file_path, model_id="cardamage-l4vtd/1")
        image = Image.open(file_path)
        detections = sv.Detections.from_inference(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=get_position_from_string(label_position))
        
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    elif chosen_model == "mask_segmentation":
        print("Mask segmentation model selected")  # Debugging line
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join("model", "cfg_file.yaml"))  # Update this path
        cfg.MODEL.WEIGHTS = os.path.join("model", "maskrcnn.pth")  # Update this path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_mask
        predictor = DefaultPredictor(cfg)
        
        im = cv2.imread(file_path)
        output = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata_cardmg, scale=1.5)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        annotated_image = Image.fromarray(out.get_image())

    elif chosen_model == "both_models":
        print("Both models selected")  # Debugging line
        # Bounding Box
        custom_configuration = InferenceConfiguration(confidence_threshold=threshold_bbox)
        CLIENT.configure(custom_configuration)
        results = CLIENT.infer(file_path, model_id="cardamage-l4vtd/1")
        image = Image.open(file_path)
        detections = sv.Detections.from_inference(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=get_position_from_string(label_position))
        
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Mask Segmentation
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join("model", "cfg_file.yaml"))  # Update this path
        cfg.MODEL.WEIGHTS = os.path.join("model", "maskrcnn.pth")  # Update this path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_mask
        predictor = DefaultPredictor(cfg)
        
        im = cv2.imread(file_path)
        output = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata_cardmg, scale=1.5)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        mask_image = Image.fromarray(out.get_image())

        # Combine images
        combined_image = combine_images(annotated_image, mask_image, align='horizontal')
        annotated_image = combined_image

    if annotated_image:
        annotated_image_path = os.path.join(ANNOTATED_DIR, file.filename)
        annotated_image.save(annotated_image_path)

    images = get_uploaded_files()
    image_options = "".join(f'<option value="{img}">{img}</option>' for img in images)
    label_positions = [
        "CENTER", "CENTER_LEFT", "CENTER_RIGHT", "TOP_CENTER", "TOP_LEFT", "TOP_RIGHT", 
        "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"
    ]
    label_position_options = "".join(f'<option value="{pos}" {"selected" if pos == label_position else ""}>{pos}</option>' for pos in label_positions)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_options": image_options,
        "image_display": f'<h2>Annotated Image:</h2><img src="/images/{file.filename}" alt="Annotated Image">',
        "label_position_options": label_position_options,
        "selected_label_position": label_position,
        "selected_threshold_mask": threshold_mask,
        "selected_threshold_bbox": threshold_bbox,
        "file_name": file.filename,
        "selected_model": chosen_model
    })

# Route to retrieve images
@app.get("/images/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(ANNOTATED_DIR, filename)
    
    if not os.path.exists(file_path):
        return HTMLResponse(content="File not found", status_code=404)
    
    return StreamingResponse(open(file_path, "rb"), media_type="image/png")

# Route to rerun inference
@app.post("/rerun")
async def rerun_inference(request: Request, filename: str = Form(...),
                          threshold_mask_rerun: float = Form(default=0.4),
                          threshold_bbox_rerun: float = Form(default=0.4),
                          label_position: str = Form(default="TOP_RIGHT"),
                          chosen_model_rerun: str = Form(default="bounding_box")):
    
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return HTMLResponse(content="File not found", status_code=404)

    annotated_image = None
    if chosen_model_rerun == "bounding_box":
        print("Bounding box model selected")  # Debugging line
        custom_configuration = InferenceConfiguration(confidence_threshold=threshold_bbox_rerun)
        CLIENT.configure(custom_configuration)
        results = CLIENT.infer(file_path, model_id="cardamage-l4vtd/1")
        image = Image.open(file_path)
        detections = sv.Detections.from_inference(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=get_position_from_string(label_position))
        
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

    elif chosen_model_rerun == "mask_segmentation":
        print("Mask segmentation model selected")  # Debugging line
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join("model", "cfg_file.yaml"))  # Update this path
        cfg.MODEL.WEIGHTS = os.path.join("model", "maskrcnn.pth")  # Update this path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_mask_rerun
        predictor = DefaultPredictor(cfg)
        
        im = cv2.imread(file_path)
        output = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata_cardmg, scale=1.5)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        annotated_image = Image.fromarray(out.get_image())

    elif chosen_model_rerun == "both_models":
        print("Both models selected")  # Debugging line
        # Bounding Box
        custom_configuration = InferenceConfiguration(confidence_threshold=threshold_bbox_rerun)
        CLIENT.configure(custom_configuration)
        results = CLIENT.infer(file_path, model_id="cardamage-l4vtd/1")
        image = Image.open(file_path)
        detections = sv.Detections.from_inference(results)

        bounding_box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator(text_position=get_position_from_string(label_position))
        
        annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

        # Mask Segmentation
        cfg = get_cfg()
        cfg.merge_from_file(os.path.join("model", "cfg_file.yaml"))  # Update this path
        cfg.MODEL.WEIGHTS = os.path.join("model", "maskrcnn.pth")  # Update this path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold_mask_rerun
        predictor = DefaultPredictor(cfg)
        
        im = cv2.imread(file_path)
        output = predictor(im)
        v = Visualizer(im[:, :, ::-1], metadata=metadata_cardmg, scale=1.5)
        out = v.draw_instance_predictions(output["instances"].to("cpu"))
        mask_image = Image.fromarray(out.get_image())

        # Combine images
        combined_image = combine_images(annotated_image, mask_image, align='horizontal')

        annotated_image = combined_image

    if annotated_image:
        annotated_image_path = os.path.join(ANNOTATED_DIR, filename)
        annotated_image.save(annotated_image_path)

    images = get_uploaded_files()
    image_options = "".join(f'<option value="{img}">{img}</option>' for img in images)
    label_positions = [
        "CENTER", "CENTER_LEFT", "CENTER_RIGHT", "TOP_CENTER", "TOP_LEFT", "TOP_RIGHT", 
        "BOTTOM_LEFT", "BOTTOM_CENTER", "BOTTOM_RIGHT"
    ]
    label_position_options = "".join(f'<option value="{pos}" {"selected" if pos == label_position else ""}>{pos}</option>' for pos in label_positions)
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "image_options": image_options,
        "image_display": f'<h2>Annotated Image:</h2><img src="/images/{filename}" alt="Annotated Image">',
        "label_position_options": label_position_options,
        "selected_label_position": label_position,
        "selected_threshold_mask": threshold_mask_rerun,
        "selected_threshold_bbox": threshold_bbox_rerun,
        "file_name": filename,
        "selected_model": chosen_model_rerun
    })