from config import UPLOAD_DIR, ANNOTATED_DIR, templates
from utils import get_uploaded_files
from roboflow_model.roboflow_model import run_bbox
from mask_model.mask import run_mask
from utils import combine_images
from chatbot.chatbot import ChatSession
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi import File, UploadFile, Form
from fastapi.requests import Request
from chatbot.chatbot import ChatSession
import json
import os

async def read_root_wrapper(request: Request, image_display: str = '', label_position: str = "TOP_RIGHT", 
                            threshold_bbox: float = 0.4, threshold_mask: float = 0.4,
                            file_name: str = '', chosen_model: str = 'bounding_box'):
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



async def upload_image_wrapper(request: Request, file: UploadFile = File(...),
                          threshold_mask: float = Form(default=0.4),
                          threshold_bbox: float = Form(default=0.4),
                          label_position: str = Form(default="TOP_RIGHT"),
                          chosen_model: str = Form(default="bounding_box"), chat_session: ChatSession = ChatSession()):
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    annotated_image = None
    metadata = {}

    if chosen_model == "bounding_box":
        annotated_image, metadata = run_bbox(threshold_bbox, file_path, label_position)
        
    elif chosen_model == "mask_segmentation":
        annotated_image, metadata = run_mask(threshold_mask, file_path)

    elif chosen_model == "all_models":
        annotated_image, metadata_bbox = run_bbox(threshold_bbox, file_path, label_position)
        mask_image, metadata_mask = run_mask(threshold_mask, file_path)
        combined_image = combine_images(annotated_image, mask_image, align='horizontal')
        annotated_image = combined_image
        metadata = {"roboflow_detections":metadata_bbox }

    if annotated_image:
        annotated_image_path = os.path.join(ANNOTATED_DIR, file.filename)
        annotated_image.save(annotated_image_path)

    # Initialize ChatSession and pass metadata
    try:
        metadata_json = json.dumps(metadata)
    except Exception as e:
        metadata_json = metadata
        print(f"Error serializing metadata to JSON: {e}")
    print("Passing this metadata to chatbot")
    print(metadata_json)
    chatbot_response = chat_session.run_request(f"Here is the car damage metadata: {metadata_json}")

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
        "selected_model": chosen_model,
        "chatbot_response": chatbot_response  # Pass chatbot response to the template
    })


async def get_image_wrapper(filename: str):
    file_path = os.path.join(ANNOTATED_DIR, filename)
    
    if not os.path.exists(file_path):
        return HTMLResponse(content="File not found", status_code=404)
    
    return StreamingResponse(open(file_path, "rb"), media_type="image/png")

async def rerun_inference_wrapper(request: Request, filename: str = Form(...),
                          threshold_mask_rerun: float = Form(default=0.4),
                          threshold_bbox_rerun: float = Form(default=0.4),
                          label_position: str = Form(default="TOP_RIGHT"),
                          chosen_model_rerun: str = Form(default="bounding_box"), chat_session: ChatSession = ChatSession()):
    
    file_path = os.path.join(UPLOAD_DIR, filename)

    if not os.path.exists(file_path):
        return HTMLResponse(content="File not found", status_code=404)

    annotated_image = None
    metadata = {}
    if chosen_model_rerun == "bounding_box":
        annotated_image, metadata = run_bbox(threshold_bbox_rerun, file_path, label_position)

    elif chosen_model_rerun == "mask_segmentation":
        annotated_image, metadata = run_mask(threshold_mask_rerun, file_path)
    

    elif chosen_model_rerun == "all_models":
        annotated_image, metadata_bbox = run_bbox(threshold_bbox_rerun, file_path, label_position)
        mask_image, metadata_mask = run_mask(threshold_mask_rerun, file_path)
        combined_image = combine_images(annotated_image, mask_image, align='horizontal')
        annotated_image = combined_image
        metadata = {"roboflow_detections":metadata_bbox}

    if annotated_image:
        annotated_image_path = os.path.join(ANNOTATED_DIR, filename)
        annotated_image.save(annotated_image_path)

    # Initialize ChatSession and pass metadata
    try:
        metadata_json = json.dumps(metadata)
    except Exception as e:
        metadata_json = metadata
        print(f"Error serializing metadata to JSON: {e}")
    print("Passing this metadata to chatbot")
    print(metadata_json)
    chatbot_response = chat_session.run_request(f"Here is the car damage metadata: {metadata_json}")

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
        "selected_model": chosen_model_rerun,
        "chatbot_response": chatbot_response  # Pass chatbot response to the template
    })
