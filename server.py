from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse
from typing import List
import os
from detectron2.config import get_cfg
from fastapi.requests import Request
from detectron2.data import MetadataCatalog
from routes import read_root_wrapper, upload_image_wrapper, get_image_wrapper, rerun_inference_wrapper
from config import UPLOAD_DIR, ANNOTATED_DIR, templates, metadata_cardmg
from chatbot.chatbot import ChatSession
from fastapi import HTTPException


# Initialize FastAPI
app = FastAPI()

# Initialize chat session

chat_session = ChatSession()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, image_display: str = '', label_position: str = "TOP_RIGHT", threshold_bbox: float = 0.4, threshold_mask: float = 0.4,
                             file_name: str = '', chosen_model: str = 'bounding_box'):
    return await read_root_wrapper(request, image_display, label_position, threshold_bbox, threshold_mask,
                                     file_name, chosen_model)

# Route to handle image upload
@app.post("/upload")
async def upload_image(request: Request, file: UploadFile = File(...),
                          threshold_mask: float = Form(default=0.4),
                          threshold_bbox: float = Form(default=0.4),
                          label_position: str = Form(default="TOP_RIGHT"),
                          chosen_model: str = Form(default="bounding_box")):
    return await upload_image_wrapper(request, file, threshold_mask, threshold_bbox, label_position, chosen_model, chat_session)

# Route to retrieve images
@app.get("/images/{filename}")
async def get_image(filename: str):
    return await get_image_wrapper(filename)

# Route to rerun inference
@app.post("/rerun")
async def rerun_inference(request: Request, filename: str = Form(...),
                          threshold_mask_rerun: float = Form(default=0.4),
                          threshold_bbox_rerun: float = Form(default=0.4),
                          label_position: str = Form(default="TOP_RIGHT"),
                          chosen_model_rerun: str = Form(default="bounding_box")):
    return await rerun_inference_wrapper(request, filename, threshold_mask_rerun, threshold_bbox_rerun, label_position, chosen_model_rerun, chat_session)

@app.post("/chat")
async def chat_with_bot(request: Request, message: str = Form(...)):
    try:
        response = chat_session.run_request(message)
        return {"response": response}
    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while chatting with the bot.")

