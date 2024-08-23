from utils import get_position_from_string, get_uploaded_files, load_api_keys
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
from PIL import Image
import supervision as sv

api_keys = load_api_keys('keys.json')

roboflow_api_key = api_keys.get('roboflow_api_key')

def run_bbox(threshold_bbox: float, file_path: str, label_position: str):
    # Initialize inference client
    CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=roboflow_api_key
    )
    custom_configuration = InferenceConfiguration(confidence_threshold=threshold_bbox)
    CLIENT.configure(custom_configuration)
    results = CLIENT.infer(file_path, model_id="cardamage-l4vtd/1")
    image = Image.open(file_path)
    detections = sv.Detections.from_inference(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=get_position_from_string(label_position))
    
    annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)
    return annotated_image, detections
