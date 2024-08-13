import os
from fastapi.templating import Jinja2Templates
from detectron2.data import MetadataCatalog

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
    json_file='model/coco_annotations_damage.json',
    image_root=UPLOAD_DIR,
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
