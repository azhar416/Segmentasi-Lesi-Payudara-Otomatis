from ultralytics import YOLO

def load_yolo(weight_path):
    """
    Load the YOLO model with the specified weight path.

    Args:
        weight_path (str): The path to the YOLO weight file.
    
    Returns:
        YOLO: The loaded YOLO model.
    """
    return YOLO(weight_path)