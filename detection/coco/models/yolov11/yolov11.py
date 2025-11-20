import os
from ultralytics import YOLO

def get_yolov11(weights_dir):
    weights_path = os.path.join(weights_dir, 'yolo11n.pt') #TODO update weights
    model = YOLO(weights_path, verbose=False)
    model.eval()
    # disable layer fusing. This is necessary, otherwise the individual layers to inject won't be available during simulation
    model.model.is_fused = lambda: True
    return model