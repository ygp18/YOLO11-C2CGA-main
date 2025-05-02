import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train/road_mark_11_CGA_lr0.001/weights/best.pt')
    model.val(data='dataset/road_mark/road_mark.yaml',
              split='test',
              imgsz=640,
              batch=32,
              # iou=0.7,
              # rect=False,
              save_json=True,
              project='runs/val',
              name='exp',
              )