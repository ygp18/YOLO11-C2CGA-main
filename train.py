import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/11/yolo11-C2CGA.yaml')
    # model = YOLO('runs/train/exp/weights/last.pt')
    model.train(data='dataset/brain_tumor_02/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=32,
                close_mosaic=0,
                workers=4,
                optimizer='SGD',
                patience=0,
                resume=True,
                project='runs/train',
                name='exp',
                )
