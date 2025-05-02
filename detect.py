import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO('runs/train/exp/weights/best.pt') # select your model.pt path
    model.predict(source='dataset/brain_tumor_02/valid/images/1-1_png.rf.4f8a8acebd1fed547f2ccdef810e39b1.jpg',
                  imgsz=640,
                  project='runs/detect',
                  name='eps',
                  save=True,
                  # conf=0.2,
                  # iou=0.7,
                  # agnostic_nms=True,
                  # visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )