import os
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():
    data_yaml = os.environ.get("YOLO_DATA", "./data/cleavage.yaml")
    base_weights = os.environ.get("YOLO_BASE_WEIGHTS", "yolo11s.pt")
    device = os.environ.get("YOLO_DEVICE", "0")
    project = os.environ.get("YOLO_PROJECT", "runs/detect")
    name = os.environ.get("YOLO_RUN_NAME", "embryo_yolo11")

    model = YOLO(base_weights)
    model.train(data=data_yaml, imgsz=800, epochs=200, batch=16, device=device, project=project, name=name, patience=30, workers=4, seed=0, optimizer="SGD", lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, cos_lr=True, box=7.5, cls=0.5, dfl=1.5, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, scale=0.5, fliplr=0.5, mosaic=1.0, mixup=0.0, copy_paste=0.0)
    model.val(data=data_yaml, imgsz=800, device=device, split="val", iou=0.6, conf=0.01, max_det=300)


if __name__ == "__main__":
    main()
