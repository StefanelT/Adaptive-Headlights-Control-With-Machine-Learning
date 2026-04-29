from ultralytics import YOLO
model = YOLO('yolov8s.pt')
if __name__ == '__main__':
    print("Incepe antrenamentul")
    model.train(
        data='ML_Car_headlight-1/data.yaml',
        epochs=80,
        patience=20,
        imgsz=640,
        device=0,
        workers=0,
        batch=8
    )
    print("Antrenamentul s-a terminat cu succes.")





