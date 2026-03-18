from ultralytics import YOLO

if __name__ == '__main__':
    num_epochs = 10000
    img_size = 128
    batch_size = 64

    model = YOLO('./yolo11n.pt')
    results = model.train(
        data='coco8.yaml',
        epochs=num_epochs,
        imgsz=img_size,
        device=0,
        batch=batch_size,
        save_period=100,
        project='./training_runs',
        name=f'./trained_{num_epochs}epochs_{img_size}imgsize',
        patience=0,
    )