import matplotlib.pyplot as plt

def test_img(model, loader):
    for img, target in loader: break
    print(f'Target labels: {target[0]['labels']}')
    plt.imshow(img[0])
    plt.savefig('image.jpg')

    width, _ = img[0].size
    result = model.predict(img, imgsz=width)
    print(f'Predicted labels: {result[0].boxes.cls}')
    annotated_image = result[0].plot()
    plt.imshow(annotated_image)
    plt.savefig('annotated.jpg')