from ultralytics import YOLO
import cv2
import glob


model_path = 'runs/detect/train2/weights/best.pt'

model = YOLO(model_path)

test_images = glob.glob('ML_Car_headlight-1/test/images/*.jpg')

images_to_show = test_images[:20 ]

if not images_to_show:
    print("Nu s-au gasit imagini!")
else:
    print(f"S-au incarcat {len(test_images)} imagini")

    results = model(images_to_show)

    for i, result in enumerate(results):
        img_with_boxes = result.plot()

        cv2.imshow(f'Detectie {i + 1} (Apasa SPACE)', img_with_boxes)

        print(f"🖼️  Afisez poza {i + 1}. Apasa tasta SPACE (sau Enter) pe poza pentru a trece mai departe...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("Test finalizat!")