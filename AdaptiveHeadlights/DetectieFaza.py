from ultralytics import YOLO
import cv2
import glob

model_path = 'runs/detect/train2/weights/best.pt'

test_images_path = 'ML_Car_headlight-1/test/imagesByMe/*.jpg'

model = YOLO(model_path)

test_images = glob.glob(test_images_path)[:15]

if not test_images:
    print("Nu s-au gasit poze")
else:
    print(f"S-au incarcat {len(test_images)} imagini")

    for i, img_path in enumerate(test_images):
        frame = cv2.imread(img_path)
        results = model(frame, conf=0.20)
        result = results[0]

        if len(result.boxes) > 0:
            status = "DETECTAT: Masina in fata!"
            action = ">> FAZA SCURTA <<"
            color = (0, 0, 255)
            thickness = 2
        else:
            status = "Drum Liber"
            action = ">> FAZA LUNGA <<"
            color = (0, 255, 0)
            thickness = 2

        annotated_frame = result.plot()

        cv2.putText(annotated_frame, status, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(annotated_frame, action, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)

        cv2.imshow(f'Adaptive Headlights System', annotated_frame)
        print(f"Imaginea {i + 1}: {action}")

        cv2.waitKey(0)

    cv2.destroyAllWindows()