from ultralytics import YOLO
import cv2
import time

model_path = 'runs/detect/train2/weights/best.pt'
video_source = 'VideoProject3.mp4'

DELAY_SECONDS = 0.2

model = YOLO(model_path)

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print(f"Eroare: Nu se poate deschide fisierul '{video_source}'.")
    exit()

print("✅ Video pornit. Apasa tasta 'q' ca sa il opresti.")

current_beam_state = "HIGH"

pending_state = None
last_switch_time = 0

while True:
    success, frame = cap.read()

    if not success:
        print("Sfarsitul videoului.")
        break

    results = model(frame, conf=0.15, verbose=False)
    result = results[0]

    cars_detected = len(result.boxes) > 0

    if cars_detected:
        target_state = "LOW"
    else:
        target_state = "HIGH"


    if target_state != current_beam_state:

        if pending_state != target_state:
            pending_state = target_state
            last_switch_time = time.time()

        elif time.time() - last_switch_time > DELAY_SECONDS:
            current_beam_state = target_state  # Abia acum schimbam oficial
            pending_state = None

    else:
        pending_state = None


    if current_beam_state == "LOW":
        color = (0, 0, 255)
        command_signal = 0
    else:
        color = (0, 255, 0)
        command_signal = 1

    annotated_frame = result.plot()

    cv2.circle(annotated_frame, (50, 50), 30, color, -1)

    text_display = f"HEADLIGHTS: {current_beam_state} BEAM"
    cv2.putText(annotated_frame, text_display, (90, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.putText(annotated_frame, f"ECU Signal: {command_signal}", (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_PLAIN, 1.5, (200, 200, 200), 2)

    if pending_state:
        cv2.putText(annotated_frame, "Stabilizing...", (90, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    cv2.imshow('Adaptive Headlights - Real Time Integration', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()