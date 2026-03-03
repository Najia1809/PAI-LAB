from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Initialize HOG + SVM People Detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Start webcam
cap = cv2.VideoCapture(0)

# Stability variables
stable_count = 0
frame_counter = 0

def generate_frames():
    global stable_count, frame_counter

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize for better performance
        frame = cv2.resize(frame, (640, 480))

        # Improve lighting slightly (helps in dark rooms)
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

        # Detect people
        (rects, weights) = hog.detectMultiScale(
            frame,
            winStride=(4, 4),
            padding=(8, 8),
            scale=1.05
        )

        boxes = []
        confidences = []

        # Filter weak detections
        for i, (x, y, w, h) in enumerate(rects):
            if weights[i] > 0.6:
                boxes.append([x, y, x + w, y + h])
                confidences.append(float(weights[i]))

        # Apply Non-Maximum Suppression
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            score_threshold=0.6,
            nms_threshold=0.3
        )

        person_count = 0

        if len(indices) > 0:
            for i in indices.flatten():
                (x1, y1, x2, y2) = boxes[i]
                person_count += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Stability logic (update count every 8 frames)
        frame_counter += 1
        if frame_counter % 8 == 0:
            stable_count = person_count

        # Display count
        cv2.putText(
            frame,
            f'People Count: {stable_count}',
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)