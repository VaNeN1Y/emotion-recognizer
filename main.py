import cv2
from detector.face import FaceDetector
from detector.emotion import EmotionRecognizer


def main():
    cap = cv2.VideoCapture(0)

    face_detector = FaceDetector()
    emotion_recognizer = EmotionRecognizer()

    while True:
        cam, frame = cap.read()

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_data = face_detector.detect_face(frame_rgb)

        if face_data:
            box = face_data["box"]
            x, y, w, h = box
            # print(box)

            clear_face = frame_rgb[y:y + h, x:x + w]

            emotion, probability = emotion_recognizer.recognize(clear_face)

            print(emotion, "-", probability)

        cv2.imshow("cam", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
