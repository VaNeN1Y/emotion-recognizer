import cv2
from detector.face import FaceDetector
from detector.emotion import EmotionRecognizer
from collections import deque, Counter


def main():
    cap = cv2.VideoCapture(0)

    face_detector = FaceDetector()
    emotion_recognizer = EmotionRecognizer()

    emotion_arr = deque(maxlen=7)

    while True:
        cam, frame = cap.read()
        if not cam:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_data = face_detector.detect_face(frame_rgb)

        if face_data:
            box = face_data["box"]
            x, y, w, h = box
            # print(box)

            clear_face = frame_rgb[y:y + h, x:x + w]
            clear_face = cv2.resize(clear_face, (224, 224))

            emotion, probability = emotion_recognizer.recognize(clear_face)

            if emotion != "No img" and probability != 0.0:
                emotion_arr.append((emotion, probability))
                emo_only = [e for e, p in emotion_arr]
                most_popular_emo = Counter(emo_only).most_common(1)[0][0]

                prob_only = [p for e, p in emotion_arr if e == most_popular_emo]
                emo_probability = prob_only[-1]
                cv2.putText(frame_rgb, f"{most_popular_emo} - {emo_probability:.2f}", (30, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (255, 0, 255), thickness=2, lineType=8,
                          shift=0)

            # print(emotion, "-", probability)

        cv2.imshow("camera", frame_rgb)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
