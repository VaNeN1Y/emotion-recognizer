from deepface import DeepFace


class EmotionRecognizer:
    def __init__(self):
        self.actions = ['emotion'],
        self.enforce_detection = False,
        self.detector_backend = 'skip'

    def recognize(self, face_img):
        try:
            result = DeepFace.analyze(
                face_img,
                actions=self.actions,
                enforce_detection=self.enforce_detection,
                detector_backend=self.detector_backend
            )

            emotion = result[0]['dominant_emotion']
            confidence = result[0]['emotion'][emotion]

            return emotion.capitalize(), confidence

        except Exception as error:
            return "No img", 0.0
