from mtcnn import MTCNN


class FaceDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_face(self, frame_rgb):
        try:
            faces = self.detector.detect_faces(frame_rgb)
            if faces:
                return faces[0]
            return None
        except Exception as e:
            print("Img err")
            return None
