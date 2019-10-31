from classifiers.OpenvinoReidClassifier import OpenvinoReidentificationPlayer
from openvino_detectors.openvino_detectors import OpenvinoDetector

"""
45 FPS
"""
class OpenvinoPersonDetector(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml",
                         # detector_xml="openvino_detectors/models/person-vehicle-bike-detection-crossroad-0078/person-vehicle-bike-detection-crossroad-0078.xml",
                         detection_threshold=detection_threshold)
        # self.classifier = ColorClassifer()
        self.classifier = OpenvinoReidentificationPlayer("train_uniform")

    def detect(self, frame):
        height, width = frame.shape[:-1]
        cur, new_height, new_width = self.get_detections(frame)
        detections = cur[self.d_out][0][0]
        result = []
        types = []
        for _, label, confidence, left, top, right, bottom in detections:
            if label == 1:
                if confidence > self.detection_threshold:
                    left, top, right, bottom = self.convert_detections((left, top, right, bottom), height, width,
                                                                       new_height, new_width)
                    # try:
                    #     current_class = self.classifier.predict(frame[top:bottom, left:right])
                    #     types.append(current_class)
                    result.append((left, top, right, bottom, float(confidence)))
                    # except Exception as e:
                    #     print("Error: small detect")

        return result
