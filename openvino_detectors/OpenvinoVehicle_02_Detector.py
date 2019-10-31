from openvino_detectors.openvino_detectors import OpenvinoDetector

"""
Count frame 4770
Time 176
FPS 27
"""
class OpenvinoVehicle_Adas_02_Detector(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/vehicle-adas-0002/vehicle-detection-adas-0002.xml",
                         # detector_xml="openvino_detectors/models/person-vehicle-bike-detection-crossroad-0078/person-vehicle-bike-detection-crossroad-0078.xml",
                         detection_threshold=detection_threshold)

    def detect(self, frame):
        height, width = frame.shape[:-1]
        cur, new_height, new_width = self.get_detections(frame)
        detections = cur[self.d_out][0][0]
        result = []
        for _, label, confidence, left, top, right, bottom in detections:
            if label == 1:
                if confidence > self.detection_threshold:
                    left, top, right, bottom = self.convert_detections((left, top, right, bottom), height, width,
                                                                       new_height, new_width)
                    result.append((left, top, right, bottom, float(confidence)))
        return result
