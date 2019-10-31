from openvino_detectors.openvino_detectors import OpenvinoDetector

"""
29 FPS pedestrian binary
43.7 FPS vehicle-adas-binary
"""
class OpenvinoPedestrian(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/vehicle-adas-binary/vehicle-detection-adas-binary-0001.xml",
                         detection_threshold=detection_threshold)

    def detect(self, frame):
        height, width = frame.shape[:-1]
        cur, new_height, new_width = self.get_detections(frame)
        detections = cur[self.d_out][0][0]
        result = []
        for _, _, confidence, left, top, right, bottom in detections:
            if confidence > self.detection_threshold:

                left, top, right, bottom = self.convert_detections((left, top, right, bottom), height, width,
                                        new_height, new_width)
                result.append((left, top, right, bottom, float(confidence)))
        return result