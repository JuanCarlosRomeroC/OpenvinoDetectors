from openvino_detectors.openvino_detectors import OpenvinoDetector
import numpy as np

class OpenvinoMetadata(OpenvinoDetector):
    def __init__(self, detection_threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/car_metadata/vehicle-attributes-recognition-barrier-0039.xml",
                         detection_threshold=detection_threshold)
        self.types = ["car", "bus", "truck", "van"]

    def detect(self, frame):
        cur, _, _ = self.get_detections(frame)
        detections = cur['type'][0]
        ind = np.argmax(detections)

        return self.types[ind]
