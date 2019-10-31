from scipy import spatial

from openvino_detectors.openvino_detectors import OpenvinoDetector

INC_SIZE = 0


class OpenvinoReidentification(OpenvinoDetector):
    def __init__(self, threshold):
        super().__init__(cpu_lib="openvino_detectors/lib/libcpu_extension_sse4.so",
                         detector_xml="openvino_detectors/models/reidentification/person-reidentification-retail-0079.xml",
                         detection_threshold=None)
        self.threshold = threshold

    def calc_dist(self, person1, person2):
        #TODO add reid
        p1 = self.get_detections(person1)[self.d_out][0]
        p2 = self.get_detections(person2)[self.d_out][0]
        return self.similarity(p1, p2)

    def compare_tracks(self, track1_img, track2_img):
        min_d = 100

        for i, img1 in enumerate(track1_img[0:10]):
            for j, img2 in enumerate(track2_img[-10:]):
                d = self.calc_dist(img1, img2)
                if d < min_d:
                    min_d = d
        return min_d

    def similarity(self, v1, v2):
        return 1.0 - spatial.distance.cosine(v1, v2)