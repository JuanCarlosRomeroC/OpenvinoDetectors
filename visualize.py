import json

import cv2
import numpy as np

from mapping import get_down_middle_point

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
INSIDE_CONTOUR = 1

COLORS = 3000

colours = np.random.randint(0, 255, COLORS)


def read_contour(contour_json):
    if contour_json is not None:
        cur = []
        for point in contour_json:
            cur.append((point["x"], point["y"]))
        return cur
    return None


def drawline(img, pt1, pt2, color, thickness=1, style='dotted', gap=10):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
        p = (x, y)
        pts.append(p)

    if style == 'dotted':
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        if len(pts) > 0:
            e = pts[0]
            i = 0
            for p in pts:
                s = e
                e = p
                if i % 2 == 1:
                    # cv2.line(img, s, e, color, thickness)
                    cv2.line(img, s, e, 50, thickness)
                i += 1


def drawpoly(img, pts, color=(0, 255, 0), thickness=5, style='dotted'):
    e = pts[0]
    pts.append(pts.pop(0))
    for p in pts:
        s = e
        e = p
        img = cv2.line(img, s, e, 50, thickness)
    return img


def drawrect(img, pt1, pt2, color, track_id, thickness=1, style='dotted'):
    if style == 'solid':
        cv2.rectangle(img, pt1, pt2, (0, 250, 0), thickness)
        cv2.putText(img, str(track_id), pt2, cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), thickness=4)
    else:
        pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        drawpoly(img, pts, color, thickness, style)


def draw_boxes_on_image(img, boxes):
    # left, top, right, bottom, track, phantom
    for box in boxes:
        left, top, right, bottom, track_id, phantom, _ = box
        # color = (colours[(3 * track_id) % COLORS],
        #          colours[(3 * track_id + 1) % COLORS],
        #          colours[(3 * track_id + 2) % COLORS])
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))

        if phantom == 1:
            style = 'dashed'
            continue
        else:
            style = 'solid'

        color = 50

        drawrect(img, p1, p2, color, track_id, 3, style)



def draw_only_detections(img, boxes):
    # left, top, right, bottom, track, phantom
    track_id = 10
    for box in boxes:
        left, top, right, bottom = box
        color = (colours[(3 * track_id) % COLORS],
                 colours[(3 * track_id + 1) % COLORS],
                 colours[(3 * track_id + 2) % COLORS])
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))

        style = 'solid'

        color = 50

        drawrect(img, p1, p2, color, track_id, 3, style)

def draw_only_detections_rcnn(img, boxes):
    # left, top, right, bottom, track, phantom
    track_id = 10
    for box in boxes:
        left, top, right, bottom,_ = box
        color = (colours[(3 * track_id) % COLORS],
                 colours[(3 * track_id + 1) % COLORS],
                 colours[(3 * track_id + 2) % COLORS])
        p1 = (int(left), int(top))
        p2 = (int(right), int(bottom))

        style = 'solid'

        color = 50

        drawrect(img, p1, p2, color, track_id, 3, style)

def draw_track(img, boxes):
    # left, top, right, bottom, track, phantom
    for box in boxes:
        left, top, right, bottom, track_id, phantom = box
        color = (colours[(3 * track_id) % COLORS],
                 colours[(3 * track_id + 1) % COLORS],
                 colours[(3 * track_id + 2) % COLORS])
        point_legs = get_down_middle_point(box)

        if phantom == 1:
            style = 'dashed'
            continue
        else:
            style = 'solid'

        color = 50

        img = cv2.circle(img, point_legs, 15, (0, 0, 255), -1)
    return img


def draw_text_on_image(img, text, bottom_left, font_face=cv2.FONT_HERSHEY_TRIPLEX,
                       font_scale=2, color=SCALAR_WHITE, font_thickness=2):
    cv2.putText(img, str(text), bottom_left, font_face, font_scale, color, font_thickness)


def visualize(source_path, output_path, recognitions_path, config_path):
    print("Visualizing " + source_path)
    video = cv2.VideoCapture(source_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = int(video.get(cv2.CAP_PROP_FPS))
    print(fps)
    new_video = cv2.VideoWriter(output_path, fourcc=fourcc, fps=fps, apiPreference=0,
                                frameSize=(int(1920 / 4), int(1080 / 4)))
    print("Saving to " + output_path)
    print("Reading " + str(recognitions_path))
    config_json = json.load(open(config_path))
    data = json.load(open(recognitions_path))
    print("After Reading " + str(recognitions_path))
    boxes = data["colored-boxes"]
    class_frames = data["class-frames"]
    in_frames = class_frames["in"]
    # out_frames = class_frames["out"]
    print("Len boxes = " + str(len(boxes)))
    contour = read_contour(config_json["contour"])
    contour_table = read_contour(config_json["contour_table"])

    for i, info_frame in enumerate(boxes):
        _, img = video.read()
        # if i==100:
        #     break
        # if not only_legs:
        draw_boxes_on_image(img, info_frame)

        img = drawpoly(img, contour)
        # img = drawpoly(img, contour_table)
        draw_text_on_image(img, "In: " + str(in_frames[i][0]), (200, 200), color=SCALAR_RED)
        # draw_text_on_image(img, "Out: " + str(out_frames[i][0]), (200, 300), color=SCALAR_RED)
        # draw_only_detections(img, info_frame)
        # else:
        #     img = draw_track(img, info_frame)

        img = cv2.resize(img, (int(1920 / 4), int(1080 / 4)))
        # crop_img = img[0:880, 0:1920]
        new_video.write(img)
        if i % 1000 == 0:
            print(str(i) + " of " + str(len(boxes)))
    new_video.release()

# if __name__ == "__main__":
#     source_path = "/home/sergej2/Desktop/test_recognition.mp4"
#     output_path = "visualize_video/vis_test.avi"
#     tracker_config = json.load(open("detections/tracker_config.json", "r"))
#     camera_config = {
#         "contour": [{"y": 660, "x": 491}, {"y": 583, "x": 834}, {"y": 722, "x": 943}, {"y": 847, "x": 456}],
#         "entering": [{"y": 672, "x": 713}, {"y": 803, "x": 768}, {"y": 672, "x": 713}],
#         "debug": True}
#     visualize(source_path, output_path, "result.txt")

#
