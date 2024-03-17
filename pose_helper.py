from enum import Enum
from typing import NamedTuple, Tuple, List
import math
import torch
from icecream import ic

# Mapping from index to body part
class BodyPart(Enum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


#  BodyPart are the index map index to body part name as string
def get_body_part(index):
    return BodyPart(index).name


# get only right body parts
def is_interesting_body_part(index):
    # nose is false
    if index == 0:
        return False
    # skip eye
    if index == 1 or index == 2:
        return False
    return index % 2 == 0


Bone = NamedTuple("Bone", [("bottom", torch.tensor), ("top", torch.tensor)])


class Body:
    # take a [(x,y)],[conf] as input
    def __init__(self, keypoints: List[[int, int]], conf):
        self.keypoints = keypoints
        self.conf = conf

    def make_bone(self, bottom: BodyPart, top: BodyPart) -> Bone:
        return Bone(self.get_part(bottom), self.get_part(top))

    def get_part(self, part: BodyPart):
        return self.keypoints[part.value]

    def spine(self) -> Bone:
        return self.make_bone(BodyPart.RIGHT_HIP, BodyPart.RIGHT_EAR)

    def r_upper_arm(self) -> Bone:
        return self.make_bone(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW)

    def r_arm_lower(self) -> Bone:
        return self.make_bone(BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST)

    def r_leg_upper(self) -> Bone:
        return self.make_bone(BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE)

    def r_leg_lower(self) -> Bone:
        return self.make_bone(BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE)

    def hip_angle(self):
        return make_angle(self.r_leg_upper(), self.spine())

    def knee_angle(self):
        return make_angle(self.r_leg_upper(), self.r_leg_lower())

    def armpit_angle(self):
        return make_angle(self.spine(), self.r_leg_upper())

    def spine_vertical(self):
        return bone_to_vertical(self.spine())


def make_angle(bone1: Bone, bone2: Bone) -> float:

    hold_bone = bone1

    if torch.equal(bone1.top, bone2.top):
        # if passed in wrong ends swap them
        hold_bone = Bone(bone2.top, bone2.bottom)
    else:
        assert torch.equal(
            bone1.bottom, hold_bone.bottom
        ), "Both bones should have same bottom"

    x1, y1 = bone1.bottom
    x2, y2 = bone1.top
    x3, y3 = hold_bone.top
    # I need to add both angles
    angle_1 = math.atan2(y2 - y1, x2 - x1)
    angle2 = math.atan2(y3 - y2, x3 - x2)
    return int(math.degrees(angle_1 + angle2))


def bone_to_horizontal(bone1: Bone):
    x1, y1 = bone1.bottom
    x2, y2 = bone1.top
    return abs(int(math.degrees(math.atan2(y2 - y1, x2 - x1))))


def bone_to_vertical(bone1: Bone):
    x1, y1 = bone1.bottom
    x2, y2 = bone1.top
    return abs(int(math.degrees(math.atan2(x2 - x1, y2 - y1))))


def add_pose(results, im):
    import cv_helper
    import cv2

    # self
    # get highest confidence person

    kp = results[0].keypoints  # noqa
    keypoints = kp.xyn[0]
    keypoints_pixel = kp.xy[0]
    confidence = kp.conf[0]
    # ic(keypoints, confidence)
    font_scale = 0.5  # should be dynamic based on image size
    b = Body(keypoints, confidence)
    ic(1)
    ic(b.spine(), b.r_leg_upper())
    ic(b.hip_angle(), b.knee_angle(), b.armpit_angle())
    im = cv_helper.write_text(
        im,
        f"hip:{b.hip_angle()}\narmpit:{b.armpit_angle()}\nback:{b.spine_vertical()}",
        (50, 200),
        1,
    )
    for i, conf in enumerate(confidence):
        if conf > 0.5 and is_interesting_body_part(i):
            x, y = keypoints_pixel[i]
            origin = (int(x), int(y))
            cv2.circle(im, origin, 5, (0, 0, 255), -1)
            cv_helper.write_text(
                im,
                get_body_part(i),
                origin,
                font_scale,
            )
    return im
