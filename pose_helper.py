from enum import Enum
from typing import NamedTuple, List
import math
import torch


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

    def neck(self) -> Bone:
        return self.make_bone(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_EAR)

    def spine(self) -> Bone:
        return self.make_bone(BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER)

    def r_upper_arm(self) -> Bone:
        return self.make_bone(BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW)

    def r_total_arm(self) -> Bone:
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

    def neck_to_head(self):
        neck = bone_to_vertical(self.neck())
        spine = bone_to_vertical(self.spine())
        return int(neck - spine)


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


def bone_to_horizontal(bone: Bone):
    x1, y1 = bone.bottom
    x2, y2 = bone.top
    angle_in_radians = math.atan2(y2 - y1, x2 - x1)  # Swap x and y
    return int(abs(math.degrees(angle_in_radians)))


def bone_to_vertical(bone: Bone):
    return abs(int(90 - bone_to_horizontal(bone)))


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
    stats = f""" Hip:{b.hip_angle()}
 LowerLeg:{bone_to_vertical(b.r_leg_lower())}
 Back:{b.spine_vertical()}
 Neck:{b.neck_to_head()}
 Arm:{bone_to_horizontal(b.r_total_arm())}"""

    im = cv_helper.write_text(
        im,
        stats,
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
    # Draw bones
    bones = [
        b.neck(),
        b.spine(),
        b.r_upper_arm(),
        b.r_arm_lower(),
        b.r_leg_upper(),
        b.r_leg_lower(),
    ]
    # pick some nice colors,
    blue, red, green, purple, yellow = (
        (255, 0, 0),
        (0, 0, 255),
        (0, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
    )
    colors = [blue, red, green, purple, yellow, blue]
    for color, bone in zip(colors * 2, bones):
        # give a different color to each bone, handle more bones then colors
        # bone is normalized 0->1, scale to image resolution
        im = cv2.line(
            im,
            cv_helper.scale_point_to_image(im, bone.bottom),
            cv_helper.scale_point_to_image(im, bone.top),
            color,
            2,
        )

    return im
