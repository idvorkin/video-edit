from enum import Enum
from typing import NamedTuple
import math
import torch
import ultralytics


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


class SwingRepCounter:
    def __init__(self):
        self.HINGE = 0
        self.STRAIT = 1
        self.rep = 0
        self.state = self.HINGE
        self.transition_counter = 0
        # Threshold for confirming a state transition
        self.transition_threshold = 5  # Adjust based on your needs

    def frame(self, is_hinge: bool):
        if is_hinge:
            if self.state == self.STRAIT:
                # Increment transition counter if there's a potential transition
                self.transition_counter += 1
                # Check if the transition is confirmed (exceeded the threshold)
                if self.transition_counter >= self.transition_threshold:
                    self.rep += 1
                    self.state = self.HINGE
                    # Reset the counter after confirming the transition
                    self.transition_counter = 0
            else:
                # Reset the counter if it's a false alarm
                self.transition_counter = 0
        else:
            if self.state == self.HINGE:
                # Increment transition counter if there's a potential transition
                self.transition_counter += 1
                # Transition to STRAIT if confirmed
                if self.transition_counter >= self.transition_threshold:
                    self.state = self.STRAIT
                    # Reset the counter after confirming the transition
                    self.transition_counter = 0
            else:
                # Reset the counter if it returns to STRAIT without a full transition
                self.transition_counter = 0


class Body:
    # take a [(x,y)],[conf] as input
    def __init__(self, keypoints: ultralytics.engine.results.Keypoints):
        self.predicted_keypoints = keypoints
        self.keypoints = keypoints.xyn[0]
        self.conf = keypoints.conf[0]

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
    if torch.equal(bone1.top, bone2.top):
        # if passed in wrong ends swap them
        bone2 = Bone(bone2.top, bone2.bottom)
    else:
        assert torch.equal(
            bone1.bottom, bone2.bottom
        ), "Both bones should have same bottom"

    touch = bone1.bottom
    top_end = bone1.top
    bottom_end = bone2.top

    # I need to add both angles
    top_bone_angle = math.atan2(top_end[1] - touch[1], top_end[0] - touch[0])
    bottom_bone_angle = math.atan2(bottom_end[1] - touch[1], bottom_end[0] - top_end[0])
    return int(math.degrees(abs(top_bone_angle) + abs(bottom_bone_angle)))


def bone_to_horizontal(bone: Bone):
    x1, y1 = bone.bottom
    x2, y2 = bone.top
    angle_in_radians = math.atan2(y2 - y1, x2 - x1)  # Swap x and y
    return int(abs(math.degrees(angle_in_radians)))


def bone_to_vertical(bone: Bone):
    return abs(int(90 - bone_to_horizontal(bone)))


def add_pose(keypoints: ultralytics.engine.results.Keypoints, frame, rep, im, label):
    import cv_helper
    import cv2

    # self
    # get highest confidence person

    # validate keypoints

    keypoints_xyn = keypoints.xyn[0]
    keypoints_pixel = keypoints.xy[0]
    confidence = keypoints.conf[0]
    # ic(keypoints, confidence)
    font_scale = 0.5  # should be dynamic based on image size
    b = Body(keypoints)
    stats = f""" REP: {rep}
 Hip:{b.hip_angle()}
 LowerLeg:{bone_to_vertical(b.r_leg_lower())}
 Back:{b.spine_vertical()}
 Neck:{b.neck_to_head()}
 Arm:{bone_to_horizontal(b.r_total_arm())}
 Frame:{frame}
 {label}
 """

    # if person is on the left, draw the box on the right
    is_on_right = keypoints_xyn[BodyPart.RIGHT_HIP.value][0] > 0.5
    box_top_left = [0, 200]
    box_top_left[0] = 50 if is_on_right else im.shape[1] - 400
    # ic(is_on_right, box_top_left,im.shape)

    im = cv_helper.write_text(
        im,
        stats,
        box_top_left,
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
    colors = [blue, red, green, purple, yellow, blue, red, green, purple, yellow, blue]
    for i, bone in enumerate(bones):
        color = colors[i]
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
