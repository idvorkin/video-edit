from  enum  import Enum, auto

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
def is_right_body_part(index):
    return index % 2 == 0

#
class Body():
    # take a [(x,y)],[conf] as input
    def __init__(self, keypoints, conf):
        self.keypoints = keypoints
        self.conf = conf
    def make_bone(self, bottom:BodyPart, top:BodyPart):
        # return a line
        line = (self.keypoints[bottom] , self.keypoints[top])
        return line

    def spine(self):
        return self.make_bone(BodyPart.RIGHT_HIP, BodyPart.RIGHT_EAR)

    def vertical_diff(self, bone):
        # compute angle between bone and a vertical line
        return

