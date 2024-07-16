from dataclasses import dataclass


@dataclass
class KittaOdometory:
    annotations: str
    trueX: float = 0.0
    trueY: float = 0.0
    trueZ: float = 0.0

    def __post_init__(self):
        with open(self.annotations) as f:
            self.annotations = f.readlines()

    def getXYZ(self, frame_id):  #specialized for KITTI odometry dataset
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return x, y, z