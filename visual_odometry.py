import numpy as np 
import cv2

STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21), 
                #maxLevel = 3,
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
    # shape: [k,2] [k,1] [k,1]
    kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  

    st = st.reshape(st.shape[0])
    kp1 = px_ref[st == 1]
    kp2 = kp2[st == 1]

    return kp1, kp2


def distance(p, prev_p):
    return np.sqrt((p[0] - prev_p[0])**2 + (p[1] - prev_p[1])**2 + (p[2] - prev_p[2])**2)

class PinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, 
                k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]

class VisualOdometry:
    def __init__(self, cam):
        self.frame_stage = 0
        self.cam: PinholeCamera = cam
        self.new_frame = None
        self.last_frame = None
        self.cur_R = None
        self.cur_t = None
        self.px_ref = None
        self.px_old = None
        self.px_cur = None
        self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

    def detect_new_randmarks(self, frame):
        ref = self.detector.detect(frame)
        ref = np.array([x.pt for x in ref], dtype=np.float32)
        return ref

    def processFrame(self):
        # detect first keypoints
        if self.px_ref is None:
            self.px_ref = self.detect_new_randmarks(self.new_frame)
            return 

        # tracking
        self.px_ref, px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
        E, mask = cv2.findEssentialMat(px_cur, self.px_ref,
                                       focal=self.cam.fx, pp=(self.cam.cx, self.cam.cy),
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask = cv2.recoverPose(E, px_cur, self.px_ref,
                                       focal=self.cam.fx, pp=(self.cam.cx, self.cam.cy))
        diff = np.mean(np.abs(self.px_ref - px_cur)[:])
        if self.cur_R is None:
            self.cur_R = R
            self.cur_t = t
        elif (1.5 < diff): # stop position update when optical flow is small
            self.cur_t = self.cur_t + self.cur_R @ t 
            self.cur_R = R @ self.cur_R

        # save keypoints for draw
        self.px_old = self.px_ref.copy()
        self.px_cur = px_cur.copy()

        # Add keypoints if there are only a few existing keypoints
        if (self.px_ref.shape[0] < kMinNumFeature):
            px_cur = self.detect_new_randmarks(self.new_frame)
        self.px_ref = px_cur

    def update(self, img: np.ndarray):
        assert(img.ndim==2 and img.shape[0]==self.cam.height and img.shape[1]==self.cam.width), \
              "Frame: provided image has not the same size as the camera model or image is not grayscale"
        self.new_frame = img
        self.processFrame()
        self.last_frame = self.new_frame
