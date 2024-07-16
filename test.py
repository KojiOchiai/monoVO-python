from dataclasses import dataclass
import numpy as np 
import cv2

from visual_odometry import PinholeCamera, VisualOdometry
from kitti_odometrory import KittaOdometory


@dataclass
class Point:
	x: float
	y: float

def draw_color(img_id):
	return (img_id*255/4540, 255-img_id*255/4540, 0)

KITTI_DATASET = '/Users/user01/projects/datasets/KITTI_grayscale/'
cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)
vo = VisualOdometry(cam)
ko = KittaOdometory(KITTI_DATASET + 'KITTI_odometry_poses/00.txt')
draw_center = Point(290, 90)

traj = np.zeros((600,600,3), dtype=np.uint8)

for img_id in range(4541):
	# estimated position
	img = cv2.imread(KITTI_DATASET + 'KITTI_odometry_gray/00/image_0/'+str(img_id).zfill(6)+'.png', 0)
	vo.update(img)
	cur_t = vo.cur_t
	if(img_id > 2):
		k = 0.8
		x, y, z = k * cur_t[0], k * cur_t[1], k * cur_t[2]
	else:
		x, y, z = 0., 0., 0.
	draw_x, draw_y = int(x+draw_center.x), int(z+draw_center.y)

	# true position
	trueX, trueY, trueZ = ko.getXYZ(img_id)
	true_x, true_y = int(trueX+draw_center.x), int(trueZ+draw_center.y)

	# draw
	cv2.circle(traj, (draw_x, draw_y), 1, draw_color(img_id), 1)
	cv2.circle(traj, (true_x, true_y), 1, (0,0,255), 2)
	cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
	text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
	cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

	cv2.imshow('Road facing camera', img)
	cv2.imshow('Trajectory', traj)
	cv2.waitKey(1)

cv2.imwrite('map.png', traj)
