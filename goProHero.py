import time
import os
import cv2
import numpy as np
from goprocam import GoProCamera, constants
from random import randrange
from constants import *

def reset_camera(go_pro):
    go_pro.delete("all")

def rand_crop(img):
	x, y, _ = img.shape
	x1 = randrange(0, x - CROP_SIZE)
	y1 = randrange(0, y - CROP_SIZE)
	img = img[x1:x1+CROP_SIZE, y1:y1+CROP_SIZE, :]

	return img

def video_labels(go_pro, category):
	go_pro.shoot_video(duration=60)
	custom_file = f"{SAVE_FILE}/label-videos/{category}.MP4"
	go_pro.downloadLastMedia(custom_filename=custom_file)

	vidcap = cv2.VideoCapture(custom_file)
	frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
	indices = np.asarray(np.round(np.linspace(0, len(frame_count) - 1, NUM_LABELS)).astype(int))
	training_indices = indices[:int(NUM_LABELS*.8)]
	testing_indices = indices[int(NUM_LABELS*.8):]

	i = 0
	success, frame = vidcap.read()

	while success:
		if i in training_indices:
			cv2.imwrite(f"{SAVE_FILE}/training/{category}/{i}.jpg", frame)

		if i in testing_indices:
			cv2.imwrite(f"{SAVE_FILE}/testing/{category}/{i}.jpg", frame)

		i += 1

def video_stream(go_pro):
	timestr = time.strftime("%Y%m%d-%H%M%S")
	go_pro.shoot_video(duration=SECONDS_BUFFER)
	go_pro.downloadLastMedia(custom_filename=f"{SAVE_FILE}/gopro/cam_video.MP4")
	vidcap = cv2.VideoCapture(f"{SAVE_FILE}/gopro/cam_video.MP4")

	frames = []
	success, frame = vidcap.read()
	frames.append(frame)
	count = 0

	while success:
		frame = rand_crop(frame)
		frames.append(frame)

		if SAVE:
			cv2.imwrite(f"{SAVE_FILE}/gopro/{timestr}_frame{count}.jpg", frame)

		success, frame = vidcap.read()
		count += 1
	go_pro.delete("last")

	return frames

# def media_download_transfer_delete():
	# media = go_pro.downloadAll()
	# for i in media:
	#	shutil.move('./100GOPRO-{}'.format(i), './images/{}'.format(i))
	# go_pro.delete("all")