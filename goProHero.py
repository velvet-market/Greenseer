from goprocam import GoProCamera, constants
import os
import cv2

go_pro = GoProCamera.GoPro()
go_pro.mode(mode="0")
go_pro.video_settings(res="1080p", fps="30")


def reset_camera():
    go_pro.delete("all")



def video_stream(interval):
  #  while True:
    go_pro.shoot_video(duration=interval)
    #os.chdir(os.getcwd())
    go_pro.downloadLastMedia(custom_filename="data/gopro/cam_video.MP4")
    vidcap = cv2.VideoCapture("data/gopro/cam_video.MP4")
    success, image = vidcap.read()
    count = 0
    while success:
        image = cv2.resize(image, [512, 512])
        cv2.imwrite("data/gopro/frame%d.jpg" % count, image)
        success, image = vidcap.read()
        count += 1
    go_pro.delete("last")



# go_pro.video_settings(res="4k",fps="120")

# go_pro.shoot_video(duration=30)
# go_pro.downloadLastMedia(custom_filename="video")

# def media_download_transfer_delete():
# media = go_pro.downloadAll()
# for i in media:
#	shutil.move('./100GOPRO-{}'.format(i), './images/{}'.format(i))
# go_pro.delete("all")

reset_camera()
video_stream(5)
