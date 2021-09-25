from goprocam import GoProCamera, constants
import shutil

go_pro = GoProCamera.GoPro()
go_pro.video_settings(res="4k",fps="60")

#go_pro.listMedia(True)

def take_photo_transfer_delete():
	go_pro.take_photo(timer=5)
	go_pro.downloadLastMedia(custom_filename="rock.png")
	go_pro.delete("last")

def video_stream(interval):
	while True:
		media = go_pro.downloadLastMedia(go_pro.shoot_video(duration=interval))
		go_pro.delete("all")
		print(media)

#go_pro.video_settings(res="4k",fps="120")

#go_pro.shoot_video(duration=30)
#go_pro.downloadLastMedia(custom_filename="video")

#def media_download_transfer_delete():
	#media = go_pro.downloadAll()
	#for i in media:
	#	shutil.move('./100GOPRO-{}'.format(i), './images/{}'.format(i))
	#go_pro.delete("all")

video_stream()
