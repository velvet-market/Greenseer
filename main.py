"""
MIT License

Copyright (c) 2019 Isaiah Nields

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import cv2
import time
import schedule
import time
import glob
import os
from keras.models import load_model
from constants import *
from goProHero import *
import winsound

def predict(img):
    # if the size of the image is too small, report none
    if img.size < 100:
        return -1, 'none'

    # predict the image class
    pred = model.predict(img.reshape(1, *img.shape), batch_size=1).flatten()
    indices = pred.argsort()[::-1]

    score_dict = {}
    for i in indices.flatten():
        score_dict[CATEGORY[i]] = pred[i]

    return score_dict

def check_background(frames):
    kept_frames = []
    for index, frame in enumerate(frames):
        img = frame 
        bg_score = get_background_score(img)
        if PRINT:
            print(f"{index}: {bg_score}")
        
        if bg_score > CUTOFF_SCORE:
            if SAVE:
                cv2.imwrite(f"{SAVE_FILE}/input/{index}.jpg", img)
            kept_frames.append(frame)
    return kept_frames   

def get_background_score(img):
    denom = img.shape[0] * img.shape[1] * img.shape[2]
    mask = fgbg.apply(img) / 255
    return np.sum(mask) / denom

def clean_images():
    files = glob.glob(f"{SAVE_FILE}/input/*.jpg") + glob.glob(f"{SAVE_FILE}/gopro/*.jpg")
    for file in files:
        os.remove(file)

def run_ml(frames):
    files = np.asarray(check_background(frames))

    if len(files) > MIN_FILES:
        indices = np.asarray(np.round(np.linspace(0, len(files) - 1, MIN_FILES)).astype(int))
        files = files[indices]
    else:
        return

    # ====================================
    t3 = time.time()
    total_scores = {}
    for index, file in enumerate(files):
        img = file
        # cv2.imshow('img', img)

        score_dict = predict(img)
        if PRINT:
            print(f'ML MODEL {index}: {score_dict}')

        for cat, score in score_dict.items():
            if not cat in total_scores:
                total_scores[cat] = score
            else:
                total_scores[cat] += score
        cv2.waitKey(0)
    t4 = time.time()
    print(f"MODEL TOOK {t4-t3} SECONDS")
    # ====================================
    
    prediction = max(total_scores, key=total_scores.get)
    print(f"TOTAL SCORES: {total_scores}")
    print(f"PREDICTED VALUE: {prediction}")
        
    if prediction != CURRENT_BIN:
        beep(correct=False)
    else:
        beep(correct=True)

    # updateDatabase(prediction)        

    cv2.destroyAllWindows()
    clean_images()

def beep(correct):
    if correct:  
        winsound.PlaySound(f"{SAVE_FILE}/sounds/good.wav", winsound.SND_FILENAME)
    else:
        winsound.PlaySound(f"{SAVE_FILE}/sounds/bad.wav", winsound.SND_FILENAME)

if __name__ == '__main__':
    model = load_model('models/deeptrash.h5')
    fgbg = cv2.createBackgroundSubtractorMOG2()
    go_pro = GoProCamera.GoPro()
    go_pro.mode(mode="0")
    go_pro.video_settings(res="1080p", fps="60")

    # t0 = time.time()
    # reset_camera(go_pro)
    # frames = video_stream(go_pro)
    # run_ml(frames)
    # t1 = time.time()
    # print(f"TOTAL TOOK {t1-t0} SECONDS")

    video_labels(go_pro, "plastic")

    # schedule.every(30).seconds.do(run)
    # while True:
    #     scheduling.run_pending()
    #     time.sleep(1)