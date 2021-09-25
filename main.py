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
import winsound

FPS = 30
SECONDS_BUFFER = 5
NUM_FILES = 5
CUTOFF_SCORE = 0.075
CATEGORY = ["nothing", "compost", "recycle", "trash"]
CURRENT_BIN = "recycle"
SAVE_FILE = "data"

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

def check_background():
    frames = glob.glob(f"{SAVE_FILE}/gopro/*.jpg")
    for frame in frames:
        img = cv2.imread(frame)
        bg_score = get_background_score(img)
        print(f"{frame}: {bg_score}")
        if bg_score > CUTOFF_SCORE:
            cv2.imwrite(f"{SAVE_FILE}/input/{os.path.basename(frame)}", img)
            os.remove(frame)
            os.remove(f"{SAVE_FILE}/gopro/cam_video.MP4")

def get_background_score(img):
    denom = img.shape[0] * img.shape[1] * img.shape[2]
    mask = fgbg.apply(img) / 255
    return np.sum(mask) / denom

def run_ml():
    check_background()

    files = glob.glob(f"{SAVE_FILE}/input/*")
    if len(files) > 5:
        indices = np.round(np.linspace(0, len(files) - 1, 5)).astype(int)
        files = files[indices]

    total_scores = {}
    if files:
        t0 = time.time()

        for file in files:
            img = cv2.imread(file)
            # cv2.imshow('img', img)

            score_dict = predict(img)
            print(f'ML MODEL {file}: {score_dict}')

            for cat, score in score_dict.items():
                if not cat in total_scores:
                    total_scores[cat] = score
                else:
                    total_scores[cat] += score
            cv2.waitKey(0)

        t1 = time.time()
        print(f"TOOK {t1-t0} SECONDS")
            
        prediction = max(total_scores, key=total_scores.get)
            
        if prediction != CURRENT_BIN:
            beep(correct=False)
        else:
            beep(correct=True)

        # updateDatabase(prediction)

    cv2.destroyAllWindows()

def beep(correct):
    if correct:  
        winsound.PlaySound(f"{SAVE_FILE}/sounds/good.wav", winsound.SND_FILENAME)
    else:
        winsound.PlaySound(f"{SAVE_FILE}/sounds/bad.wav", winsound.SND_FILENAME)

if __name__ == '__main__':
    model = load_model('deeptrash.h5')
    fgbg = cv2.createBackgroundSubtractorMOG2()

    run_ml()
    # schedule.every(30).seconds.do(run)
    # while True:
    #     scheduling.run_pending()
    #     time.sleep(1)