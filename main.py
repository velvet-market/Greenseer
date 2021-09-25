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

FPS = 60
SECONDS_BUFFER = 5
CUTOFF_SCORE = 0.075
CATEGORY = ["nothing", "compost", "recycle", "trash"]
CURRENT_BIN = "recycle"
SAVE_FILE = "data"

def predict(img):
    # if the size of the image is too small, report none
    if img.size < 100:
        return -1, 'none'

    # predict the image class
    pred = model.predict(img.reshape(1, *img.shape)[0:480, 80: 560])

    # get the index of the maximum prediction value
    idx = np.argmax(pred)

    # return the index along with its associated category
    return idx, CATEGORY[idx]

def check_background():
    frames = glob.glob(f"{SAVE_FILE}/gopro/*")
    for frame in frames:
        img = cv2.imread(frame)
        bg_score = get_background_score(img)
        print(f"{frame}: {bg_score}")
        if bg_score > CUTOFF_SCORE:
            cv2.imwrite(f"{SAVE_FILE}/input/{os.path.basename(frame)}", img)
            os.remove(frame)

def get_background_score(img):
    denom = img.shape[0] * img.shape[1] * img.shape[2]
    mask = fgbg.apply(img) / 255
    return np.sum(mask) / denom

def run_ml(show=True, crop=True, prediction_threshold=5):
    check_background()
    imgs = glob.glob(f"{SAVE_FILE}/input/*")

    total_scores = {}

    if imgs:
        for img in imgs:
            cv2.imshow('img', img)
            idx, category = predict(img)
            print(f'ML MODEL {idx}: {category}')
            cv2.waitKey(0)

            # idx, score_dict = predict(img)
            for cat, score in score_dict:
                if not cat in total_score:
                    total_score[cat] = score[cat]
                else:
                    total_score[cat] += score[cat]
            prediction = max(total_scores, key=stats.get)


    cv2.destroyAllWindows()

def beep_trash(category):
    print('Running dump trash routine for', category, CATEGORY[category % 3])

if __name__ == '__main__':
    model = load_model('deeptrash.h5')
    fgbg = cv2.createBackgroundSubtractorMOG2()

    run_ml(show=True, crop=False)

    # schedule.every(30).seconds.do(run)
    # while True:
    #     scheduling.run_pending()
    #     time.sleep(1)