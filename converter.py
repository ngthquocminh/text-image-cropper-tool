import cv2
import sys
import os
import numpy as np


def main(arg):
    img_path = arg[0]
    img_out = arg[1]

    files = os.listdir(img_path)
    for index, file in enumerate(files):
        if str.lower(file.split(".")[-1]) not in ["jpg", "png"]:
            continue

        img = cv2.imread(img_path + "/" + file)

        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(img_out + "/mono_" + file, gray)
            print("[DONE]: ", file)
        except:
            print("[FAILED]: ", file)

if __name__ == "__main__":
    main(sys.argv[1:])




