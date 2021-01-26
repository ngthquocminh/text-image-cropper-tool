import cv2
import sys
import os
import numpy as np


def main(arg: object) -> object:
    img_path = arg[0]
    img_out = arg[1]
    height = int(arg[2])

    if not os.path.exists(img_path) or not os.path.exists(img_out):
        print("Path error")
        return

    files = os.listdir(img_path)
    for index, file in enumerate(files):
        if str.lower(file.split(".")[-1]) not in ["jpg", "png"]:
            continue

        img = cv2.imread(img_path + "/" + file)
        _w = int(img.shape[1])
        _h = int(img.shape[0])

        # if _h <= height or _w <= height:
        #     print("[SKIP]: ", file)

        dim = lambda r, s: (int(s * r), s)

        try:
            resized = cv2.resize(img, dim(_w / _h, height), interpolation=cv2.INTER_AREA)
            cv2.imwrite(img_out + "/resized_" + file, resized)
            print("[DONE]: ", file)
        except:
            print("[FAILED]: ", file)


if __name__ == "__main__":
    main(sys.argv[1:])




