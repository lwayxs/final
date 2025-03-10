import cv2
import os

path = ''
files = os.listdir(path)
for file in files:
    file_path = os.path.join(path, file)
    I = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR)

    b, g, r = cv2.split(I)

    b1 = cv2.equalizeHist(b)
    g1 = cv2.equalizeHist(g)
    r1 = cv2.equalizeHist(r)

    O = cv2.merge([b1, g1, r1])

    cv2.imwrite(file_path, O)