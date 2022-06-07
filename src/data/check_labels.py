from ocr2yolon import read_ann_file, YoloAnnotationXYWHNorm
import cv2


data = read_ann_file('./0frame001965.txt')
ann = data[0][:-1].split(' ')
print(ann[1])
print(ann)
x1 = int(abs(float(ann[1]) - 0.5 * float(ann[3])) * 1920)
y1 = int(abs(float(ann[2]) - 0.5 * float(ann[4])) * 1080)
x2 = int(abs(float(ann[1]) + 0.5 * float(ann[3])) * 1920)
y2 = int(abs(float(ann[2]) + 0.5 * float(ann[4])) * 1080)
print(x1, y1, x2, y2)
#496 867 156 50
image = cv2.imread('./0frame001965.jpg')
image = cv2.rectangle(image, (x1, y1), (x2, y2), (155, 255, 0), 2)
cv2.imwrite('./new.jpg', image)
