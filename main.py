import numpy as np
import cv2
def circularity_area(area,perimeter):
    return (4*np.pi)*(area/(perimeter*perimeter))
def applyFunc(img_path,kernel=(12,12),min_color=[22, 93, 0],max_color=[45, 255, 255],c_method1=cv2.RETR_EXTERNAL,c_method2=cv2.CHAIN_APPROX_SIMPLE,min_area=20,min_c=0.8):
    image = cv2.imread(img_path)
    original = image[:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #image = cv2.erode(image,(3,3))
    #image = cv2.dilate(image,kernel)
    lower = np.array(min_color, dtype="uint8")
    upper = np.array(max_color, dtype="uint8")
    mask = cv2.inRange(image, lower, upper)

    cnts = cv2.findContours(mask, c_method1, c_method2)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c,True)
        print(area,perimeter)
        if perimeter == 0 and area < min_area:
            continue
        circularity = circularity_area(area,perimeter)
        if not (min_c < circularity < 1.1):
            continue
        cv2.rectangle(original, (x, y), (x + w, y + h), (255,255,255), 2)
    return original,mask
original,mask = applyFunc('test2.jpg')
cv2.imshow('test2.jpg', original)
cv2.imshow('tst',mask)
cv2.waitKey(0)
