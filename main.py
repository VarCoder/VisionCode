import numpy as np
import cv2

def circularity_area(area, perim):
    p_2 = perim ** 2
    return (4 * np.pi) * (area / p_2) if p_2 != 0 else 0

default_params = {
    'erosion-steps': 0, 'dilation-steps': 0,
    'erosion-size': 3, 'dilation-size': 12,
    'min-color': [22, 93, 0],
    'max-color': [45, 255, 255],
    'contour-hierarchy': False, 'contour-lossy-compression': True,
    'min-area': 20, 'min-circularity': 0.8
}

def extract_cv(original, params= default_params):
    image = original[:]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    for _ in range(params['erosion-steps']):
        image = cv2.erode(image, (params['erosion-size'], params['erosion-size']))
    for _ in range(params['dilation-steps']):
        image = cv2.dilate(image, (params['dilation-size'], params['dilation-size']))
    lower = np.array(params['min_color'], dtype="uint8")
    upper = np.array(params['max_color'], dtype="uint8")
    mask = cv2.inRange(image, lower, upper)
    cnts = cv2.findContours(
        mask,
        cv2.RETR_TREE if params['contour-hierachy'] else cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE if params['contour-lossy-compression'] else cv2.CHAIN_APPROX_NONE
    )
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    bounding_rects = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c,True)
        if perimeter == 0 and area < params['min-area']:
            continue
        circularity = circularity_area(area,perimeter)
        if not (params['min-circularity'] < circularity < 1.1):
            continue
        bounding_rects.append((x, y, w, h))
    return {
        'rects': bounding_rects,
        '_mask': mask
    }

original = cv2.imread('test2.jpg')
results = extract_cv(original)

for (x, y, w, h) in results['rects']:
    cv2.rectangle(original, (x, y), (x + w, y + h), (255, 255, 255), 2)

cv2.imshow('test2.jpg', results['original'])
cv2.imshow('tst', results['_mask'])

cv2.waitKey(0)
