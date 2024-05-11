from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from recognizer.lp_recognizer_frame_data import LPRecognizerFrameData, LPRecognizerLPData
from recognizer.lp_bounding_box import LPBoundingBox
from cv2.typing import MatLike
from torch import Tensor
from easyocr import Reader
from motpy import Detection, MultiObjectTracker, Track
import cv2 as cv
import numpy as np
import sympy

RELEVANT_VEHICLE_CLASSES = [2,3,5,7] #2: car, 3: motorcycle, 5: bus, 7: truck
OVERLAP_THRESHOLD = 0.5

NUM_TO_CHAR = {
    "1": "I",
    "0": "O",
    "2": "Z",
    "3": "B",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "G",
    "{": "1",
    "}": "1",
    "[": "1",
    "]": "1",
}

CHAR_TO_NUM = {
    "I": "1",
    "O": "0",
    "Z": "2",
    "B": "3",
    "A": "4",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "G": "9",
    "I": "{",
    "1": "}",
    "1": "[",
    "1": "]",

}

def crop_image(image: MatLike, box: LPBoundingBox) -> Tensor:
    xt, yt, xb, yb = box.xtop, box.ytop, box.xbottom, box.ybottom
    height, width, _ = image.shape
    x1, y1, x2, y2 = int(xt * width), int(yt * height), int(xb * width), int(yb * height)
    return image[y1:y2, x1:x2]

def process_plate_img(img: MatLike) -> tuple[MatLike, MatLike]:
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask = cv.inRange(gray_img, 0, 150)
    inv_mask = cv.bitwise_not(mask)
    return inv_mask, gray_img

def preprocess_img_for_ocr(img: MatLike) -> MatLike:
    mask, gray_img = process_plate_img(img)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    max_area = -1
    max_contour = np.array([])
    mcfull = None
    for c in contours:
        area = cv.contourArea(c)
        if area < 400:
            continue
        if area <= max_area:
            continue
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.06 * peri, True)
        mcfull = c
        if len(approx) != 4:
            continue
        max_area = area
        max_contour = approx
    if max_area == -1:
        copmask = mask.copy()
        rgbmask = cv.cvtColor(copmask, cv.COLOR_GRAY2BGR)
        return (rgbmask, False)
    s1,s2,s3 = max_contour.shape
    points = max_contour.reshape(s1, 2)
    coords = np.zeros((4, 2), dtype='float32')

    psum = points.sum(axis=1)
    coords[0] = points[np.argmin(psum)]
    coords[3] = points[np.argmax(psum)]
    pdiff = np.diff(points, axis=1)
    coords[1] = points[np.argmin(pdiff)]
    coords[2] = points[np.argmax(pdiff)]

    tl, tr, bl, br = coords
    bw = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
    tw = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
    rh = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)
    lh = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
    max_w = max(bw, tw)
    max_h = max(rh, lh)

    plate_ar = 11/52
    if max_h/max_w > plate_ar:
        max_h = max_w * plate_ar
    else:
        max_w = max_h / plate_ar

    target_points = np.float32([[0,0], [int(max_w), 0], [0, int(max_h)], [int(max_w), int(max_h)]])
    mat = cv.getPerspectiveTransform(coords, target_points)
    res = cv.warpPerspective(gray_img, mat, (int(max_w), int(max_h)))
    _, thresh = cv.threshold(gray_img, 64, 255, cv.THRESH_BINARY_INV)
    return (cv.cvtColor(res, cv.COLOR_GRAY2BGR), True)

def try_convert_platenumber(plate: str) -> str | None:
    plate = plate.replace(" ", "")
    gibberish = "()\{\}[]<>"
    plate_excluded_gibberish = plate
    for c in gibberish:
        plate_excluded_gibberish = plate_excluded_gibberish.replace(c, "")
    plen = len(plate)
    pwogibberishlen = len(plate_excluded_gibberish)
    pfits = plen == 7 or plen == 8
    pwogibfits = pwogibberishlen == 7 or pwogibberishlen == 8
    if not pfits or not pwogibfits:
        return None

class LPRecognizer:
    def __init__(self, lpr_model: YOLO, carr_model: YOLO = None, lpr_reader: Reader = None, follow_cont = True) -> None:
        self.lpr_reader: Reader = lpr_reader
        self.carr_model: YOLO = carr_model
        self.lpr_model: YOLO = lpr_model
        self.frame_data: list[LPRecognizerFrameData] = []
        self.follow_cont: bool = follow_cont and carr_model is not None
        self.frame_counter: int = 0
        self.cont_classes = RELEVANT_VEHICLE_CLASSES
        self.tracker: MultiObjectTracker = None

    def setup_tracking(self, fps: int) -> None:
        if not self.follow_cont:
            return
        self.tracker = MultiObjectTracker(dt=1/fps)
    
    def recognize_frame(self, frame: MatLike) -> LPRecognizerFrameData:
        if self.follow_cont and self.tracker is not None:
            return self.recognize_frame_cont(frame)
        lpr_results: Results = self.lpr_model(frame)[0] #we only input one frame, so we only need the first element
        frame_data = LPRecognizerFrameData()
        frame_data.frame = self.frame_counter

        id_count = 0
        for bounding_box in lpr_results.boxes:
            lpdata = LPRecognizerLPData()
            lpdata.box = self.process_lp_box(bounding_box)
            lpdata.confidence = float(bounding_box.conf)
            self.read_lp(frame, lpdata)
            frame_data.lps[f"{id_count}"] = lpdata
            id_count += 1

        self.frame_counter += 1
        self.frame_data.append(frame_data)
        return frame_data
    
    def recognize_frame_cont(self, frame: MatLike) -> LPRecognizerFrameData:
        car_results: Results = self.carr_model(frame)[0]
        lpr_results: Results = self.lpr_model(frame)[0]
        car_tracking_detections: list[Detection] = [self.yolo_to_motpy_detection(box) for box in car_results.boxes if box.cls in self.cont_classes]
        tracks = self.tracker.step(detections=car_tracking_detections)
        last_results = self.frame_data[-1] if self.frame_counter > 0 else None
        new_results = LPRecognizerFrameData()
        new_results.frame = self.frame_counter
        for t in tracks:
            bbox = self.process_car_box(frame, t)
            lpdata = LPRecognizerLPData()
            if last_results is not None and f"{t.id}" in last_results.lps:
                last_lpd = last_results.lps[f"{t.id}"]
                lpdata.plate = last_lpd.plate
                lpdata.plate_confidence = last_lpd.plate_confidence
            lpdata.car_box = bbox
            new_results.lps[f"{t.id}"] = lpdata
        id_count = 0
        for bounding_box in lpr_results.boxes:
            lp_box = self.process_lp_box(bounding_box)
            lpdata = self.find_car_of_lp(lp_box, new_results)
            if lpdata is None:
                lpdata = LPRecognizerLPData()
                new_results.lps[f"uc{id_count}"] = lpdata
                id_count += 1
            lpdata.box = lp_box
            lpdata.confidence = float(bounding_box.conf)
            self.read_lp(frame, lpdata)
        self.frame_counter += 1
        self.frame_data.append(new_results)
        return new_results
    
    def find_car_of_lp(self, lp_box: LPBoundingBox, frame_data: LPRecognizerFrameData) -> LPRecognizerLPData | None:
        selected, selected_threshold = None, 0
        for lpdata in frame_data.lps.values():
            if lpdata.car_box is None or lpdata.box is not None:
                continue
            overlap = lpdata.car_box.overlap_amount(lp_box)
            if overlap > selected_threshold and overlap > OVERLAP_THRESHOLD:
                selected = lpdata
                selected_threshold = overlap
        return selected

    
    def process_lp_box(self, bounding_box: Boxes) -> LPBoundingBox:
        xt, yt, xb, yb = bounding_box.xyxyn[0]
        lp_box = LPBoundingBox(float(xt), float(yt), float(xb), float(yb))
        return lp_box
    
    def process_car_box(self, frame: MatLike, track: Track) -> LPBoundingBox:
        height, width, _ = frame.shape
        xta, yta, xba, yba = track.box
        xt, yt, xb, yb = xta / width, yta / height, xba / width, yba / height
        return LPBoundingBox(float(xt), float(yt), float(xb), float(yb))

    
    def yolo_to_motpy_detection(self, boxes: Boxes) -> Detection:
        xta, yta, xba, yba = boxes.xyxy[0]
        pos = np.array([float(xta), float(yta), float(xba), float(yba)])
        pos.reshape(2,2)
        return Detection(pos, float(boxes.conf), int(boxes.cls))
    
    def read_lp(self, image: MatLike, lpdata: LPRecognizerLPData):
        if self.lpr_reader is None:
            return
        lp_image = crop_image(image, lpdata.box)
        lp_image, res = preprocess_img_for_ocr(lp_image)
        conf_mult = 1 if res else 0.5
        max_conf = 0
        if lpdata.plate_confidence is not None:
            max_conf = lpdata.plate_confidence
        max_len = 0
        if lpdata.plate is not None:
            max_len = len(lpdata.plate)
        read_results = self.lpr_reader.readtext(lp_image)
        for _, text, conf in read_results:
            conf *= conf_mult
            if len(text) < max_len:
                continue
            max_len = len(text)
            max_conf = conf
            lpdata.plate = text
            lpdata.plate_confidence = conf
        
