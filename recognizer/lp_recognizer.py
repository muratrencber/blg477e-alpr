from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes
from recognizer.lp_recognizer_frame_data import LPRecognizerFrameData, LPRecognizerLPData
from recognizer.lp_bounding_box import LPBoundingBox
from cv2.typing import MatLike
from torch import Tensor
from easyocr import Reader
import cv2 as cv

def crop_image(image: MatLike, box: LPBoundingBox) -> Tensor:
    xt, yt, xb, yb = box.xtop, box.ytop, box.xbottom, box.ybottom
    height, width, _ = image.shape
    x1, y1, x2, y2 = int(xt * width), int(yt * height), int(xb * width), int(yb * height)
    return image[y1:y2, x1:x2]

def preprocess_img_for_ocr(img: MatLike) -> MatLike:
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 64, 255, cv.THRESH_BINARY_INV)
    return gray

class LPRecognizer:
    def __init__(self, lpr_model: YOLO, carr_model: YOLO = None, lpr_reader: Reader = None, follow_cont = True) -> None:
        self.lpr_reader: Reader = lpr_reader
        self.carr_model: YOLO = carr_model
        self.lpr_model: YOLO = lpr_model
        self.frame_data: list[LPRecognizerFrameData] = []
        self.follow_cont: bool = follow_cont and carr_model is not None
        self.frame_counter: int = 0
    
    def recognize_frame(self, frame: MatLike) -> LPRecognizerFrameData:
        if self.follow_cont:
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
    
    def process_lp_box(self, bounding_box: Boxes) -> LPBoundingBox:
        xt, yt, xb, yb = bounding_box.xyxyn[0]
        lp_box = LPBoundingBox(float(xt), float(yt), float(xb), float(yb))
        return lp_box
    
    def read_lp(self, image: MatLike, lpdata: LPRecognizerLPData):
        if self.lpr_reader is None:
            return
        lp_image = crop_image(image, lpdata.box)
        lp_image = preprocess_img_for_ocr(lp_image)
        lpdata.croppedim = lp_image
        max_conf = 0
        max_len = 0
        read_results = self.lpr_reader.readtext(lp_image)
        for _, text, conf in read_results:
            if len(text) < max_len or conf < max_conf:
                continue
            max_len = len(text)
            max_conf = conf
            lpdata.plate = text
            lpdata.plate_confidence = conf
        
    def recognize_frame_cont(self, frame: MatLike) -> LPRecognizerFrameData:
        raise NotImplementedError("Continous recognition is not implemented yet")
