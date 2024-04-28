from recognizer.lp_bounding_box import LPBoundingBox

class LPRecognizerLPData:
    def __init__(self):
        self.box: LPBoundingBox = None
        self.car_box: LPBoundingBox = None
        self.confidence: float = None
        self.plate: str = None
        self.plate_confidence: float = None
        self.croppedim = None

    def clone(self):
        clone = LPRecognizerLPData()
        clone.box = self.box
        clone.car_box = self.car_box
        clone.confidence = self.confidence
        clone.plate = self.plate
        clone.plate_confidence = self.plate_confidence
        return clone