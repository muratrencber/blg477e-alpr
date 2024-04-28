from recognizer.lp_recognizer_lp_data import LPRecognizerLPData

class LPRecognizerFrameData:
    def __init__(self):
        self.frame: int = 0
        self.lps: dict[str, LPRecognizerLPData] = {}
    def clone(self):
        clone = LPRecognizerFrameData()
        clone.frame = self.frame
        for key, value in self.lps.items():
            clone.lps[key] = value.clone()
        return clone