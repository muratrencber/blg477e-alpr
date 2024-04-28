class LPBoundingBox:
    def __init__(self, xtop: float, ytop: float, xbottom: float, ybottom: float, normalized: bool = True) -> None:
        self.xtop: float = xtop
        self.ytop: float = ytop
        self.xbottom: float = xbottom
        self.ybottom: float = ybottom
        self.normalized: bool = normalized
    def normalize(self, width: int, height: int) -> None:
        self.xtop /= width
        self.ytop /= height
        self.xbottom /= width
        self.ybottom /= height