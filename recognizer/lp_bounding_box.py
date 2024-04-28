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
    def overlap_amount(self, child: "LPBoundingBox") -> float:
        nxt, nyt, nxb, nyb = self.xtop, self.ytop, self.xbottom, self.ybottom
        cxt, cyt, cxb, cyb = child.xtop, child.ytop, child.xbottom, child.ybottom
        if cxt > nxt:
            nxt = min(cxt, nxb)
        if cyt > nyt:
            nyt = min(cyt, nyb)
        if cxb < nxb:
            nxb = max(cxb, nxt)
        if cyb < nyb:
            nyb = max(cyb, nyt)
        csize = (cxb - cxt) * (cyb - cyt)
        nsize = (nxb - nxt) * (nyb - nyt)
        return nsize / csize
    def get_points(self, width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        p1 = (int(self.xtop * width), int(self.ytop * height))
        p2 = (int(self.xbottom * width), int(self.ybottom * height))
        return (p1, p2)
        