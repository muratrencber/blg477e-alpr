from recognizer.lp_recognizer_frame_data import LPRecognizerFrameData, LPRecognizerLPData
from recognizer.lp_bounding_box import LPBoundingBox
from recognizer.lp_recognizer import LPRecognizer
import csv

CSV_COLUMNS = ["frame", "lp_id", "lp_plate", "lp_box", "car_box", "lp_box_confidence", "lp_plate_confidence"]

def format_box(box: LPBoundingBox) -> str:
    if box is None:
        return None
    return f"{box.xtop},{box.ytop},{box.xbottom},{box.ybottom}"

def to_box(box_str: str) -> LPBoundingBox:
    if box_str is None or box_str.strip() == "":
        return None
    xtop, ytop, xbottom, ybottom = map(float, box_str.split(","))
    return LPBoundingBox(xtop, ytop, xbottom, ybottom)

def write_csv_lpdata(frame: int, id: str, data: LPRecognizerLPData, csv_writer) -> None:
    csv_writer.writerow([frame, id, data.plate, format_box(data.box), format_box(data.car_box), data.confidence, data.plate_confidence])

def write_csv_framedata(data: LPRecognizerFrameData, csv_writer) -> None:
    for id, lpdata in data.lps.items():
        write_csv_lpdata(data.frame, id, lpdata, csv_writer)

def write_csv_recognizer(recognizer: LPRecognizer, target_path: str) -> None:
    with open(target_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(CSV_COLUMNS)
        for data in recognizer.frame_data:
            write_csv_framedata(data, csv_writer)

def read_csv_lpdata(row) -> LPRecognizerLPData:
    data = LPRecognizerLPData()
    data.plate = row["lp_plate"]
    data.box = to_box(row["lp_box"])
    data.car_box = to_box(row["car_box"])
    if row["lp_box_confidence"] != "":
        data.confidence = float(row["lp_box_confidence"])
    if row["lp_plate_confidence"] != "":
        data.plate_confidence = float(row["lp_plate_confidence"])
    return data

def read_csv_framedata_list(path: str) -> dict[str, LPRecognizerFrameData]:
    with open(path, newline="") as csvfile:
        csv_reader = csv.DictReader(csvfile)
        frames = {}
        for row in csv_reader:
            frame = int(row["frame"])
            if frame not in frames:
                frames[frame] = LPRecognizerFrameData()
            framedata: LPRecognizerFrameData = frames[frame]
            id = row["lp_id"]
            lpdata = read_csv_lpdata(row)
            framedata.lps[id] = lpdata
        return dict(sorted(frames.items()))