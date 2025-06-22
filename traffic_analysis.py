import supervision as sv
from rfdetr import RFDETRBase
from rfdetr.util.coco_classes import COCO_CLASSES

from typing import Dict, List, Set, Iterable
import cv2
import os
import numpy as np
from tqdm import tqdm
from tkinter import Tk
from tkinter.filedialog import askopenfilename

VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]
VEHICLE_CLASS_COLORS = {
    3: sv.Color.from_hex("#FF8133"),
    6: sv.Color.from_hex("#42DB06"),
    8: sv.Color.from_hex("#2DC582"),
    4: sv.Color.from_hex("#FFB52D"),
}

#a73b
ZONE_OUT_POLYGONS = [
    np.array([[250, 925], [700, 875], [700, 925], [250, 975]]),
    np.array([[725, 875], [1450, 800], [1450, 850], [725, 925]]),
]

#a729
# ZONE_OUT_POLYGONS = [
#     np.array([[150, 675], [350, 675], [350, 825], [150, 825]]),
#     np.array([[550, 955], [1725, 855], [1725, 905], [550, 1005]]),
# ]

def save_gt_output(frame_id: int, detections: sv.Detections, output_file: str):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "a") as f:
        for xyxy, track_id in zip(detections.xyxy, detections.tracker_id):
            if track_id is None:
                continue
            x1, y1, x2, y2 = xyxy
            w = x2 - x1
            h = y2 - y1
            f.write(f"{frame_id},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f},1,-1,-1,-1\n")

def initiate_polygon_zones(
    polygons: List[np.ndarray],
    triggering_anchors: Iterable[sv.Position] = [sv.Position.BOTTOM_CENTER],
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            triggering_anchors=triggering_anchors,
        )
        for polygon in polygons
    ]

class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_class_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections: sv.Detections,
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id, class_id in zip(detections_out_zone.tracker_id, detections_out_zone.class_id):
                self.tracker_id_to_class_id[tracker_id] = class_id
                self.counts.setdefault(zone_out_id, {})
                self.counts[zone_out_id].setdefault(class_id, set())
                self.counts[zone_out_id][class_id].add(tracker_id)
        return detections

class VideoProcessor:
    def __init__(
        self,
        source_weights_path: str,
        source_video_path: str,
        target_video_path: str = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.8,
    ) -> None:
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path

        self.cap = cv2.VideoCapture(source_video_path)
        self.frame_idx = 0

        self.video_name = os.path.splitext(os.path.basename(source_video_path))[0]

        self.image_dir = os.path.join("data/images", self.video_name)
        os.makedirs(self.image_dir, exist_ok=True)

        self.model = RFDETRBase(pretrain_weights=source_weights_path)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.BLACK, text_scale=0.3, text_thickness=1)
        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=150,
            minimum_matching_threshold=0.7,
            frame_rate=30,
            minimum_consecutive_frames=4
        )

        self.zones_out = initiate_polygon_zones(ZONE_OUT_POLYGONS)
        self.width = self.video_info.width
        self.height = self.video_info.height
        self.detections_manager = DetectionsManager()
        self.blocked_ids = set()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(self.target_video_path, self.video_info) as sink:
                for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        detections = self.model.predict(frame, threshold=self.conf_threshold)
        
        valid_indices = [
            i for i, class_id in enumerate(detections.class_id)
            if COCO_CLASSES[class_id] in VEHICLE_CLASSES
        ]
        filtered_detections = detections[valid_indices]
        tracked_detections = self.tracker.update_with_detections(filtered_detections)

        detections_out_zones = [
            tracked_detections[zone_out.trigger(detections=tracked_detections)]
            for zone_out in self.zones_out
        ]

        self.detections_manager.update(tracked_detections, detections_out_zones)

        save_gt_output(
            frame_id=self.frame_idx,
            detections=tracked_detections,
            output_file="data/gt/ccc26170a73b_1.20241022.1743/run.txt"
        )

        return self.annotate_frame(frame, tracked_detections)

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()

        for zone_out in self.zones_out:
            annotated_frame = sv.draw_polygon(annotated_frame, zone_out.polygon, sv.Color.GREEN)

        labels = []
        for class_id, tracker_id, confidence in zip(detections.class_id, detections.tracker_id, detections.confidence):
            class_name = COCO_CLASSES.get(class_id, "unknown")
            label = f"#{tracker_id} {confidence:.2f}"
            labels.append(label)

        annotated_frame = self.box_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.label_annotator.annotate(annotated_frame, detections, labels)

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)

            offset_x = 20
            offset_y = 20

            class_order = ["car", "bus", "truck", "motorcycle"]
            positions = [
                (-offset_x, -offset_y), (offset_x, -offset_y),
                (-offset_x, offset_y), (offset_x, offset_y)
            ]

            for i, class_name in enumerate(class_order):
                class_id = next((k for k, v in COCO_CLASSES.items() if v == class_name), None)
                if class_id is None:
                    continue

                count = len(self.detections_manager.counts.get(zone_out_id, {}).get(class_id, set()))
                dx, dy = positions[i]
                text_anchor = sv.Point(x=zone_center.x + dx, y=zone_center.y + dy)

                annotated_frame = sv.draw_text(
                    scene=annotated_frame,
                    text=f"{count}",
                    text_anchor=text_anchor,
                    background_color=VEHICLE_CLASS_COLORS.get(class_id, sv.Color.WHITE),
                    text_scale=0.3,
                    text_thickness=1
                )
        
        image_path = os.path.join(self.image_dir, f"{self.frame_idx + 1}.jpg")
        cv2.imwrite(image_path, annotated_frame)
        self.frame_idx += 1

        return annotated_frame

if __name__ == "__main__":
    Tk().withdraw()
    source_video_path = askopenfilename(
        title="Select input video file",
        filetypes=[("All files", "*.*")]
    )
    if not source_video_path:
        print("No file selected. Exiting.")
        exit()
    output_dir = os.path.join(".", "data", "processed_video")
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(source_video_path)
    name_no_ext, ext = os.path.splitext(base_name)
    target_video_path = os.path.join(output_dir, f"{name_no_ext}_output{ext}"
    )
    processor = VideoProcessor(
        # source_weights_path=args.source_weights_path,
        # source_video_path=args.source_video_path,
        # target_video_path=args.target_video_path,
        source_weights_path="rf-detr-base.pth",
        source_video_path=source_video_path,
        target_video_path=target_video_path,
    )
    processor.process_video()