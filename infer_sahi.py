import time
from ultralytics import YOLO
import numpy as np
import cv2
from tracker.byte_tracker import BYTETracker
import argparse
import os


from sahi.utils.yolov8 import (
    download_yolov8s_model,
)

# Import required functions and classes
from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from sahi.prediction import visualize_object_predictions
from IPython.display import Image
from numpy import asarray

from ultralytics.utils.plotting import Annotator, colors, save_one_box

def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=120, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument('--aspect_ratio_thresh', type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument("--video_dir", type=str, default='/yolov8/videos',help="Path to the directory containing videos")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def get_color(idx):

    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)

    return color

def get_video_files(video_dir):
    video_files = []
    for root, dirs, files in os.walk(video_dir):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.MP4')):  # 根据需要添加其他视频格式
                video_files.append(os.path.join(root, file))
    return video_files

def plot_tracking(image, tlwhs, obj_ids,  frame_id=0, fps=0., ids2=None):

    im = np.ascontiguousarray(np.copy(image))

    text_scale = 2
    text_thickness = 2
    line_thickness = 3

    cv2.putText(im, 'frame: %d fps: %.2f num: %d' % (frame_id, fps, len(tlwhs)),
                (0, int(15 * text_scale)), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)

    for i, tlwh in enumerate(tlwhs):
        x1, y1, w, h = tlwh
        intbox = tuple(map(int, (x1, y1, x1 + w, y1 + h)))
        obj_id = int(obj_ids[i])
        id_text = '{}'.format(int(obj_id))
        if ids2 is not None:
            id_text = id_text + ', {}'.format(int(ids2[i]))
        color = get_color(abs(obj_id))

        cv2.rectangle(im, intbox[0:2], intbox[2:4], color=color, thickness=line_thickness)
        cv2.putText(im, id_text, (intbox[0], intbox[1]), cv2.FONT_HERSHEY_PLAIN, text_scale, (0, 0, 255),
                    thickness=text_thickness)
    return im

def main(args,weights_path,video_dir,framePath):

    # 设置模型文件
    model = YOLO(weights_path)
    tracker = BYTETracker(args, frame_rate=24)
    

    # model.conf = 0.25  # NMS confidence threshold
    # model.iou = 0.1  # NMS IoU threshold
    # model.classes = [0]  # perform detection on only several classes


    # 读取视频
    
    video_files = get_video_files(video_dir)

    for video_path in video_files:
        # 解析视频文件名（不含扩展名）
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        # 创建结果视频文件的路径
        result_video_path = os.path.join('./result', f'{video_name}_tracked.mp4')

        # 创建跟踪结果文本文件的路径
        results_file = os.path.join('./result', f'{video_name}_tracking_results.txt')

        # 创建sahi预测视频文件
        result_video_path_forsahi = os.path.join('./result/sahi', f'{video_name}_predict.mp4')

        # 确保结果目录存在
        os.makedirs(os.path.dirname(result_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(result_video_path_forsahi), exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        out_sahi = cv2.VideoWriter(result_video_path_forsahi, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_id = 0
        # results = []


        # 如果文件不存在，则创建文件
        if not os.path.exists(results_file):
            with open(results_file, "w"):
                pass  # 什么都不做，只是创建文件
        
        with open(results_file, "a") as file:
            while True:
                ret_val, frame = cap.read()

                if ret_val:
                    t0 = time.time()
                    # 添加sahi
                    # results_nosahi = model(frame, iou=0.1, conf=0.1, imgsz=1920)[0]
                    # # 初始化model
                    model.conf = 0.1  # NMS confidence threshold
                    model.iou = 0.4  # NMS IoU threshold
                    model.classes = [0]  # perform detection on only several classes
                    # 初始化sahi-model
                    detection_model = AutoDetectionModel.from_pretrained(
                        model_type='yolov8',
                        model=model)
                    # 获得detection结果
                    results_boxes = get_sliced_prediction(
                        frame,
                        detection_model,
                        slice_height=640,
                        slice_width=640,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2
                    )
                    # 保存sahi预测图像
                    # img_sahi = cv2.imread(frame,cv2.IMREAD_UNCHANGED)
                    # img_converted = cv2.cvtColor(img_sahi, cv2.COLOR_BGR2RGB)
                    numpydata = asarray(frame)
                    result_sahi_video = visualize_object_predictions(
                        numpydata,
                        object_prediction_list=results_boxes.object_prediction_list,
                        hide_labels=None,
                        output_dir='./result/sahi',
                        file_name= os.path.join('./result/sahi', f'{frame_id}_prdict'),
                        export_format='jpg'
                    )
                    out_sahi.write(result_sahi_video['image'])



                    extracted_boxes = []
                    confidence_scores = []
                    categories = []

                    for obj_pred in results_boxes.object_prediction_list:
                        # 提取目标框位置
                        bbox = obj_pred.bbox.to_xyxy()

                        # x_min, y_min, x_max, y_max = bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax
                        extracted_boxes.append(np.array(bbox))

                        # 提取置信度
                        confidence = obj_pred.score.value
                        confidence_scores.append(confidence)

                        # 提取类别
                        category = obj_pred.category.name
                        categories.append(category)

                    
                    # print('extracted_boxes___________________________')
                    # print(np.array(extracted_boxes))
                    # print(type(np.array(extracted_boxes)))
                    # print('results.boxes.cpu().numpy().xyxy___________________________')
                    # print(results.boxes.cpu().numpy().xyxy)
                    # print(type(results.boxes.cpu().numpy().xyxy))
                    # print('confidence_scores___________________________')
                    # print(confidence_scores)
                    # print(type(confidence_scores))
                    # print('results.boxes.cpu().numpy().conf___________________________')
                    # print(results.boxes.cpu().numpy().conf)
                    # print(type(results.boxes.cpu().numpy().conf))
                
                    
                    # 数据转化：

                    # boxes = results_boxes.boxes.cpu().numpy()
                    # boxes = extracted_boxes
                    # print(len(boxes))
                    if len(results_boxes.object_prediction_list) > 0:
                        bboxes = np.array(extracted_boxes)
                        scores = np.array(confidence_scores)

                        online_targets = tracker.update(bboxes,scores)
                        online_tlwhs = []
                        online_ids = []
                        online_scores = []
                        for i, t in enumerate(online_targets):
                            # tlwh = t.tlwh
                            tlwh = t.tlwh_yolox
                            tid = t.track_id
                            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
                            # if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                            if tlwh[2] * tlwh[3] > args.min_box_area:
                                online_tlwhs.append(tlwh)
                                online_ids.append(tid)
                                online_scores.append(t.score)
                                # results.append(
                                #     f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                # )
                                # 保存位置信息到文件
                                # 将跟踪后的结果追加到文件中
                                file.write(
                                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )

                            # 将跟踪后的帧写入视频

                        t1 = time.time()
                        time_ = (t1 - t0) * 1000

                        online_im = plot_tracking(frame, online_tlwhs, online_ids, frame_id=frame_id + 1,
                                                fps=1000. / time_)
                        out.write(online_im)
                        # cv2.imshow("frame", online_im)

                    else:
                        t1 = time.time()
                        time_ = (t1 - t0) * 1000
                        cv2.putText(frame, 'frame: %d fps: %.2f num: %d' % (frame_id, 1000. / time_, 0),
                                    (0, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=2)
                        out.write(frame)
                        # cv2.imshow("frame",frame) 0

                    if cv2.waitKey(10) == 'q':
                        break

                    frame_id += 1
                    t2 = time.time()
                    print("infer and track time: {} ms".format((t2 - t0) * 1000))
                    print()
                else:
                    break
        # 释放视频写入对象
        out.release()
        out_sahi.release()
        # 释放视频捕获对象
        cap.release()


if __name__ == "__main__":
    args = make_parser().parse_args()

    weights_path = "/workspace/yolov8_byte/weights/yolov8_p2_sahi/best.pt"
    video_dir = args.video_dir
    framePath = '/yolov8/20231219yolo/images/train/video_2023_12_13_03_00_12_frame_frame_000119.PNG'

    main(args, weights_path, video_dir, framePath)

