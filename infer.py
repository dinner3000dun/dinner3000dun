import time
from ultralytics import YOLO
import numpy as np
import cv2
from tracker.byte_tracker import BYTETracker
import argparse
import os

from numpy import asarray


def make_parser():
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.1, help="matching threshold for tracking")
    parser.add_argument('--aspect_ratio_thresh', type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument("--video_dir", type=str, default='/yolov8/videos',help="Path to the directory containing videos")
    parser.add_argument('--min_box_area', type=float, default=100, help='filter out tiny boxes')
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

def main(args,weights_path,video_path):
    
    # 设置模型文件
    model = YOLO(weights_path)
    tracker = BYTETracker(args, frame_rate=24)
    
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
        result_video_predict = os.path.join('./result/sahi', f'{video_name}_predict.mp4')


        # 确保结果目录存在
        os.makedirs(os.path.dirname(result_video_path), exist_ok=True)
        os.makedirs(os.path.dirname(result_video_predict), exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        results = []

        # 设置视频保存参数
        out = cv2.VideoWriter(result_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        out_sahi = cv2.VideoWriter(result_video_predict, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        frame_id = 0
        results_file = "./result/tracking_results.txt"

        # yolov8测试
        result_pic = model.predict(
                os.path.join(args.video_dir, video_name+'.MP4'),
                
                save_conf=True,
                show=False,
                save=True,
                save_txt=True,
                save_crop=True,)[0]

        # 如果文件不存在，则创建文件
        if not os.path.exists(results_file):
            with open(results_file, "w"):
                pass  # 什么都不做，只是创建文件
        
        with open(results_file, "a") as file:
            while True:
                ret_val, frame = cap.read()

                if ret_val:
                    t0 = time.time()
                    
                    results_boxes = model(frame,imgsz=1920,)[0]
                   

                    boxes = results_boxes.boxes.cpu().numpy()
                    print(len(boxes))
                    if len(boxes) > 0:
                        bboxes = boxes.xyxy
                        scores = boxes.conf

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
                                results.append(
                                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                                )
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
                        # cv2.imshow("frame",frame)

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

        # 释放视频捕获对象
        cap.release()


if __name__ =="__main__":
    args = make_parser().parse_args()

    weights_path = "./weights/yolov8s-p6/best.pt"
    video_dir = args.video_dir
    video_path = "/yolov8/video/SAM_0459.MP4"

    # weights_path = "./weights/yolov8n.pt"
    # video_path = "/yolov8/video/palace.mp4"

    main(args,weights_path,video_path)