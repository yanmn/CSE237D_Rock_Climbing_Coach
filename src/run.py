import os
import argparse
import cv2
import numpy as np
import json

from utils.video_utils import get_video_array, crop_video
from utils.extraction import process_video
from utils.pc_complete_utils import compute_percent_complete, compute_percent_complete_color
from utils.pose_features import get_num_moves, compute_time_elapsed, compute_total_distance_traveled
from utils.move_validity_utils import getColorRoute, getPercentMoveValidity, getPercentHoldValidity

def get_parser():
    parser = argparse.ArgumentParser(description='Run script to produce a report from a climb video')
    parser.add_argument('-d', '--dir', type=str, default='./src/test_data',
                        help='filepath of climb video and hold image for generating report.txt')
    return parser

def get_data(args, get_cropped=True):
    files = os.listdir(args.dir)

    for file in files:
        if file.endswith('.mp4'):
            vid_path = os.path.join(args.dir, file)
        elif file.endswith('.jpg') or file.endswith('.png'):
            img_path = os.path.join(args.dir, file)
    
    raw_vid = get_video_array(vid_path)
    hold_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return raw_vid, hold_img, img_path, vid_path

def main(args):
    raw_vid, hold_img, img_path, vid_path = get_data(args)
    
    # process video, extract all necessary information
    raw_vid, climb_holds_used, holds, colors, results_arr, landmarks_arr, all_positions, sig_positions, significances = process_video(args.dir, raw_vid, hold_img, vid_path, img_path)
    print("Video frames: ", raw_vid.shape[0])
    print("Holds Used Frames: ", len(climb_holds_used))
    print("All Positions Frames: ", len(all_positions['left_hand']))
    print("Sig Position Frames: ", len(sig_positions['left_hand']))

    # image = cv2.imread(img_path)
    # for rect in holds:
    #     point1, point2 = rect[0], rect[1]
    #     cv2.rectangle(image, point1, point2, (0, 255, 0), 2)

    # cv2.imshow('Image with Multiple Rectangles', image)



    percent_complete = compute_percent_complete_color(holds, colors, all_positions)
    
    num_moves, move_holds_used, distinct_holds_used = get_num_moves(climb_holds_used, significances)
    print("Move Holds: ", len(move_holds_used))
    print("Distinct Holds: ", len(distinct_holds_used))

    route_color = getColorRoute(distinct_holds_used, holds, colors)
    hold_validity = getPercentHoldValidity(distinct_holds_used, colors, route_color)
    move_validity = getPercentMoveValidity(move_holds_used, colors, route_color)
    
    time_elapsed = compute_time_elapsed(raw_vid, holds, all_positions=all_positions, fps=30)
    if time_elapsed == -1:
        raise Exception("Total Time Elapsed could Not Be Computed")
    total_distance = compute_total_distance_traveled(args.dir, sig_positions)
    
    # print("")
    # print("% Complete: ", percent_complete)
    # print("# of Moves Taken: ", num_moves)
    # print("Hold Validity: {:.2f}".format(hold_validity * 100))
    # print("Move Validity: {:.2f}".format(move_validity * 100))
    # print("Climb Duration: {} sec".format(time_elapsed))
    # print("Total Distance Climbed: {:.2f} px".format(total_distance))

    print("")
    print("% 本条线路完成度: ", percent_complete)
    print("# 运动员岩点行进间移动次数: ", num_moves)
    print("Control的岩点: {:.2f}%".format(hold_validity * 100))
    print("有效移动: {:.2f}%".format(move_validity * 100))
    print("持续攀爬时间: {} sec".format(time_elapsed))
    print("总爬升距离: {:.2f} px".format(total_distance))

    with open(os.path.join(args.dir, 'report.json'), 'w') as f:
        to_write = {
            'pc_complete': percent_complete,
            'num_moves': num_moves,
            'hold_validity': np.around(hold_validity * 100, decimals=2),
            'move_validity': np.around(move_validity * 100, decimals=2),
            'climb_duration': time_elapsed,
            'total_distance': np.around(total_distance, decimals=2)
        }
        json.dump(to_write, f)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    main(args)