import mediapipe as mp
from scipy import spatial
import cv2
from utils.hold_utils import get_holds_and_colors
from utils.pc_complete_utils import get_holds_used
from utils.pose_utils import get_significant_frames_motion_graph
import matplotlib.colors as mcolors

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

def plot_image(img, results, cx, cy, elapsed_time, unique_con_idx, holds, colors):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
#   cv2.circle(img, (cx, cy), 5, (255,0, 150), cv2.FILLED)
#   cv2.putText(img, str(int(elapsed_time)), (50, 50), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 3)
  #cv2_imshow(img) # if using colab
  
    if unique_con_idx: 
        selected_rock = []
        selected_rock_color = []
        for i in range(len(unique_con_idx)):
            selected_rock.append(holds[unique_con_idx[i]])
            selected_rock_color.append(colors[unique_con_idx[i]])
        for rect, color_name in zip(selected_rock, selected_rock_color):
            point1, point2 = rect[0], rect[1]
            # 获取颜色的RGB值，并转换为BGR
            color = mcolors.to_rgb(color_name)
            color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
            cv2.rectangle(img, point1, point2, color_bgr, 2)
    
    cv2.imshow('image', img)
    cv2.waitKey(1)

    return img

def check_similarity(list1, list2):
    """Returns similarity between two lists of landmark coordinates"""
    result = 1 - spatial.distance.cosine(list1, list2)
    return result

def rock_contact(limbs, holds):
    contact_holds_idx = []
    jx, jy = limbs
    for i in range(len(holds)):
        h_xmin, h_ymin = holds[i][0]
        h_xmax, h_ymax = holds[i][1]
        if jx <= h_xmax and jx >= h_xmin and jy <= h_ymax and jy >= h_ymin:
            contact_holds_idx.append(i)

    return contact_holds_idx



def get_video_pose(dir, vid_arr, vid_path, holds, colors):
    """
    Returns all pose information from a video
    vid_arr: np array

    returns:
        all_results: list of mediapipe results
        all_landmarks: list(list) each sublist contains all landmarks for a specific frame
        dict_coordinates: keys are joints, values are list(tuple) 
                            each tuple is x,y coordinate for that joint at a specific frame
    """
    pose = mp.solutions.pose.Pose()

    cap = cv2.VideoCapture(vid_path)
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    fps = cap.get(cv2.CAP_PROP_FPS)
    elapsed_time = frame_number / fps
    output_img_list = []

    dict_coordinates = {'left_hand': [], 'right_hand': [], 'left_hip': [], 'right_hip': [], 'left_leg': [], 'right_leg': []}
    all_landmarks = []
    all_results = []
    frames = []
    significances = []
    for i in range(vid_arr.shape[0]):
        img = vid_arr[i]
        results = pose.process(img)
        
        if results.pose_landmarks is not None:
            frames.append(i)
            lm_list = []
            for id, lm in enumerate(results.pose_landmarks.landmark):  
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lm_list.append(cx)
                lm_list.append(cy)
        
            left_hand_con_idx = rock_contact((lm_list[38], lm_list[39]), holds)
            right_hand_con_idx = rock_contact((lm_list[40], lm_list[41]), holds)
            left_leg_con_idx = rock_contact((lm_list[62], lm_list[63]), holds)
            right_leg_con_idx = rock_contact((lm_list[64], lm_list[65]), holds)

            con_idx = left_hand_con_idx + right_hand_con_idx + left_leg_con_idx + right_leg_con_idx
            combined_set = set(con_idx)
            unique_con_idx = list(combined_set)

            output_img_list.append(plot_image(img, results, cx, cy, elapsed_time, unique_con_idx, holds, colors))
    
            # prev = lm_list

            all_landmarks.append(lm_list)
            all_results.append(results)
            dict_coordinates['left_hand'].append((lm_list[38], lm_list[39])) #left_index - x, y 
            dict_coordinates['right_hand'].append((lm_list[40], lm_list[41])) #right_index - x, y
            dict_coordinates['left_hip'].append((lm_list[46], lm_list[47])) #left_hip - x, y
            dict_coordinates['right_hip'].append((lm_list[48], lm_list[49])) #right_hip - x, y
            dict_coordinates['left_leg'].append((lm_list[62], lm_list[63])) #left_foot - x, y
            dict_coordinates['right_leg'].append((lm_list[64], lm_list[65])) #right_foot - x, y

    # output a video consisting of just the processed frames
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float 'width'
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float 'height'
    img_size = (width, height)
    fps_output = 20
    output_video = cv2.VideoWriter(dir + '/pose_estimation.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps_output, img_size)
    
    for i in range(len(output_img_list)):
        output_video.write(output_img_list[i])
    
    output_video.release()
    print('Output Video Released: ', fps_output, 'fps')
    
    significances = get_significant_frames_motion_graph(dir, all_landmarks)

    return frames, all_results, all_landmarks, dict_coordinates, significances

def process_video(dir, video, hold_img, vid_path, img_path):
    # joint_positions stores all landmarks for all frames -- even insignificant frames
    # significances denotes whether the frame was significant

    holds, colors = get_holds_and_colors(hold_img)

    image = cv2.imread(img_path)
    count_color = 0
    for rect in holds:
        point1, point2 = rect[0], rect[1]
        color = mcolors.to_rgb(colors[count_color])
        count_color = count_color+1
        color_bgr = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))
        cv2.rectangle(image, point1, point2, color_bgr, 2)

    cv2.imshow('Image with Multiple Rectangles', image)

    frames, results_arr, landmarks_arr, all_positions, significances = get_video_pose(dir, video, vid_path, holds, colors)
    video = video.take(frames, axis=0)


    sig_positions = {'left_hand': [], 'right_hand': [], 'left_hip': [], 'right_hip': [], 'left_leg': [], 'right_leg': []}
    for i in range(len(significances)):
        if significances[i]:
            sig_positions['left_hand'].append(all_positions['left_hand'][i])
            sig_positions['right_hand'].append(all_positions['right_hand'][i])
            sig_positions['left_hip'].append(all_positions['left_hip'][i])
            sig_positions['right_hip'].append(all_positions['right_hip'][i])
            sig_positions['left_leg'].append(all_positions['left_leg'][i])
            sig_positions['right_leg'].append(all_positions['right_leg'][i])

    # num moves should be sum(significances) - 1

    # holds, colors, wall_model = predict_holds_colors(hold_img, wall_model=None)    
    
    climb_holds_used = get_holds_used(holds, all_positions)

    return video, climb_holds_used, holds, colors, results_arr, landmarks_arr, all_positions, sig_positions, significances
