import cv2
import numpy as np 



def compute_xor(mask, ground_truth):
    """
    Compares two binary masks of the same size.
    Returns the number of differing pixels.
    """
    xor_result = cv2.bitwise_xor(mask, ground_truth)
    differing_pixels = cv2.countNonZero(xor_result)
    return differing_pixels



def compute_background_model(video_path, num_frames=100, sample_interval=5):
    #finds avg background image over num_frames number of frames. 
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    saved_frames = 0
    while saved_frames < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_interval == 0:
            frames.append(frame.astype(np.float32))
            saved_frames += 1
        frame_count += 1
    cap.release()
    background = np.mean(frames, axis=0).astype(np.uint8)
    return background



def remove_small_blobs(mask, min_area=500): 
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    refined_mask = np.zeros(mask.shape, dtype=np.uint8)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            refined_mask[labels == label] = 255
    return refined_mask


def background_subtraction(frame, background, kernel_size = (5,5), hsv_thresholds = (30,30,30)): 

    # convertto HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    background_hsv = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
    
    if frame_hsv.shape != background_hsv.shape:
        background_hsv = cv2.resize(background_hsv, (frame_hsv.shape[1], frame_hsv.shape[0]))
    
    diff = cv2.absdiff(frame_hsv, background_hsv)
    value_diffs = cv2.split(diff)
    
    masks = [cv2.threshold(value_diffs[i], hsv_thresholds[i], 255, cv2.THRESH_BINARY)[1] for i in range(3)]
    h_mask, s_mask, v_mask = masks

    # combine masks
    foreground_mask = cv2.bitwise_or(cv2.bitwise_or(h_mask, s_mask), v_mask)
    
    # refinements
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size) #change to (7,7) to fill bigger white holes (but might lose fine details)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel)
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_DILATE, kernel)
    foreground_mask = remove_small_blobs(foreground_mask, min_area=500)

    return foreground_mask



def get_background_models(cams):
    #return all 4 cam background models
    background_models = {}
    for cam in cams: 
        background_vid_path = f"data/{cam}/background.avi"   
        background = compute_background_model(background_vid_path, num_frames=30, sample_interval=1)
        background_models[cam] = background            
    return background_models 


def process_video(cam, background_model):
    #run the model on a video capture to extract fore/background and show
    video = cv2.VideoCapture(f"data/{cam}/video.avi")
    rand_frame=0
    while True:
        ret, frame = video.read()
        if not ret:
            break 
        
        mask = background_subtraction(frame, background_model, kernel_size=(7,7), hsv_thresholds= (30, 30, 30))

        if rand_frame==10:
            print("Saving image...")
            cv2.imwrite(f"data/{cam}/sampled_image.png", frame)
            cv2.imwrite(f"data/{cam}/foreground_mask.png", mask)
        rand_frame+=1

        cv2.imshow(f"{cam} - Input Frame", frame)
        cv2.imshow(f"{cam} - Background Model", background_model)
        cv2.imshow(f"{cam} - Foreground Mask", mask) 
        cv2.imshow(f"{cam} - Foreground Only", cv2.bitwise_and(frame, frame, mask=mask))

        #press q to exit
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break




def optimize_thresholds(background, frame, ground_truth_mask):
    best_score = float('inf')
    best_thresholds = (0,0,0)
    for h_t in range(0, 101, 5):
        for s_t in range(0, 101, 5):
            for v_t in range(0, 101, 5):
                mask = background_subtraction(frame, background, kernel_size=(7,7), hsv_thresholds=(h_t, s_t, v_t))
                # Compute XOR or F1 score
                score = compute_xor(mask, ground_truth_mask)
                if score < best_score:
                    best_score = score
                    best_thresholds = (h_t, s_t, v_t)
    return best_thresholds, best_score


def test_optimization():
    cams = [f"cam{i}" for i in range(1, 5)]
    for cam in cams:
        background = compute_background_model(f"data/{cam}/background.avi", num_frames=30, sample_interval=1) 

        frame = cv2.imread(f"data/{cam}/sampled_image.png")
        ground_truth_mask = cv2.imread(f"data/{cam}/cleaned_mask.png", cv2.IMREAD_GRAYSCALE)
        
        best_thresholds, best_score = optimize_thresholds(frame, background, ground_truth_mask)
        print(f"Camera {cam}: Best thresholds: {best_thresholds} with XOR error: {best_score}")
        
        best_mask = background_subtraction(frame, background, kernel_size=(7,7), hsv_thresholds=best_thresholds)
        
        cv2.imshow(f"{cam} - Ground Truth Mask", ground_truth_mask)
        cv2.imshow(f"{cam} - Optimized Mask", best_mask)

    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main(): 

    cams = [f"cam{i}" for i in range(1, 5)]
    background_models = get_background_models(cams)
    for cam in cams:
        process_video(cam, background_models[cam])
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    # test_optimization()



#grid search kernel size (5,5 7,7)
#gaussian distrib for each pixel
