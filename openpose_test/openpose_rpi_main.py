import cv2
import time
import numpy as np
import pickle
import requests
import json

MODE = "COCO"
filename = 'finalized_model_v1.sav'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))


if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose/mpi/pose_deploy_linevec.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]


camera = cv2.VideoCapture(0)

time_per_frame = 3
lastUpdate = time.time()
last_ball_time = 0


def classify_pose(points, img_width, img_height):
    POSSIBLE_POSES = ["sitting", "standing", "fallen"]
    print(points)
    nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear = points

    y_threshold = 75
    pose_classification = "none"
    predicted_probability = "none"

    # print(nose)
    # print(right_shoulder)
    # print(left_shoulder)

    flat_list = []
    for coord in points:
        if coord:
            flat_list.append(coord[0])
            flat_list.append(coord[1])
        else:
            flat_list.append(-1) # Missing X
            flat_list.append(-1) # Missing Y

    flat_list = np.array([flat_list])
    if len(np.unique(flat_list)) == 1:
        pose_classification = "none"
    else:
        predicted_pose = loaded_model.predict(flat_list)
        predicted_probability = loaded_model.predict_proba(flat_list)
        print(predicted_pose)
        print(predicted_probability)

        if len(predicted_pose) > 0 and len(predicted_probability) > 0 and predicted_probability[0][predicted_pose[0]] > 0.3:
            pose_classification = POSSIBLE_POSES[predicted_pose[0]]
            predicted_probability = predicted_probability[0][predicted_pose[0]]
        else:
            pose_classification = "none"

    # Determine which direction to move the servo
    action = 0
    threshold = 50

    if nose:
        n_diff = nose[0] - (img_width / 2.0)
        if abs(n_diff) < threshold:
            action = 0
        else:
            action = int(65.0 * n_diff / float(img_width))
    elif neck:
        n_diff = neck[0] - (img_width / 2.0)
        if abs(n_diff) < threshold:
            action = 0
        else:
            action = int(65.0 * n_diff / float(img_width))
    
    print(img_width, img_height)
    print(nose)
    print(neck)
    print(action)

    # print(predicted_probability[0][predicted_pose[0]])
    # print(pose_classification)

    if action > 15:
        action = 15
    elif action < -15:
        action = -15

    return pose_classification, predicted_probability, action


class_data = []
def collect_pose_data(points):
    # nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear = points
    # flat_list = [item for sublist in points for item in sublist]
    flat_list = []
    for coord in points:
        if coord:
            flat_list.append(coord[0])
            flat_list.append(coord[1])
        else:
            flat_list.append(-1) # Missing X
            flat_list.append(-1) # Missing Y

    class_data.append(flat_list)


# Main Embedded Raspberry Pi Loop
# while True:
# i = 0
# num_data_per_class = 110
# time.sleep(3)
true_action = 0
while True:
    if time.time() - lastUpdate < time_per_frame:
        # Ideally we would not sleep, but set a timer that interrupts instead of busy waiting
        time.sleep(time_per_frame / 5)
        continue
    lastUpdate = time.time()
    (grabbed, frame) = camera.read()
    # frame = cv2.imread("kireet.jpg")
    frameCopy = np.copy(frame)

    cv2.imwrite('test-output.jpg', frame)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

    t = time.time()
    # input image dimensions for the network
    inWidth = 368
    inHeight = 368
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False)

    net.setInput(inpBlob)

    output = net.forward()
    print("time taken by network : {:.3f}".format(time.time() - t))

    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
        
        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold : 
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    # Draw Skeleton
    # paired_points = []
    # for pair in POSE_PAIRS:
    #     partA = pair[0]
    #     partB = pair[1]

    #     if points[partA] and points[partB]:
    #         cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
    #         cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    possible_pose, pose_probability, action_difference = classify_pose(points, frameHeight, frameWidth)

    #nose, neck, right_shoulder, right_elbow, right_wrist, left_shoulder, left_elbow, left_wrist, right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle, right_eye, left_eye, right_ear, left_ear = points
    #cv2.circle(frame, (int(nose[0]), int(nose[1]-100)), 20, (0, 0, 0), thickness=-1, lineType=cv2.FILLED)
    #cv2.imwrite('test-output-123.jpg', frame)

    # cv2.imshow('Output-Keypoints', frameCopy)
    # cv2.imshow('Output-Skeleton', frame)

    # cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    # cv2.imwrite('Output-Skeleton.jpg', frame)
    data = {
      'predicted_pose': possible_pose,
      'pose_probability': str(pose_probability)
    }   
    url = "http://127.0.0.1:5000"
    # url = "https://caramel-logic-231220.appspot.com"
    # response = requests.post('https://caramel-logic-231220.appspot.com', data=data)
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    response = requests.post(url + '/pose_status', data=json.dumps(data), headers=headers)
    true_action += action_difference

    if true_action > 90:
        true_action = 90
    elif true_action < -90:
        true_action = -90

    if true_action != 0:
        payload = { 'rotate': true_action }
        response = requests.get(url + '/rotate_servo', params=payload)

    # print(response)

    print("Total time taken : {:.3f}".format(time.time() - t))
    print("Expected pose: " + possible_pose)




# np.save('none.npy', np.array(class_data))
# save the class_data as a numpy array

# cv2.waitKey(0)
