from flask import Flask, request, Response, jsonify, render_template
import cv2
import requests
import time
import numpy as np

app = Flask(__name__)

app.config['POSE'] = "none"
app.config['PROB'] = "0"
app.config['SIT_COUNT'] = 0
app.config['STAND_COUNT'] = 0
app.config['FALL_COUNT'] = 0


@app.route("/")
def template_test():
    return render_template('template.html', my_string="Wheeeee!", my_list=[0,1,2,3,4,5])

@app.route("/pose_status", methods=['POST'])
def update_current_pose():
    pose = request.json['predicted_pose']
    prob = request.json['pose_probability']
    app.config['POSE'] = pose
    app.config['PROB'] = prob
    if pose == 'sitting':
        app.config['SIT_COUNT'] += 1
    elif pose == 'standing':
        app.config['STAND_COUNT'] += 1
    elif pose == 'fallen':
        app.config['FALL_COUNT'] += 1

    ret = {
        'pose': app.config['POSE'],
        'prob': app.config['PROB'],
        'sit_count': app.config['SIT_COUNT'],
        'stand_count': app.config['STAND_COUNT'],
        'fall_count': app.config['FALL_COUNT']
    }
    return jsonify(ret)

@app.route("/reset_counts", methods=['GET'])
def reset_counts():
    app.config['SIT_COUNT'] = 0
    app.config['STAND_COUNT'] = 0
    app.config['FALL_COUNT'] = 0
    ret = {
        'pose': app.config['POSE'],
        'prob': app.config['PROB'],
        'sit_count': app.config['SIT_COUNT'],
        'stand_count': app.config['STAND_COUNT'],
        'fall_count': app.config['FALL_COUNT']
    }
    return jsonify(ret)


@app.route("/get_pose", methods=['GET'])
def get_current_pose():
    ret = {
        'pose': app.config['POSE'],
        'prob': app.config['PROB'],
        'sit_count': app.config['SIT_COUNT'],
        'stand_count': app.config['STAND_COUNT'],
        'fall_count': app.config['FALL_COUNT']
    }
    return jsonify(ret)


@app.route("/rotate_servo", methods=['GET', 'POST'])
def rotate_servo():
    # ?rotate_val = some-value
    rotate_val = request.args.get('rotate')
    rotate_val = min(90, int(rotate_val))
    rotate_val = max(-90, int(rotate_val))

    particle_cloud_api = "https://api.particle.io/v1/devica6301ad5ddc428"
    access_token = "650d89d752326cced1f93db0d9a"

    data = {
      'access_token': '650d89d752326cced1f93db0d9a6301ad5ddc428',
      'args': str(rotate_val)
    }   
    response = requests.post('https://api.particle.io/v1/devices/1e003e000f47373333353132/rotateOdisus', data=data)
    print(response)
    return 'Sent rotate command ' + str(rotate_val)


@app.route("/get_pose", methods=['POST'])
def get_pose_from_image():

    MODE = "COCO"

    if MODE is "COCO":
        protoFile = "static/pose/coco/pose_deploy_linevec.prototxt"
        weightsFile = "static/pose/coco/pose_iter_440000.caffemodel"
        nPoints = 18
        POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

    elif MODE is "MPI" :
        protoFile = "pose/mpi/pose_deploy_linevec.prototxt"
        weightsFile = "pose/mpi/pose_iter_160000.caffemodel"
        nPoints = 15
        POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

    r = request
    # convert string of image data to uint8
    # nparr = np.fromstring(r.data, np.uint8)
    nparr = np.asarray(bytearray(r.data), dtype=np.uint8)
    # decode image
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # frame = cv2.imread("kireet.jpg")

    frameCopy = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    # do some fancy processing here....
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
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
            cv2.circle(frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


    # build a response dict to send back to client
    response = {'message': 'image received. size={}x{}'.format(frame.shape[1], frame.shape[0])
                }
    # encode response using jsonpickle
    response_pickled = jsonpickle.encode(response)

    # cv2.imwrite('Output-Keypoints.jpg', frameCopy)
    # cv2.imwrite('Output-Skeleton.jpg', frame)

    return Response(response=response_pickled, status=200, mimetype="application/json")

    # cv2.imshow('Output-Keypoints', frameCopy)
    # cv2.imshow('Output-Skeleton', frame)


    # print("Total time taken : {:.3f}".format(time.time() - t))
    # cv2.waitKey(0)

if __name__ == "__main__":
    app.run(threaded=False, debug=True)