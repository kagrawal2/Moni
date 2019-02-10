import requests
import json
import cv2

# addr = 'http://localhost:5000'
addr = "https://caramel-logic-231220.appspot.com"
test_url = addr + '/get_pose'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

img = cv2.imread('kireet.jpg')
# encode image as jpeg
_, img_encoded = cv2.imencode('.jpg', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tobytes(), headers=headers)
# decode response
print(json.loads(response.text))
