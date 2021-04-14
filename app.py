from flask import Flask
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import keyboard


test=10 
app = Flask(__name__)

@app.route('/')
def index():
    return "Hello, World!"


@app.route('/video')
def video():
    #load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # initialize the video stream and allow the cammera sensor to warmup
    print("[INFO] starting video stream...")
    #vs = VideoStream(src=0).start()
    #time.sleep(2.0)
    cap=cv2.VideoCapture("video.mp4")

    if(cap.isOpened()== False):
        print("Error Opening Video Stream Or File")



    while(cap.isOpened()) :
        #grab the frame from the threaded video stream and resize it
        #to have a maximum width of 400 pixels
        #frame = vs.read()
        ret,frame = cap.read()

        frame = imutils.resize(frame, width=1000)
        #grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))
        # pass the blob through the network and obtain the detections and
        # predictions

        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < 0.5:
                continue

            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int") 
             
            #draw the bounding box of the face along with the associated
            #probability
            text ="Reenal" #"{:.2f}%".format(confidence * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        global test 
        if test == 1:
            break       

        # if the `q` key was pressed, break from the loop
        
        '''if key == ord("q"):
            break'''
    
   
    cv2.destroyAllWindows()
    #cap.stop()
    

    #matplotlib.pyplot.switch_backend('Agg')     
    return "video"

@app.route('/off')
def off():
    global test
    test=1
    
    return('off')
    #keyboard.wait('esc')
    '''if keyboard.is_pressed('a'):
        return('a key has been pressed')
    else:
        return('key has been not pressed')'''
            
   


if __name__ == '__main__':
    app.run(debug=True)

