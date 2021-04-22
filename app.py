#credits to: https://www.youtube.com/watch?v=ON_JubFRw8M
#credits to: https://www.youtube.com/watch?v=MAjbzx2zq-c
import cv2
import numpy as np
from flask import Flask, render_template, Response
import time
app = Flask(__name__)
sub = cv2.createBackgroundSubtractorMOG2()


@app.route('/')
def index():
    return render_template('index.html')

def gen():
    webCamFeed = True
    cap = cv2.VideoCapture(0)
    #cap.set(10,160)
    heightImg = 800
    widthImg = 1200

    count = 0

    while True:
        imgBlank = np.zeros((heightImg, widthImg, 3), np.uint8)
        
        if webCamFeed:success, img = cap.read()
        img = cv2.resize(img, (0, 0), None, 1, 1)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
        imgThreshold = cv2.Canny(imgGray, 250, 200)
        
        imgContours = img.copy()
        imgBigContour = img.copy()
        contours, hierarchy = cv2.findContours(imgThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgContours, contours, -1, (0,255,0), 10)
        
        biggest = np.array([])
        max_area = 0
        #Loop door alle contours
        for i in contours:
            #Maak hier een area van
            area = cv2.contourArea(i)
            #Check of dit echt de grootste area is
            if area > 700:
                peri = cv2.arcLength(i, True)
                approx = cv2.approxPolyDP(i, 0.02 * peri, True)
                if area > max_area and len(approx) == 4:
                    biggest = approx
                    max_area = area
        if biggest.size != 0:
            myPoints = biggest.reshape((4,2))
            myPointsNew = np.zeros((4,1,2), dtype = np.int32)
            add = myPoints.sum(1)
            myPointsNew[0] = myPoints[np.argmin(add)]
            myPointsNew[3] = myPoints[np.argmax(add)]
            diff = np.diff(myPoints, axis=1)
            myPointsNew[1] = myPoints[np.argmin(diff)]
            myPointsNew[2] = myPoints[np.argmax(diff)]
            biggest = myPointsNew
            cv2.drawContours(imgBigContour, biggest, -1, (0,255,0), 20)
            pts1 = np.float32(biggest)
            pts2 = np.float32([[0,0],[widthImg,0],[0,heightImg], [widthImg,heightImg]])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            imgBigContour = cv2.warpPerspective(img,matrix, (widthImg, heightImg))
    #         cv2.imshow('Document scanner', imgWarpColored)
        
        #cv2.imshow('Document scanner', imgBigContour)
        frame = cv2.imencode('.jpg', imgBigContour)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.1)
        
        #cv2.imshow('Document scanner', imgThreshold)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')