import argparse
import datetime
import imutils
import time
import cv2
import numpy
import pdb
import math
import pykalman


class Projector():
        def __init__(self,boardCorners,screenWidth,boardWidth,boardLength,zoneFrame):
                assert(boardCorners[0][1]==boardCorners[1][1])
                assert(boardCorners[2][1]==boardCorners[3][1])

                self.boardCorners = boardCorners
                self.r0 = float(boardCorners[2][0]-boardCorners[3][0])/2
                self.r1 = float(boardCorners[1][0]-boardCorners[0][0])/2
                self.h = float(boardCorners[3][1] - boardCorners[0][1])
                self.bottom = float(boardCorners[3][1])
                self.screenWidth=float(screenWidth)
                self.boardWidth = boardWidth
                self.boardLength = boardLength

                self.zoneFrame=zoneFrame
                self.displayHeight,self.displayWidth,channels = zoneFrame.shape
                self.img=numpy.zeros((self.displayHeight,self.displayWidth,3))

                self.prevInterPosition=None


        def computeFlatCoordinates(self,p):
                x,y=p
                x2 = float(x)-self.screenWidth/2
                y2 = self.bottom-float(y)

                b=self.r1/(self.r0-self.r1)
                a=self.r0*b
                c=self.r0*self.r1*self.h/((self.r0-self.r1)*(self.r0-self.r1))

                z1=c/(-y2+c/b)-b
                x1=x2*(c/(-y2+c/b))/a/2+0.5
                return(x1*self.boardWidth,z1*self.boardLength)

        def convertToPixel(self,p):
                x, y=p
                return (int(x*self.displayHeight/self.boardLength),int(y*self.displayWidth/self.boardWidth))

        def clearDisplay(self):
                self.img=numpy.zeros((self.displayHeight,self.displayWidth,3))

        def plot(self,points,color=(255,255,255),size=2):
                for point in points:
                    cv2.circle(self.img,self.convertToPixel(point),size,color)

        def getIntersectedZones(self,p):
                x, y=p
                x2,y2 = self.convertToPixel((x,y))
                if self.prevInterPosition is None:
                        self.prevInterPosition=(x2,y2)
                        return []
                else:
                        (x1,y1) = self.prevInterPosition
                d = math.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))
                if (d==0):
                        return []
                dx = (x2-x1)/d
                dy = (y2-y1)/d

                x,y = x1,y1
                zones = []
                previousZone=0
                for i in range(0,int(d)):
                        zone = int(self.zoneFrame[int(y)][int(x)])
                        #print(x,y,zone)
                        if (zone!=previousZone) and (zone!=0):
                                zones+=[zone]
                        previousZone=zone
                        x += dx
                        y += dy

                self.prevInterPosition=(x2,y2)
                return(zones)


        def display(self):
                cv2.imshow("2D View",self.img)





class BallDetector():

        mask=None
        background=None

        def __init__(self,projector,threshold=100):
                self.projector=projector
                self.threshold=threshold


        def initialProcessing(self,frame):
                assert(not(self.mask is None))
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv = cv2.GaussianBlur(hsv, (5,5),0)
                hsv = cv2.bitwise_and(hsv,self.mask)
                return(hsv)


        def updateMask(self,ppframe):
                if self.mask is None:
                        height, width, channels = ppframe.shape
                        self.mask = numpy.zeros((height,width,3), numpy.uint8)
                        cv2.fillConvexPoly(self.mask,numpy.array(self.projector.boardCorners),cv2.cv.RGB(255,255,255))

        def updateBackground(self,ppframe):
                if self.background is None:
                        self.background = cv2.split(ppframe)




        def getCandidatesFromPPFrame(self,ppframe):
                hsvSplit = cv2.split(ppframe);

                #gray = cv2.bitwise_and(gray,mask)
                # compute the absolute difference between the current frame and

                frameDelta = cv2.absdiff(hsvSplit[0],self.background[0]) + cv2.absdiff(hsvSplit[1],self.background[1]) + cv2.absdiff(hsvSplit[2],self.background[2])

                #thresh = cv2.threshold(frameDelta, 160, 0, cv2.THRESH_TOZERO_INV)[1]
                thresh = cv2.threshold(frameDelta, self.threshold, 255, cv2.THRESH_BINARY)[1]


                # dilate the thresholded image to fill in holes, then find contours
                # on thresholded image
                thresh = cv2.dilate(thresh, None, iterations=2)

                cv2.imshow("Thresh",thresh)

                (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
                cnts = [c for c in cnts if cv2.contourArea(c) > args["min_area"]]

                cnts = sorted(cnts,key=lambda c: -cv2.contourArea(c))

                def getCenter(c):
                        (x, y, w, h) = cv2.boundingRect(c)
                        return((x+w/2,y+h/2))

                coordinates = [ getCenter(c) for c in cnts ]
                #cv2.imshow("Thresh", thresh)
                #cv2.imshow("Frame Delta", frameDelta)


                return(cnts,coordinates)

        def getCandidates(self,frame):
                self.updateMask(frame)
                ppframe = self.initialProcessing(frame)
                self.updateBackground(ppframe)
                return(self.getCandidatesFromPPFrame(ppframe))

class OriginalFrameDisplayer():
        def __init__(self,projector):
                self.projector = projector

        def display(self,frame,cnts,text):
                cv2.polylines(frame,[numpy.array(self.projector.boardCorners)],True,(0, 255, 0), 1)
                for c in cnts:
                        # if the contour is too small, ignore it

                        # compute the bounding box for the contour, draw it on the frame,
                        # and update the text
                        (x, y, w, h) = cv2.boundingRect(c)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)



                # draw the text and timestamp on the frame
                cv2.putText(frame, text,
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

                # show the frame and record if the user presses a key
                cv2.imshow("Camera", frame)


class Predictor():
        trackIndex=0

        obs=[]
        pred1=[]
        pred2=[]

        def observe(self,p):
                x, y=p
                xp,yp=None,None
                if (self.trackIndex>=3):
                        vxinit=self.obs[1][0]-self.obs[0][0]
                        vyinit=self.obs[1][1]-self.obs[0][1]
                        xinit=self.obs[0][0]
                        yinit=self.obs[0][1]
                        initcovariance=1.0e-3*numpy.eye(6)
                        transistionCov=1.0e-4*numpy.eye(6)
                        observationCov=1.0e-5*numpy.eye(2)
                        initstate=[xinit,yinit,vxinit,vyinit,0,0.1]
                        Transition_Matrix=[[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,0.9,0,1,0],[0,0,0,0.9,0,1],[0,0,0,0,1,0],[0,0,0,0,0,1]]
                        Observation_Matrix=[[1,0,0,0,0,0],[0,1,0,0,0,0]]
                        kf=pykalman.KalmanFilter(
                                transition_matrices=Transition_Matrix,
                                observation_matrices =Observation_Matrix,
                                initial_state_mean=initstate,
                                initial_state_covariance=initcovariance,
                                transition_covariance=transistionCov,
                                observation_covariance=observationCov)

                        def computeAngle(x1,y1,x2,y2):
                            cosa = (x1*x2 + y1*y2) / (math.sqrt(x1*x1 + y1*y1) * math.sqrt(x2*x2 + y2*y2))
                            return(math.acos(cosa))

                        vx,vy=x-self.obs[-1][0],y-self.obs[-1][1]

                        obsPlusMasked1=numpy.ma.asarray(self.obs[0:(len(self.obs)-1)]+[[0,0],[0,0]])
                        obsPlusMasked1[-1]=numpy.ma.masked
                        obsPlusMasked1[-2]=numpy.ma.masked

                        (pred1,cov1)=kf.smooth(obsPlusMasked1)

                        vxp1,vyp1=pred1[-1][0]-pred1[-2][0],pred1[-1][1]-pred1[-2][1]

                        a1=computeAngle(vx,vy,vxp1,vyp1)

                        obsPlusMasked2=numpy.ma.asarray(self.obs+[[0,0]])
                        obsPlusMasked2[-1]=numpy.ma.masked
                        (pred2,cov2)=kf.smooth(obsPlusMasked2)

                        vxp2,vyp2=pred2[-1][0]-pred2[-2][0],pred2[-1][1]-pred2[-2][1]

                        a2=computeAngle(vx,vy,vxp2,vyp2)

                        self.pred1=zip(pred2[:,0],pred1[:,1])
                        self.pred2=zip(pred2[:,0],pred2[:,1])

                        print("angle:",a1,a2,"vitesse:",math.sqrt(vx*vx + vy*vy))
                        vtm2=math.sqrt(math.pow(self.obs[-3][0]-self.obs[-2][0],2) +\
                                        math.pow(self.obs[-3][1]-self.obs[-2][1],2))
                        if (a2>0.5 or a1>1) and (vtm2>1):
                            self.trackIndex=1
                            #self.pred=[]
                            self.obs=[[x,y]]
                            print("bing",x,y)
                            return((xp,yp))

                self.obs += [[x,y]]
                self.trackIndex+=1
                return((xp,yp))

class ParameterProvider():
        ramp1ComboMultiplier = 2
        ramp1Score = 1000


class PinballLogicProcessor():

        STATE_NO_ZONE = 0
        STATE_ZONE_RAMP_1 = 1

        STATE_GAME_NORMAL = 2
        STATE_GAME_COMBO = 3

        EVENT_ZONES = 4
        EVENT_RAMP_1_SUCCESSFUL = 5
        EVENT_RAMP_1_START = 6

        parameterProvider=ParameterProvider()

        zoneState = STATE_NO_ZONE
        zoneStateTime = 0
        gameState = STATE_GAME_NORMAL
        gameStateTime = 0



        def processZone(self, time,zone):
                def setZoneState(state):
                        self.zoneState=state
                        self.zoneStateTime=time

                if (self.state==self.NO_ZONE):
                        if zone==1:
                                setZoneState(self.STATE_ZONE_RAMP_1)
                                self.processEvent(self.EVENT_RAMP_1_START)
                if (self.state==self.ZONE_RAMP_1):
                        if zone==2:
                                setZoneState(self.STATE_NO_ZONE)
                                self.processEvent(self.EVENT_RAMP_1_SUCCESSFUL)
                        elif time-self.zoneStateTime>5:
                                setZoneState(self.STATE_NO_ZONE)

        def processEvent(self, event):
                def setGameState(state,param):
                        self.gameState=state
                        self.gameStateTime=time
                        self.gameStateParameter=param

                if self.gameState==self.STATE_GAME_NORMAL:
                        if event==self.EVENT_RAMP_1_SUCCESSFUL:
                                self.score+=self.parameterProvider.ramp1Score
                                screen.updateScore(self.score)
                                setGameState(self.STATE_GAME_COMBO,self.parameterProvider.ramp1ComboMultiplier)

                if self.gameState==self.STATE_GAME_COMBO:
                        if event==self.EVENT_RAMP_1_SUCCESSFUL:
                                self.score+=self.parameterProvider.ramp1Score*self.gameStateParameter
                                screen.updateScore(self.score)
                                setGameState(self.STATE_GAME_COMBO,self.gameStateParameter*self.parameterProvider.ramp1ComboMultiplier())
                        elif time>self.gameStateTime+self.parameters.ramp1ComboTime:
                                setGameState(self.STATE_GAME_NORMAL)


        def update(self, time,zones=[],eventType=None,param=None):
                self.processZone(time,zones)
                if (eventType==self.EVENT_ZONES):
                        zones=param
                        for zone in zones:
                                self.processZone(time,zone)




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=20, help="minimum area size")
ap.add_argument("-f", "--frame",type=int, default=1,help="go to frame")

args = vars(ap.parse_args())

def main(args):
        # if the video argument is None, then we are reading from webcam
        if args.get("video", None) is None:
                camera = cv2.VideoCapture(0)
                time.sleep(0.25)

        # otherwise, we are reading from a video file
        else:
                camera = cv2.VideoCapture(args["video"])
                camera.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,160)




        background=None
        waitTime = 1


        mask = None

        zoneFrame = numpy.zeros((600,400,1),numpy.uint8)
        cv2.fillConvexPoly(zoneFrame,numpy.array([[330,0],[370,0],[370,70],[330,70]]),1)
        cv2.fillConvexPoly(zoneFrame,numpy.array([[0,0],[20,0],[20,40],[0,40]]),2)
        projector = Projector([ [130,30], [350,30], [500,250],[0,250] ], 500,40.0,60.0,zoneFrame)



        ballDetector = BallDetector(projector)
        originalFrameDisplayer = OriginalFrameDisplayer(projector)
        predictor = Predictor()
        (grabbed, frame) = camera.read()
        if (args["frame"]>1):
                camera.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,args["frame"])

        frame = imutils.resize(frame, width=500)




        # loop over the frames of the video
        while True:
                # grab the current frame and initialize the occupied/unoccupied
                # text
                (grabbed, frame) = camera.read()
                # if the frame could not be grabbed, then we have reached the end
                # of the video
                if not grabbed:
                        break
                frame = imutils.resize(frame, width=500)

                # resize the frame, convert it to grayscale, and blur it

                cnts,coordinates = ballDetector.getCandidates(frame)
                if (len(coordinates)>0)                                                                                                                                                                                                                                                                                    :
                        (x,y) = projector.computeFlatCoordinates(coordinates[0])

                        (xp,yp) = predictor.observe((x,y))
                        projector.clearDisplay()
                        projector.plot(predictor.obs,color=(255,0,0))
                        projector.plot(predictor.pred1,color=(0,0,255),size=1)
                        projector.plot(predictor.pred2,color=(0,255,0),size=1)
                        projector.display()
                        zones=projector.getIntersectedZones((x,y))

                # loop over the contours
                #cv2.imshow("Hue", cv2.split(hsv)[2])

                originalFrameDisplayer.display(frame,cnts,str(camera.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)))

                key = cv2.waitKey(waitTime) & 0xFF

                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                        break

                if key == ord("p"):
                        if waitTime == 1:
                                waitTime = 100000
                        else:
                                waitTime = 1



        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()
print(args)
main(args)
