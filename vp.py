import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time
import mediapipe as mp
import HandTrackingModule as htm
import math
import random
import time
brushThickness=5
eraserThickness=150
drawColor =(17,100,255)
brushThicknessSelect=750
brushColorSelect=(400,15)
selectedShape=-1 #0==> circle, #1==>Rectangle #2==>line #3==> selected area erase


cap=cv2.VideoCapture(0)
detector =htm.handDetector(detectionCon=0.85)
ptip=[(0,0),(0,0),(0,0),(0,0),(0,0)]
tip=[]
imgCanvasList=[np.zeros((720,1280,3),np.uint8)]
imgCanvas = imgCanvasList[0]
canvasNo=0
changeCanvas=True
cropped=0
topX=math.inf
topY=math.inf
bottomX=-1
bottomY=-1
#################
while True:
    #capture image
    success,img =cap.read()
    img = cv2.resize(img,(1280,720))     


    #Find Hand landmarks
    img =detector.findHands(img)

    

    lmList = detector.findPosition(img , draw=False)
    if len(lmList)!=0:
        #finger-tips
        tip=[lmList[4][1:],lmList[8][1:],lmList[12][1:],lmList[16][1:],lmList[20][1:]]
        if ptip==[(0,0),(0,0),(0,0),(0,0),(0,0)]:
            ptip=tip    
        #Check which fingers are up
        fingers = detector.fingersUp()


        #Controll mode--> three fingers are up
        if fingers == [0,1,1,1,0]:
            if selectedShape==0:
                cv2.circle(img,tip[1],brushThickness,drawColor,1)
            elif selectedShape==1:
                l=brushThickness*2
                w=brushThickness
                cv2.rectangle(img, (tip[1][0]-l,tip[1][1]-w),(tip[1][0]+l,tip[1][1]+w), drawColor, 3)    
            else:     
                cv2.circle(img,tip[2],brushThickness,drawColor,cv2.FILLED)
            #thickness controll
            if (tip[2][1]<150) and (700<=tip[2][0]<=800):
                brushThicknessSelect=tip[2][0]
                brushThickness=max(2,min(math.ceil(abs(brushThicknessSelect-700)/2),40))
            
            #color controll
            elif 400<=tip[2][0]<=655 and 15<=tip[2][1]<=115:
                brushColorSelect=tip[2]
                drawColor=(tip[2][0]-400,(tip[2][1]-15)*2,255)

            #circle select
            elif 920<=tip[2][0]<=980 and tip[2][1]<80:
                selectedShape=0    
            #rectangle select
            elif 1000<=tip[2][0]<=1080 and tip[2][1]<80:
                selectedShape=1  
            #pencil select
            elif 1090<=tip[2][0]<=1170 and tip[2][1]<80:
                selectedShape=2 

            #selected area erase    
            elif 280<=tip[2][0]<=380 and tip[2][1]<80:
                selectedShape=3     
            #next canvas   
            elif changeCanvas==True and 850<=tip[2][0]<=890 and 30<=tip[2][1]<=50:
                canvasNo=canvasNo+1
                if canvasNo>=len(imgCanvasList):
                    imgCanvasList.append(np.zeros((720,1280,3),np.uint8))
                imgCanvas=imgCanvasList[canvasNo]
                changeCanvas=False
            #previous canvas   
            elif changeCanvas==True and 850<=tip[2][0]<=890 and 80<=tip[2][1]<=100:
                canvasNo=((canvasNo-1)%len(imgCanvasList))
                imgCanvas=imgCanvasList[canvasNo]
                changeCanvas=False                     

        #Erase mode--> two fingers are upif 400<=tip[2][0]<=655 and 15<=tip[2][1]<=115:
        elif fingers == [0,1,1,0,0]:
            src=(tip[1][0], tip[1][1] - 25)
            dest=(tip[2][0], tip[2][1] + 25)
            if selectedShape==3:
                src=(topX,topY)
                dest=(bottomX,bottomY)
                selectedShape=-1
                plt.imsave('cropped'+str(cropped)+'.png',imgCanvas)
            cv2.rectangle(img, src, dest, (255,255,255), cv2.FILLED)
            cv2.rectangle(imgCanvas, src, dest, (0,0,0), cv2.FILLED)
            

        #Drawing mode--> Index finger is up
        elif fingers == [0,1,0,0,0]:
            changeCanvas=True
            #cicle
            if selectedShape==0:
                cv2.circle(img,tip[1],brushThickness,drawColor,1)
                cv2.circle(imgCanvas,tip[1],brushThickness*2,drawColor,1)
                selectedShape=-1
            #rectangle    
            elif selectedShape==1:
                l=brushThickness*2
                w=brushThickness
                cv2.rectangle(img, (tip[1][0]-l,tip[1][1]-w),(tip[1][0]+l,tip[1][1]+w), drawColor, 3)
                cv2.rectangle(imgCanvas, (tip[1][0]-l,tip[1][1]-w),(tip[1][0]+l,tip[1][1]+w), drawColor, 3)
                selectedShape=-1
            #pencil    
            elif selectedShape==2:     
                cv2.line(img, ptip[1], tip[1], drawColor , brushThickness)
                cv2.line(imgCanvas, ptip[1], tip[1], drawColor , brushThickness)
            #select area to erase
            elif selectedShape==3:  
                destX=(ptip[1][0]+tip[1][0])//2
                destY=(ptip[1][1]+tip[1][1])//2   
                cv2.line(img, ptip[1],(destX,destY), (255,255,255) , 1)
                cv2.line(imgCanvas,ptip[1],(destX,destY), (255,255,255) , 1)
                topX=min(topX,tip[1][0]) 
                topY=min(topY,tip[1][1])
                bottomX=max(bottomX,tip[1][0]) 
                bottomY=max(bottomY,tip[1][1])       

        ptip=tip


    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)




    #brush thickness controller
    cv2.line(img, (700,50), (brushThicknessSelect,50), (61, 29, 0) , 10)
    cv2.line(img, (brushThicknessSelect,50), (800,50), (60, 10, 100) , 10)
    cv2.circle(img,(brushThicknessSelect,50),15,(155, 152, 229),cv2.FILLED)
    #brush color controller
    for i in range(100):
        for j in range(255):
            img[i+15][j+400]=[j,i*2,255]
    cv2.circle(img,brushColorSelect,8,(1, 0, 0),cv2.FILLED)
    #Circle
    cv2.circle(img,(950,50),30,drawColor,3) 
    #rectangle
    cv2.rectangle(img, (1000,20), (1080,80), drawColor, 3)
    #pencil
    cv2.line(img, (1100,20), (1120,80), drawColor , 2) 
    cv2.line(img, (1120,80), (1140,20), drawColor , 2) 
    cv2.line(img, (1140,20), (1180,80), drawColor , 2)
    #next canvas
    cv2.line(img, (850,30), (890,40), drawColor , 2)
    cv2.line(img, (890,40), (850,50), drawColor , 2)
    #previous canvas
    cv2.line(img, (850,90), (890,80), drawColor , 2)
    cv2.line(img, (850,90), (890,100), drawColor , 2)
    #show canvas no    
    cv2.putText(img, str(int(canvasNo+1)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3) 
    #eraze selected area
    for i in range(280,380,10):
        cv2.line(img, (i,20), (i+4,20), drawColor , 2)
        cv2.line(img, (i,80), (i+4,80), drawColor , 2)
    for i in range(20,80,10):
        cv2.line(img, (280,i), (280,i+4), drawColor , 2)
        cv2.line(img, (380,i), (380,i+4), drawColor , 2)    

    if selectedShape==0:                
        cv2.line(img, (920,90), (980,90),drawColor ,4)
    elif selectedShape==1:                
        cv2.line(img, (1000,90), (1080,90),drawColor ,4)
    elif selectedShape==2:                
        cv2.line(img, (1110,90), (1180,90),drawColor ,4)
    elif selectedShape==3:                
        cv2.line(img, (280,90), (380,90),drawColor ,4)    

               
    
    cv2.imshow("Draw With Hand Gesture",img)
    if int(cv2.waitKey(1))!=-1:
        canvasCnt=0
        for c in imgCanvasList:
            plt.imsave('canvas'+str(canvasCnt)+'.png', c)
            canvasCnt=canvasCnt+1
        break
