# 신호등 파라메터 설정                                                     <ts.py>


# Library Import ============================================================================
import os                   # File 읽기기(Read) 쓰기(Write)와 관련된 라이브러리
import sys                  # 커맨드 라인 처리를 위한 라이브러리
import pwd                  # 현재 사용자의 홈 디렉토리와 관련된 정보를 얻거나 변경하는 라이브러리
import numpy as np          # 이미지 배열 생성(OpenCV 에서 사용) 라이브러리
import cv2                  # 영상처리(OpenCV) 라이브러리 
import pickle               # 파이썬 객채 저장과 읽기 라이브러리
from math import sin, cos, atan2, sqrt, pi
from operator import itemgetter

# Constant ----------------------------------------------------------------------------------
RED     =   0,  0,255                  # Red
GREEN   =   0,255,  0                  # Green
BLUE    = 255,  0,  0                  # Blue
MAGENTA = 255,  0,255                  # Magenta(Pink)
CYAN    = 255,255,  0                  # Cyan(Sky Blue)
YELLOW  =   0,255,255                  # Yellow
WHITE   = 255,255,255                  # White
GRAY    =  64, 64, 64                  # Gray
BLACK   =   0,  0,  0                  # Black
#--------------------------------------------------------------------------------------------
def trafficSign(frame,                 # Y480 x X800 전체 LCD 스크린
                VIEW_FLAG,             # 신호등 데이터 디스프레이 T: 신호등 설정 모드 F: 주행 모드
                TSWIN_XL,              # 카메라 영역 X 축 Left  0~640
                TSWIN_YU,              # 카메라 영역 Y 축 Up    0~480
                TSWIN_XR,              # 카메라 영역 X 축 Right 0~640
                TSWIN_YD,              # 카메라 영역 Y 축 Down  0~480
                ts_rad_min,            # 신호등 최소 반지름      >8
                ts_rad_max,            # 신호등 최대 반지름      <40
                HSV_hueLeadGREEN,      # Green 시작각
                HSV_hueTailGREEN,      # Green 종료각
                HSV_hueLeadRED,        # Red 시작각
                HSV_hueTailRED,        # Red 종료각
                HSV_saturationGREEN,   # Green Saturation
                HSV_saturationRED,     # Red Saturation
                HSV_valueGREEN,        # Green Value
                HSV_valueRED,          # Red Value
                gTsV,                  # Green Threshold
                rTsV                   # Red Threshold
                ):
    '''
    신호등을 검출하여 적색 신호등일 때 'R' 녹색 신호등일 때 'G' 그리고 검출되지 않았을 때 ' '를 반환
    자율 주행(VIEW_FLAG = False), 신호등 설정(VIEW_FLAG = True) 
    frame(LCD Window) 크기는 480 x 800 
    '''
    # 카메라 이미지는 중앙에서 480 x 640 이 신호등 감지 영역이며 스크린 영역(480x800) 중앙에 위치한다.
    tsAreaX1 = TSWIN_XL + 80
    tsAreaY1 = TSWIN_YU 
    tsAreaX2 = TSWIN_XR + 80
    tsAreaY2 = TSWIN_YD

    trafficLight = ' '                 # 신호등 검출 없을 때 반환 값(' ')으로 초기화 한다.

    grayimg = frame[tsAreaY1:tsAreaY2,tsAreaX1:tsAreaX2]  # 신호등 검출 영역
    tsbgray = cv2.cvtColor(grayimg,cv2.COLOR_BGR2GRAY)      # 컬러에서 그레이 색으로 전환
    # 원 검출 ------------------------------------------------------------------------------
    circles = cv2.HoughCircles(         # 신호등 원 검출
             tsbgray,                  # 그레이 스케일 신호등 이미지 사용
             method=cv2.HOUGH_GRADIENT, # 원 검출 방법
             dp=1,                     # The inverse ratio of resolution.
             minDist=60,               # 검출된 원 중심점 간의 최소거리
             param1=100,               # Canny 에지 검출을 위한 상위 스레시홀드 값
             param2=30,                # Threshold for center detection
             minRadius=ts_rad_min,     # 원(신호등)의 최소 반지름
             maxRadius=ts_rad_max      # 원(신호등)의 최대 반지름
             )
    if circles is not None:
        #print(circles); print('count=',len(circles[0]))  # 객체 리스트 프린트 참고용
        # [[[419.5 196.5  27.4]           검출된 원은 리스트 형이며 소수점 데이터로 반환됨
        #   [414.5  75.5  27.3]           3 개의 원이 검출  
        #   [419.5 135.5  27.4]]]         원 중심의 X 좌표, Y 좌표, 반지름
        circles = np.uint16(np.around(circles))   # 정수 리스트로 변환한다.
        # 
        totalR = 0                     # HSV 검사 조건을 통과한 Red 픽셀 누적 합
        totalG = 0                     # HSV 검사 조건을 통과한 Green 픽셀 누적 합
        tsData = [[0,0,0]]             # 리스트

        for number, i in enumerate(circles[0,:], start=1):
            # 검출된 원의 실측 반지름에 의한 사각형 영역 복사            
            if (tsAreaY1+i[1]-i[2])<tsAreaY1 or (tsAreaY1+i[1]+i[2])>tsAreaY2 or\
                (tsAreaX1+i[0]-i[2])<tsAreaX1 or (tsAreaX1+i[0]+i[2])>tsAreaX2:
                continue
            
            tsa = frame[tsAreaY1+i[1]-i[2]:tsAreaY1+i[1]+i[2],
                        tsAreaX1+i[0]-i[2]:tsAreaX1+i[0]+i[2]].copy()
            # BGR 색상 좌표계를 HSV 색상 좌표계로 변환
            #print('X:',i[0],'Y:',i[1],'R:',i[2]) # 적절한 이미지 크기가 아니라면 무시하는 코드 추가 !!!
            hsv = cv2.cvtColor(tsa, cv2.COLOR_BGR2HSV)
            # 녹색 신호등 처리 
            upper_color = np.array([HSV_hueLeadGREEN,255,255])     # Upper
            lower_color = np.array([HSV_hueTailGREEN,HSV_saturationGREEN,HSV_valueGREEN]) # Lower
            # Threshold the HSV image to get only selected colors
            mask = cv2.inRange(hsv, lower_color, upper_color)
            # Green 신호용 mask 와 신호등 원래의 이미지를 And 연상하여 mask 영역만 통과시킨다.
            resG = cv2.bitwise_and(tsa,tsa,mask=mask)
            # Green 신호등에서 R, G, B 성분 히스토그램
            histGb = cv2.calcHist([resG],[0],None,[256],[0,256])  # Green 에서 Blue 성분 히스토그램
            histGg = cv2.calcHist([resG],[1],None,[256],[0,256])  # Green 에서 Green 성분 히스토그램
            histGr = cv2.calcHist([resG],[2],None,[256],[0,256])  # Green 에서 Red 성분 히스토그램
            # 적색 신호등 처리
            if HSV_hueLeadRED > HSV_hueTailRED:                  # 같은 구간에 있을 때
                upper_color = np.array([HSV_hueLeadRED,255,255]) # Upper
                lower_color = np.array([HSV_hueTailRED,HSV_saturationRED,HSV_valueRED]) # Lower
                mask = cv2.inRange(hsv, lower_color, upper_color)
            else:
                # for Red-P Area (Red 원의 위쪽)
                upper_color = np.array([HSV_hueLeadRED,255,255]) # Upper
                lower_color = np.array([0,HSV_saturationRED,HSV_valueRED])  # Lower
                maskP = cv2.inRange(hsv, lower_color, upper_color)
                # for Red-N Area (Red 원의 아래쪽)
                upper_color = np.array([179,255,255])            # Upper
                lower_color = np.array([HSV_hueTailRED,HSV_saturationRED,HSV_valueRED])  # Lower
                maskN = cv2.inRange(hsv, lower_color, upper_color)
                # 위쪽 영역의 mask 와 아래쪽 영역의 mask 를 Or 연산하여 합친다.
                mask = cv2.bitwise_or(maskP, maskN)   
            # Red 신호용 mask 와 신호등 원래의 이미지를 And 연상하여 mask 영역만 통과시킨다.
            resR = cv2.bitwise_and(tsa,tsa,mask=mask)
            # Red 신호등에서 R, G, B 성분 히스토그램
            histRb = cv2.calcHist([resR],[0],None,[256],[0,256])  # Red 에서 Blue 성분 히스토그램
            histRg = cv2.calcHist([resR],[1],None,[256],[0,256])  # Red 에서 Green 성분 히스토그램
            histRr = cv2.calcHist([resR],[2],None,[256],[0,256])  # Red 에서 Red 성분 히스토그램
            # 신호등에서 Red 와 Green 을 분해하여 추출한 이미지를 LCD에 디스프레이
            u = int(i[2])    # 검출된 신호등의 반지름 
            v = u*2          # 검출된 신호등의 지름 
            try:
                frame[40-u:40-u+v,760-u:760-u+v] = resR
                cv2.rectangle(frame,(760-u,40-u),(760-u+v,40-u+v),RED,1) # Red 사각형
                frame[40-u:40-u+v,40-u:40-u+v] = resG
                cv2.rectangle(frame,(40-u,40-u),(40-u+v,40-u+v),GREEN,1) # Green 사각형
            except:
                print('resR/G Copy Error')

            histGb[0] = 0; histGg[0] = 0; histGr[0] = 0 # 녹색 신호등의 검정색 픽셀 제거
            histRb[0] = 0; histRg[0] = 0; histRr[0] = 0 # 적색 신호등의 검정색 픽셀 제거

            if VIEW_FLAG:                               # 신호등 최소 크기 원, 최대 크기 원, 실측 원 표시
                cv2.putText(frame,f'{number}',(i[0]+tsAreaX1-7, i[1]+tsAreaY1+7),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
                cv2.circle(frame,(i[0]+tsAreaX1, i[1]+tsAreaY1),i[2],(YELLOW),1)       # 검출된 원 반지름
            # 신호등 설정 모드일 때 신호등 히스토그램 그래프를 디스프레이
            if VIEW_FLAG:
                # Red 신호등 B,G,R 성분 최대 표시 원
                rbM = int(np.amax(histRb))        # 히스토그램 데이터중 최대값
                rbmW =np.where(histRb==rbM)       # 최대값의 위치(배열 형)
                rbmWV = int(rbmW[0][0])           # 최대값의 위치(스칼라 형)
                rgM = int(np.amax(histRg))        # 히스토그램 데이터중 최대값
                rgmW =np.where(histRg==rgM)       # 최대값의 위치(배열 형)
                rgmWV = int(rgmW[0][0])           # 최대값의 위치(스칼라 형)
                rrM = int(np.amax(histRr))        # 히스토그램 데이터중 최대값
                rrmW =np.where(histRr==rrM)       # 최대값의 위치(배열 형)
                rrmWV = int(rrmW[0][0])           # 최대값의 위치(스칼라 형)
                # Green 신호등 B,G,R 성분 최대 표시 원
                gbM = int(np.amax(histGb))        # 히스토그램 데이터중 최대값
                gbmW =np.where(histGb==gbM)       # 최대값의 위치(배열 형)
                gbmWV = int(gbmW[0][0])           # 최대값의 위치(스칼라 형)
                ggM = int(np.amax(histGg))        # 히스토그램 데이터중 최대값
                ggmW =np.where(histGg==ggM)       # 최대값의 위치(배열 형)
                ggmWV = int(ggmW[0][0])           # 최대값의 위치(스칼라 형)
                grM = int(np.amax(histGr))        # 히스토그램 데이터중 최대값
                grmW =np.where(histGr==grM)       # 최대값의 위치(배열 형)
                grmWV = int(grmW[0][0])           # 최대값의 위치(스칼라 형)
                
                if rbM > 0: cv2.circle(frame,(533+rbmWV,454-rbM),4,YELLOW,1)
                if rgM > 0: cv2.circle(frame,(533+rgmWV,454-rgM),4,YELLOW,1)
                if rrM > 0: cv2.circle(frame,(533+rrmWV,454-rrM),4,YELLOW,1)
                if gbM > 0: cv2.circle(frame,(10+gbmWV,454-gbM),4,YELLOW,1)
                if ggM > 0: cv2.circle(frame,(10+ggmWV,454-ggM),4,YELLOW,1)
                if grM > 0: cv2.circle(frame,(10+grmWV,454-grM),4,YELLOW,1)
                
                # Red 신호등 히스토그램  X 시작: 533:533+256, Y:453~
                for i in range(0,256):       # Red 신호등에서 Blue 성분
                    k = int(histRb[i])
                    if k: cv2.line(frame,(533+i,453),(533+i,453-k),(BLUE),1)
                for i in range(0,256):       # Red 신호등에서 Green 성분
                    k = int(histRg[i])
                    if k: cv2.line(frame,(533+i,453),(533+i,453-k),(GREEN),1)
                for i in range(0,256):       # Red 신호등에서 Red 성분
                    k = int(histRr[i])
                    if k: cv2.line(frame,(533+i,453),(533+i,453-k),(RED),1)
                # Green 신호등 히스토그램 X 시작: 10:10+256, Y:453~
                for i in range(0,256):       # Green 신호등에서 Blue 성분
                    k = int(histGb[i])
                    if k: cv2.line(frame,(10+i,453),(10+i,453-k),(BLUE),1)
                for i in range(0,256):       # Green 신호등에서 Red 성분
                    k = int(histGr[i])
                    if k: cv2.line(frame,(10+i,453),(10+i,453-k),(RED),1)
                for i in range(0,256):       # Green 신호등에서 Green 성분
                    k = int(histGg[i])
                    if k: cv2.line(frame,(10+i,453),(10+i,453-k),(GREEN),1)

            r = int(np.sum(histRr))   # Red 신호등 히스토그램의 합
            g = int(np.sum(histGg))   # Green 신호등 히스토그램의 합
            t = [abs(g-r), g, r]      # [차이 절대값, 녹색, 적색]
            tsData.append(t)
            
        #print(tsData)
        tsSorted = sorted(tsData,key=itemgetter(0),reverse=True)  # 순서 정열
        #print(tsSorted); print()

        totalG = tsSorted[0][1]       # Green
        totalR = tsSorted[0][2]       # Red

        if totalG>totalR:             # Green 신호등
            if totalG > gTsV:
                trafficLight = 'G'
        else:                         # Red 신호등
            if totalR > rTsV:
                trafficLight = 'R'

        # 신호등 설정 모드에서 픽쎌 합산수, 포물선 그래프 디스프레이
        if VIEW_FLAG:
            cv2.ellipse(frame, (int((10+HSV_valueGREEN+10+256)/2),453),(int((256-HSV_valueGREEN)/2),int(totalG/10)),0,180,360,GREEN,2,cv2.LINE_AA)
            cv2.ellipse(frame, (int((533+HSV_valueRED+533+256)/2),453),(int((256-HSV_valueRED)/2),int(totalR/10)),0,180,360,RED,2,cv2.LINE_AA)
            cv2.putText(frame,str(totalG),(10+15+HSV_valueGREEN, 453-3-int(gTsV/10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
            cv2.putText(frame,str(totalR),(533+15+HSV_valueRED, 453-3-int(rTsV/10)),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
            xg1 = 10+HSV_valueGREEN; xg2 = 10+256; yg = 453-int(gTsV/10)
            cv2.line(frame,(xg1,yg),(xg2,yg),GREEN,1)
            xr1 = 533+HSV_valueRED; xr2 = 533+256; yr = 453-int(rTsV/10)
            cv2.line(frame,(xr1,yr),(xr2,yr),RED,1)
    else: 
    # 신호등 설정 모드에서 검출된 원이 없을 때 신호등 반지름 설정을 위한 참조 원(reference circle) 표시
        if VIEW_FLAG: 
            cv2.circle(frame,(400,60),ts_rad_min,WHITE,2)   # 신호등의 최소 조건 반지름 
            cv2.circle(frame,(400,60),ts_rad_max,CYAN,2)    # 신호등의 최대 조건 반지름 
    #----------------------------------------------------------------------------------------
    if trafficLight == 'R': frame[90:120,730:790] = RED    # 적색 신호등 검출  730:790
    if trafficLight == 'G': frame[90:120,10:70] = GREEN    # 녹색 신호등 검출  10:70
    #----------------------------------------------------------------------------------------
    return trafficLight    # 'G':녹색 신호등, 'R':적색 신호등, ' ':신호 없음
#--------------------------------------------------------------------------------------------


def controlMain(event,x,y,flags,param):
    '''
    마우스 이벤트 발생시 처리되는 함수 입니다.
    3 개의 마우스 버튼 Up/Down 이벤트와 X,Y 이동 이벤트를 처리합니다.
    '''
    global mouseX, mouseY, mouseButtenLeft, mouseButtenToggle
    
    mouseX = x; mouseY = y              # 마우스 좌표값 X, Y를 전역변수 mouseX, mouseY에 저장
    #print(mouseX, mouseY)

    if event == cv2.EVENT_LBUTTONDOWN:  # print('L-Down') # 마우스 왼쪽 버튼 이벤트 발생
        mouseButtenLeft = True
    if event == cv2.EVENT_RBUTTONDOWN:  # print('R-Down') # 마우스 오른쪽 버튼 이벤트 발생
        mouseButtenToggle = False
    '''
    if event == cv2.EVENT_LBUTTONUP:    # print('L-Up') # 마우스 왼쪽 버튼 이벤트 발생
    if event == cv2.EVENT_RBUTTONUP:    # print('R-Up') # 마우스 오른쪽 버튼 이벤트 발생
    if event == cv2.EVENT_MBUTTONDOWN:  # print('C-Down') # 마우스 가운데 버튼 이벤트 발생
    if event == cv2.EVENT_MBUTTONUP:    # print('C-Up') # 마우스 가운데 버튼 이벤트 발생
    '''
#--------------------------------------------------------------------------------------------
def angle360(centerX,centerY,mouseX,mouseY):
    '''
    좌표 값을 받아 360 분법의 각도 값을 구하고 정수형으로 반환합니다.
    '''
    d = -(180/pi)*atan2(mouseY-centerY,mouseX-centerX)
    if d<0: d = 360+d
    return (int(d))
#---------------------------------------------------------------------------------------------
def valueBox(hue):                  # HSV 이미지 생성
    hsv_image = np.zeros((20, 256, 3), dtype=np.uint8)
    hsv_image[..., 0] = hue             # hue 값             (0 번층 BGR 같으면 Blue 층)
    hsv_image[..., 1] = 255             # saturation 값 설정  (1 번층 BGR 같으면 Green 층)
    hsv_image[..., 2] = np.arange(256)  # value 값 설정       (2 번층 BGR 같으면 Red 층)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR) # HSV 이미지를 BGR로 변환
    return bgr_image
#---------------------------------------------------------------------------------------------
def main():

    global mouseButtenLeft, mouseButtenToggle

    DIRECTORY_FLAG = False                      # 신호등 데이터가 있으면 True
    imageDir = ''

    # Parameter ------------------------------------------------------------------------------
    TSWIN_XL = 0                                # [0]
    TSWIN_XR = 639                              # [1]
    TSWIN_YU = 30                               # [2]
    TSWIN_YD = 180                              # [3]
    maxRadius = 9                               # [4] 원 검출 조건(최대 반지름)
    minRadius = 8                               # [5] 원 검출 조건(최소 반지름)
    lineLeadAngGREEN = 200                      # [6] 시작(Lead) 라인 각도(/360)
    lineTailAngGREEN = 90                       # [7] 종료(Tail) 라인 각도(/360)
    lineLeadAngRED = 40                         # [8] 시작(Lead) 라인 각도(/360)
    lineTailAngRED = 300                        # [9] 종료(Tail) 라인 각도(/360)
    saturationHalfGREEN = 50                    # [10] Saturation 반지름 값 (/128)
    saturationHalfRED = 50                      # [11] Saturation 반지름 값 (/128)
    HSV_valueGREEN = 50                         # [12] 0~255 이며 작은수에서 가로가 크게 된다. OL:7
    HSV_valueRED = 50                           # [13] 0~255 이며 작은수에서 가로가 크게 된다. OL:6
    THRESH_GREEN = 50                           # [14] 0~255 이며 큰 수에서 세로가 높아 진다. OL:7
    THRESH_RED = 50                             # [15] 0~255 이며 큰 수에서 세로가 높아 진다. OL:6

    # Green ---------------------------------------------------------------------------------
    lineLeadGREENX = 0                          # 시작(Lead) 라인과 외곽 원 교차 X 좌표
    lineLeadGREENY = 0                          # 시작(Lead) 라인과 외곽 원 교차 Y 좌표
    lineTailGREENX = 0                          # 종료(Tail) 라인과 외곽 원 교차 X 좌표
    lineTailGREENY = 0                          # 종료(Tail) 라인과 외곽 원 교차 Y 좌표
    HSV_hueLeadGREEN = int(lineLeadAngGREEN/2)  # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
    HSV_hueTailGREEN = int(lineTailAngGREEN/2)  # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
    HSV_saturationGREEN = saturationHalfGREEN*2 # HSV 색상 좌표계에서 Saturation(/256)
    centerHueGREEN = 60
    # Red ------------------------------------------------------------------------------------
    lineLeadREDX = 0                            # 시작(Lead) 라인과 외곽 원 교차 X 좌표
    lineLeadREDY = 0                            # 시작(Lead) 라인과 외곽 원 교차 Y 좌표
    lineTailREDX = 0                            # 종료(Tail) 라인과 외곽 원 교차 X 좌표
    lineTailREDY = 0                            # 종료(Tail) 라인과 외곽 원 교차 Y 좌표
    HSV_hueLeadRED = int(lineLeadAngRED/2)      # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
    HSV_hueTailRED = int(lineTailAngRED/2)      # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
    HSV_saturationRED = saturationHalfRED*2     # HSV 색상 좌표계에서 Saturation(/256)
    centerHueRED = 0

    #-----------------------------------------------------------------------------------------
    # Object 리스트에 4 개의 객체 이름과 마우스와의 거리 정보(루트 연산 안한 값)가 들어 갑니다. 
    objectList = [['hueLeadRED',0],
                  ['hueTailRED',0],
                  ['saturationRED',0],
                  ['hueLeadGREEN',0],
                  ['hueTailGREEN',0],
                  ['saturationGREEN',0],
                  ['BOX_VAL_THRH_RED',0],
                  ['BOX_VAL_THRH_GREEN',0],
                  ['CIRCLE_MAX',0],
                  ['CIRCLE_MIN',0],
                  ['TSBOX_LU',0],
                  ['TSBOX_RD',0]]

    selectedObject = None     # 마우스 왼쪽 버튼에 의하여 선택된 객체

    #----------------------------------------------------------------------------------------
    if len(sys.argv) >= 2:
        t = sys.argv[1]         # 이미지 저장 디렉토리
        DIRECTORY_FLAG = True   # 디렉토리(프로젝트 이름)가 입력되었음 
        if not os.path.exists(t):
            print(f'이미지 저장 디렉터리 {t} 가 없습니다.')
            b = input('새로 생성 할까요? < Y: Yes>  <Any Key: No>')
            if b == 'y' or b == 'Y':
                os.mkdir(t)                # 디렉토리 생성 Sudo 모드로 생성
                # 생성한 디렉토리가 root 이므로 사용자 계정 pi 로 전환한다.
                # log in 사용자 계정을 가져온다.
                current_user = os.getlogin()
                user_name = current_user
                group_name = current_user
                # 사용자의 UID 가져오기
                uid = pwd.getpwnam(user_name).pw_uid
                # 그룹의 GID 가져오기
                gid = pwd.getpwnam(group_name).pw_gid
                # 파일의 소유자 변경
                os.chown(t, uid, gid)

                imageDir = t
            else:
                DIRECTORY_FLAG = False
        else:
            imageDir = t

    fn = './'+imageDir+'/_'+imageDir+'_TS.pickle'
    if DIRECTORY_FLAG and os.path.exists(fn):
        with open(fn, 'rb') as fr:
            d = pickle.load(fr)

            TSWIN_XL = d[0];             # 카메라 영역 X 축 Left  0~640 
            TSWIN_YU = d[1];             # 카메라 영역 Y 축 Up    0~480 
            TSWIN_XR = d[2];             # 카메라 영역 X 축 Right 0~640 
            TSWIN_YD = d[3];             # 카메라 영역 Y 축 Down  0~480
            minRadius = d[4];            # 신호등 최소 반지름      >8
            maxRadius = d[5];            # 신호등 최대 반지름      <40
            lineLeadAngGREEN = d[6];     # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
            lineTailAngGREEN = d[7];     # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
            lineLeadAngRED = d[8];       # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
            lineTailAngRED = d[9];       # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
            saturationHalfGREEN = d[10]; # Green Saturation
            saturationHalfRED = d[11];   # Red Saturation
            HSV_valueGREEN = d[12];      # Green Value
            HSV_valueRED = d[13];        # Red Value
            THRESH_GREEN = d[14];        # Green Threshold
            THRESH_RED = d[15];          # Red Threshold

        HSV_hueLeadGREEN = int(lineLeadAngGREEN/2)  # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
        HSV_hueTailGREEN = int(lineTailAngGREEN/2)  # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
        HSV_hueLeadRED = int(lineLeadAngRED/2)      # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
        HSV_hueTailRED = int(lineTailAngRED/2)      # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
        HSV_saturationGREEN = saturationHalfGREEN*2 # HSV 색상 좌표계에서 Saturation(/256)
        HSV_saturationRED = saturationHalfRED*2     # HSV 색상 좌표계에서 Saturation(/256)

        print('\n신호등 데이터 파일을 읽었습니다.:', fn, '\n')

    #-----------------------------------------------------------------------------------------
    while True:
        _, frame = cam.read()                 # 카메라 영상 1 프레임을 가져온다

        viewWin[:] = BLACK                    # 화면 초기화
        viewWin[:,80:80+640] = frame                                  # 카메라 영상 중앙 배치
        trafficSign(viewWin,                  # Y480 x X800 전체 LCD 스크린
                        True,                 # 신호등 데이터 디스프레이 T: 신호등 설정 모드 F: 주행 모드
                        TSWIN_XL,             # 카메라 영역 X 축 Left  0~640 
                        TSWIN_YU,             # 카메라 영역 Y 축 Up    0~480 
                        TSWIN_XR,             # 카메라 영역 X 축 Right 0~640 
                        TSWIN_YD,             # 카메라 영역 Y 축 Down  0~480
                        minRadius,            # 신호등 최소 반지름      >8
                        maxRadius,            # 신호등 최대 반지름      <40
                        HSV_hueLeadGREEN,     # Green 시작각
                        HSV_hueTailGREEN,     # Green 종료각
                        HSV_hueLeadRED,       # Red 시작각
                        HSV_hueTailRED,       # Red 종료각
                        HSV_saturationGREEN,  # Green Saturation
                        HSV_saturationRED,    # Red Saturation
                        HSV_valueGREEN,       # Green Value
                        HSV_valueRED,         # Red Value
                        THRESH_GREEN*10,      # Green Threshold
                        THRESH_RED*10         # Red Threshold
                        )

        # Hue 원 배치 -------------------------------------------------------------------------
        crop = viewWin[centerY-int(height/2):centerY-int(height/2)+height, 
                       centerX-int(width/2):centerX-int(width/2)+width]
        cv2.copyTo(logo, mask, crop)        #crop[mask > 0] = logo[mask > 0]  # 같은 동작 
        # Green Value Box 배치 ---------------------------------------------------------------
        viewWin[455:455+20, 10:10+256] = valueBox(centerHueGREEN)     
        cv2.rectangle(viewWin,(10,455),(10+256,455+20),GREEN,1)
        # Red Value Box 배치 -----------------------------------------------------------------
        viewWin[455:455+20, 533:533+256] = valueBox(centerHueRED)
        cv2.rectangle(viewWin,(533,455),(533+256,455+20),RED,1)
        # 신호등 영역 사각형 -------------------------------------------------------------------
        cv2.rectangle(viewWin,(TSWIN_XL+80,TSWIN_YU),(TSWIN_XR+80,TSWIN_YD), CYAN, 1)
        # 신호등 최대 크기 ---------------------------------------------------------------------
        cv2.circle(viewWin,(CIRCLE_X,CIRCLE_Y),maxRadius,YELLOW,1)
        cv2.circle(viewWin,(CIRCLE_X,CIRCLE_Y),minRadius,YELLOW,1)   # 신호등 최소 크기
        # RED 신호등 판단 BOX 디스프레이 --------------------------------------------------------
        cv2.rectangle(viewWin,(533+HSV_valueRED,453-THRESH_RED),(533+256,455),RED,1) # Red 신호등 판단 
        # GREEN 신호등 판단 BOX 디스프레이 ------------------------------------------------------
        cv2.rectangle(viewWin,(10+HSV_valueGREEN,453-THRESH_GREEN),(266,455),GREEN,1)  # Green 신호등 판단 

        # Hue 원에서 Red 객체 좌표값 구하기 -----------------------------------------------------
        HSV_hueLeadRED = int(lineLeadAngRED/2)      # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
        HSV_hueTailRED = int(lineTailAngRED/2)      # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
        # RED Lead Angle 객체 좌표 - 반지름 135 인 원과 Lead Angle 라인 교차 좌표값
        lineLeadREDX = centerX+int(135*cos(lineLeadAngRED*pi/180.0))             # OL:0
        lineLeadREDY = centerY-int(135*sin(lineLeadAngRED*pi/180.0)) 
        # RED Tail Angle 객체 좌표 - 반지름 135 인 원과 Tail Angle 라인 교차 좌표값 ----------------
        lineTailREDX = centerX+int(135*cos(lineTailAngRED*pi/180.0))             # OL:1
        lineTailREDY = centerY-int(135*sin(lineTailAngRED*pi/180.0))
        # RED SaturationHalf 을 반지름으로 원과 Lead각 교차 좌표값 --------------------------------
        lineLeadSatREDX = centerX+int(saturationHalfRED*cos(lineLeadAngRED*pi/180.0))
        lineLeadSatREDY = centerY-int(saturationHalfRED*sin(lineLeadAngRED*pi/180.0)) 
        # RED SaturationHalf 을 반지름으로 원과 Tail각 교차 좌표값 --------------------------------
        lineTailSatREDX = centerX+int(saturationHalfRED*cos(lineTailAngRED*pi/180.0))
        lineTailSatREDY = centerY-int(saturationHalfRED*sin(lineTailAngRED*pi/180.0))
        # RED Saturation 객체 좌표 ------------------------------------------------------------
        if lineLeadAngRED >= lineTailAngRED:
            a = int((lineLeadAngRED+lineTailAngRED)/2)%360
        else:
            a = int((lineLeadAngRED+lineTailAngRED)/2+180)%360
        saturationREDX = centerX+int(saturationHalfRED*cos(a*pi/180.0))          # OL:2
        saturationREDY = centerY-int(saturationHalfRED*sin(a*pi/180.0)) 

        # Hue 원에서 Green 객체 좌표값 구하기 ---------------------------------------------------
        HSV_hueLeadGREEN = int(lineLeadAngGREEN/2)  # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
        HSV_hueTailGREEN = int(lineTailAngGREEN/2)  # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
        # GREEN Lead Angle 객체 좌표 - 반지름 135 인 원과 Lead Angle 라인 교차 좌표값 -------------
        lineLeadGREENX = centerX+int(135*cos(lineLeadAngGREEN*pi/180.0))         # OL:3
        lineLeadGREENY = centerY-int(135*sin(lineLeadAngGREEN*pi/180.0)) 
        # GREEN Tail Angle 객체 좌표 - 반지름 135 인 원과 Tail Angle 라인 교차 좌표값 -------------
        lineTailGREENX = centerX+int(135*cos(lineTailAngGREEN*pi/180.0))         # OL:4
        lineTailGREENY = centerY-int(135*sin(lineTailAngGREEN*pi/180.0))
        # GREEN SaturationHalf 을 반지름으로 원과 Lead각 교차 좌표값 -----------------------------
        lineLeadSatGREENX = centerX+int(saturationHalfGREEN*cos(lineLeadAngGREEN*pi/180.0))
        lineLeadSatGREENY = centerY-int(saturationHalfGREEN*sin(lineLeadAngGREEN*pi/180.0)) 
        # GREEN SaturationHalf 을 반지름으로 원과 Tail각 교차 좌표값 -----------------------------
        lineTailSatGREENX = centerX+int(saturationHalfGREEN*cos(lineTailAngGREEN*pi/180.0))
        lineTailSatGREENY = centerY-int(saturationHalfGREEN*sin(lineTailAngGREEN*pi/180.0))
        # GREEN Saturation 객체 좌표 ---------------------------------------------------------
        if lineLeadAngGREEN >= lineTailAngGREEN:
            a = ((lineLeadAngGREEN+lineTailAngGREEN)/2)%360
        else:
            a = ((lineLeadAngGREEN+lineTailAngGREEN)/2+180)%360
        saturationGREENX = centerX+int(saturationHalfGREEN*cos(a*pi/180.0))      # OL:5
        saturationGREENY = centerY-int(saturationHalfGREEN*sin(a*pi/180.0)) 

        # RED -------------------------------------------------------------------------------
        cv2.line(viewWin,(lineLeadSatREDX,lineLeadSatREDY),(lineLeadREDX,lineLeadREDY),BLACK,1,cv2.LINE_AA)
        cv2.line(viewWin,(lineTailSatREDX,lineTailSatREDY),(lineTailREDX,lineTailREDY),BLACK,1,cv2.LINE_AA)

        if lineLeadAngRED >= lineTailAngRED:                     # 두 각이 동일한 영역에 있을 때이며 외곽원 Green 
            centerHueRED = int((lineLeadAngRED+lineTailAngRED)/4)   # 두 각의 중간값(360도 원)각을 구하고 다시 180 원 값을 구한다.
        else:                                          # 두 각이 교차 영역에 있을 때이며 외곽원 Red
            centerHueRED = (int((lineLeadAngRED+lineTailAngRED)/4)+90)%180   # 두 각의 중간값(360도 원)각을 구하고 다시 180 원 값을 구한다.

        # RED Lead Angle 객체 위치 디스프레이 ---------------------------------------------------
        c = WHITE if selectedObject=='hueLeadRED' and mouseButtenToggle else RED
        cv2.circle(viewWin,(lineLeadREDX,lineLeadREDY),8,c,-1)
        # RED Tail Angle 객체 위치 디스프레이 ---------------------------------------------------
        c = WHITE if selectedObject=='hueTailRED' and mouseButtenToggle else RED
        cv2.circle(viewWin,(lineTailREDX,lineTailREDY),8,c,-1)
        # RED Saturation 객체 위치 디스프레이 ---------------------------------------------------
        c = WHITE if selectedObject=='saturationRED' and mouseButtenToggle else RED
        cv2.circle(viewWin,(saturationREDX,saturationREDY),8,c,-1)
        # RED box 객체 위치 디스프레이 ----------------------------------------------------------
        c = WHITE if selectedObject=='BOX_VAL_THRH_RED' and mouseButtenToggle else RED
        cv2.circle(viewWin,(533+HSV_valueRED,453-THRESH_RED),8,c,-1)

        # GREEN -----------------------------------------------------------------------------
        cv2.line(viewWin,(lineLeadSatGREENX,lineLeadSatGREENY),(lineLeadGREENX,lineLeadGREENY),BLACK,1,cv2.LINE_AA)
        cv2.line(viewWin,(lineTailSatGREENX,lineTailSatGREENY),(lineTailGREENX,lineTailGREENY),BLACK,1,cv2.LINE_AA)
        centerHueGREEN = int((lineLeadAngGREEN+lineTailAngGREEN)/4)   # 두 각의 중간값(360도 원)각을 구하고 다시 180 원 값을 구한다.
        # -----------------------------------------------------------------------------------

        # GREEN Lead Angle 객체 위치 디스프레이 -------------------------------------------------
        c = WHITE if selectedObject=='hueLeadGREEN' and mouseButtenToggle else GREEN
        cv2.circle(viewWin,(lineLeadGREENX,lineLeadGREENY),8,c,-1)
        # GREEN Tail Angle 객체 위치 디스프레이 -------------------------------------------------
        c = WHITE if selectedObject=='hueTailGREEN' and mouseButtenToggle else GREEN
        cv2.circle(viewWin,(lineTailGREENX,lineTailGREENY),8,c,-1)
        # GREEN Saturation 객체 위치 디스프레이 -------------------------------------------------
        c = WHITE if selectedObject=='saturationGREEN' and mouseButtenToggle else GREEN
        cv2.circle(viewWin,(saturationGREENX,saturationGREENY),8,c,-1)
        # GREEN box 객체 위치 디스프레이 --------------------------------------------------------
        c = WHITE if selectedObject=='BOX_VAL_THRH_GREEN' and mouseButtenToggle else GREEN
        cv2.circle(viewWin,(10+HSV_valueGREEN,453-THRESH_GREEN),8,c,-1)
        # 신호등 최대 크기 객체 디스프레이 --------------------------------------------------------
        c = WHITE if selectedObject=='CIRCLE_MAX' and mouseButtenToggle else CYAN
        cv2.circle(viewWin,(CIRCLE_X-maxRadius,CIRCLE_Y),8,c,-1)    # 최소원의 왼쪽에 객체 표시
        # 신호등 최소 크기 객체 디스프레이 --------------------------------------------------------
        c = WHITE if selectedObject=='CIRCLE_MIN' and mouseButtenToggle else YELLOW
        cv2.circle(viewWin,(CIRCLE_X+minRadius,CIRCLE_Y),8,c,-1)     # 최대원의 오른쪽에 객체 표시
        # 신호등 영역 LU 객체 디스프레이 ---------------------------------------------------------
        c = WHITE if selectedObject=='TSBOX_LU' and mouseButtenToggle else CYAN
        cv2.circle(viewWin,(80+TSWIN_XL,TSWIN_YU),8,c,-1)    # 신호등 영역 LU 객체 표시
        # 신호등 영역 RD 객체 표시 --------------------------------------------------------------
        c = WHITE if selectedObject=='TSBOX_RD' and mouseButtenToggle else CYAN
        cv2.circle(viewWin,(80+TSWIN_XR,TSWIN_YD),8,c,-1)    # 신호등 영역 RD 객체 표시

        # RED 다른 영역에서 Lead 각과 Tail 각이 존재할 수도 있다. ----------------------------------
        if lineLeadAngRED >= lineTailAngRED:                     # 두 각이 동일한 영역에 있을 때이며 외곽원 Green 
            centerHueRED = int((lineLeadAngRED+lineTailAngRED)/4)   # 두 각의 중간값(360도 원)각을 구하고 다시 180 원 값을 구한다.
            cv2.ellipse(viewWin,(centerX,centerY),(outCirDia,outCirDia),0,-lineLeadAngRED,-lineTailAngRED,RED,3,cv2.LINE_AA)
            cv2.ellipse(viewWin,(centerX,centerY),(saturationHalfRED,saturationHalfRED),0,-lineLeadAngRED,-lineTailAngRED,BLACK,1,cv2.LINE_AA)
        else:                                          # 두 각이 교차 영역에 있을 때이며 외곽원 Red
            centerHueRED = (int((lineLeadAngRED+lineTailAngRED)/4)+90)%180   # 두 각의 중간값(360도 원)각을 구하고 다시 180 원 값을 구한다.
            cv2.ellipse(viewWin,(centerX,centerY),(outCirDia,outCirDia),0,-lineLeadAngRED,0,RED,3,cv2.LINE_AA)
            cv2.ellipse(viewWin,(centerX,centerY),(saturationHalfRED,saturationHalfRED),0,-lineLeadAngRED,0,BLACK,1,cv2.LINE_AA)

            cv2.ellipse(viewWin,(centerX,centerY),(outCirDia,outCirDia),0,-lineTailAngRED,-360,RED,3,cv2.LINE_AA)
            cv2.ellipse(viewWin,(centerX,centerY),(saturationHalfRED,saturationHalfRED),0,-lineTailAngRED,-360,BLACK,1,cv2.LINE_AA)
        
        # GREEN Mask 동일 영역에서 Lead 각과 Tail 각이 존재 --------------------------------------
        centerHueGREEN = int((lineLeadAngGREEN+lineTailAngGREEN)/4)   # 두 각의 중간값(360도 원)각을 구하고 다시 180 원 값을 구한다.
        cv2.ellipse(viewWin,(centerX,centerY),(outCirDia,outCirDia),0,-lineLeadAngGREEN,-lineTailAngGREEN,GREEN,3,cv2.LINE_AA)
        cv2.ellipse(viewWin,(centerX,centerY),(saturationHalfGREEN,saturationHalfGREEN),0,-lineLeadAngGREEN,-lineTailAngGREEN,BLACK,1,cv2.LINE_AA)

        # 객체 각도/위치 텍스트 표시 --------------------------------------------------------------
        cv2.putText(viewWin,f'{HSV_hueLeadRED}',(lineLeadREDX,lineLeadREDY),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv2.putText(viewWin,f'{HSV_hueTailRED}',(lineTailREDX,lineTailREDY),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv2.putText(viewWin,f'{HSV_saturationRED}',(saturationREDX,saturationREDY),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,BLACK)
        cv2.putText(viewWin,f'{HSV_valueRED}',(533+HSV_valueRED,453),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv2.putText(viewWin,f'/{THRESH_RED*10}',(533+70+HSV_valueRED,453-3-THRESH_RED),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)

        cv2.putText(viewWin,f'{HSV_hueLeadGREEN}',(lineLeadGREENX,lineLeadGREENY),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv2.putText(viewWin,f'{HSV_hueTailGREEN}',(lineTailGREENX,lineTailGREENY),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv2.putText(viewWin,f'{HSV_saturationGREEN}',(saturationGREENX,saturationGREENY),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,BLACK)
        cv2.putText(viewWin,f'{HSV_valueGREEN}',(10+HSV_valueGREEN,453),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv2.putText(viewWin,f'/{THRESH_GREEN*10}',(10+70+HSV_valueGREEN,453-3-THRESH_GREEN),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)

        cv2.putText(viewWin,f'{TSWIN_XL},{TSWIN_YU}',(TSWIN_XL+80+10,TSWIN_YU+17),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv2.putText(viewWin,f'{TSWIN_XR},{TSWIN_YD}',(TSWIN_XR+80-95,TSWIN_YD-5),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)

        cv2.putText(viewWin,f'{maxRadius}',(CIRCLE_X-40-maxRadius,CIRCLE_Y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv2.putText(viewWin,f'{minRadius}',(CIRCLE_X+40+minRadius,CIRCLE_Y),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)

        # 마우스 좌표값과 가장 가까운 객체 거리 계산 ------------------------------------------------
        objectList[0][1] = (lineLeadREDX-mouseX)**2 + (lineLeadREDY-mouseY)**2          # Hue R. Lead
        objectList[1][1] = (lineTailREDX-mouseX)**2 + (lineTailREDY-mouseY)**2          # Hue R. Tail
        objectList[2][1] = (saturationREDX-mouseX)**2 + (saturationREDY-mouseY)**2      # Sat. R.
        objectList[3][1] = (lineLeadGREENX-mouseX)**2 + (lineLeadGREENY-mouseY)**2      # Hue G. Lead
        objectList[4][1] = (lineTailGREENX-mouseX)**2 + (lineTailGREENY-mouseY)**2      # Hue G. Tail
        objectList[5][1] = (saturationGREENX-mouseX)**2 + (saturationGREENY-mouseY)**2  # Sat. G.
        objectList[6][1] = (533+HSV_valueRED-mouseX)**2 + (453-THRESH_RED-mouseY)**2    # Val. R.
        objectList[7][1] = (10+HSV_valueGREEN-mouseX)**2 + (453-THRESH_GREEN-mouseY)**2 # Val. G.
        objectList[8][1] = (CIRCLE_X-maxRadius-mouseX)**2 + (CIRCLE_Y-mouseY)**2        # Cir. Max
        objectList[9][1] = (CIRCLE_X+minRadius-mouseX)**2 + (CIRCLE_Y-mouseY)**2        # Cir. Min
        objectList[10][1] = (80+TSWIN_XL-mouseX)**2 + (TSWIN_YU-mouseY)**2              # Win. LU
        objectList[11][1] = (80+TSWIN_XR-mouseX)**2 + (TSWIN_YD-mouseY)**2              # Win. RD
        #------------------------------------------------------------------------------------
        if mouseButtenLeft:
            selectedObject = sorted(objectList,key=itemgetter(1))[0][0] # 마우스와 최 단거리 객체
            mouseButtenToggle = not mouseButtenToggle
            mouseButtenLeft = False                                     # 원 슈트(One Shoot)
        # 선택한 객체의 동작 실현 ---------------------------------------------------------------
        if mouseButtenToggle:                             # 객체가 선택됨
            # Red ---------------------------------------------------------------------------
            if selectedObject=='saturationRED':           # HSV Saturation
                s = int(sqrt((mouseX-centerX)**2 + (mouseY-centerY)**2)) # circle 중심으로 부터 mouse 거리
                if 0<= s and s < 128:
                    saturationHalfRED = s
                    HSV_saturationRED = saturationHalfRED*2  # HSV Saturation은 0~255 까지
            #--------------------------------------------------------------------------------
            elif selectedObject=='hueLeadRED':            # RED 시작 각도 객체 
                a = angle360(centerX,centerY,mouseX,mouseY)  
                if a < 60 and a > lineTailAngRED:         # Case 1: 0<L,T<60
                    lineLeadAngRED = a
                elif lineTailAngRED > 240 and a > lineTailAngRED:   # Case 2: 240<L,T<360
                    lineLeadAngRED = a
                elif a < 60 and lineTailAngRED > 240:     # Case 3: L<60, T>240
                    lineLeadAngRED = a
            #--------------------------------------------------------------------------------
            elif selectedObject=='hueTailRED':            # RED 종료 각도 객체
                a = angle360(centerX,centerY,mouseX,mouseY)  
                if lineLeadAngRED < 60 and a < lineLeadAngRED:      # Case 1: 0<L,T<60
                    lineTailAngRED = a
                elif a > 240 and a < lineLeadAngRED:      # Case 2: 240<L,T<360
                    lineTailAngRED = a
                elif lineLeadAngRED < 60 and a > 240:     # Case 3: L<60, T>240
                    lineTailAngRED = a
            #--------------------------------------------------------------------------------
            elif selectedObject=='BOX_VAL_THRH_RED':
                a = mouseX - 529
                if 0 <= a < 256: HSV_valueRED = a 
                a = (453) - mouseY
                if 0 <= a < 256: THRESH_RED = a
            # Green -------------------------------------------------------------------------
            elif selectedObject=='saturationGREEN':         # HSV Saturation
                s = int(sqrt((mouseX-centerX)**2 + (mouseY-centerY)**2)) # circle 중심으로 부터 mouse 거리
                if 0<= s and s < 128:
                    saturationHalfGREEN = s
                    HSV_saturationGREEN = saturationHalfGREEN*2  # HSV Saturation은 0~255 까지
            #--------------------------------------------------------------------------------
            elif selectedObject=='hueLeadGREEN':            # 마우스의 위치(원 중심으로 부터)에따른 시작선 각도
                a = angle360(centerX,centerY,mouseX,mouseY)  
                if a < 240 and a > lineTailAngGREEN: lineLeadAngGREEN = a
            #--------------------------------------------------------------------------------
            elif selectedObject=='hueTailGREEN':            # 마우스의 위치(원 중심으로 부터)에따른 종료선 각도
                a = angle360(centerX,centerY,mouseX,mouseY)  
                if a > 60 and a < lineLeadAngGREEN: lineTailAngGREEN = a
            #--------------------------------------------------------------------------------
            elif selectedObject=='BOX_VAL_THRH_GREEN':
                #a = mouseX - (80+14)
                a = mouseX - 5
                if 0 <= a < 256: HSV_valueGREEN = a 
                #a = (68+400) - mouseY
                a = 453 - mouseY
                if 0 <= a < 256: THRESH_GREEN = a
            #--------------------------------------------------------------------------------
            elif selectedObject=='CIRCLE_MAX':
                a = CIRCLE_X - mouseX
                if minRadius < a <= MAX_RADIUS:
                    maxRadius = a
            #--------------------------------------------------------------------------------
            elif selectedObject=='CIRCLE_MIN':
                a = mouseX - CIRCLE_X
                if MIN_RADIUS <= a < maxRadius:
                    minRadius = a
            #--------------------------------------------------------------------------------
            elif selectedObject=='TSBOX_LU':
                if (mouseX >= 80) and mouseX < (TSWIN_XR+80-MAX_RADIUS*2):
                    TSWIN_XL = mouseX-80
                TSWIN_YU = mouseY if TSWIN_YD-MAX_RADIUS*2 > mouseY else TSWIN_YU
            #--------------------------------------------------------------------------------               
            elif selectedObject=='TSBOX_RD':
                if mouseX < 720 and mouseX > TSWIN_XL+80+MAX_RADIUS*2:
                    TSWIN_XR = mouseX-80
                TSWIN_YD = mouseY if mouseY > TSWIN_YU+MAX_RADIUS*2  else TSWIN_YD
        #-------------------------------------------------------------------------------------
        cv2.imshow('Out', viewWin)
        #-------------------------------------------------------------------------------------
        key = cv2.waitKey(1)
        if key == ord('x') or key == ord('X') or key == ord(' ') or key == 27: break
    #-----------------------------------------------------------------------------------------
    if DIRECTORY_FLAG:             # 디렉토리(프로젝트 이름)가 존재할 때 신호등 관련 파라메터를 저장한다.
        w = [TSWIN_XL,             # 카메라 영역 X 축 Left  0~640 
             TSWIN_YU,             # 카메라 영역 Y 축 Up    0~480 
             TSWIN_XR,             # 카메라 영역 X 축 Right 0~640 
             TSWIN_YD,             # 카메라 영역 Y 축 Down  0~480
             minRadius,            # 신호등 최소 반지름      >8
             maxRadius,            # 신호등 최대 반지름      <40
             lineLeadAngGREEN,     # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
             lineTailAngGREEN,     # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
             lineLeadAngRED,       # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
             lineTailAngRED,       # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
             saturationHalfGREEN,
             saturationHalfRED,
             HSV_valueGREEN,       # Green Value
             HSV_valueRED,         # Red Value
             THRESH_GREEN,         # Green Threshold
             THRESH_RED]           # Red Threshold
             
        t = './'+imageDir+'/_'+imageDir+'_TS.pickle'
        with open(t, 'wb') as fw:
            pickle.dump(w, fw, pickle.HIGHEST_PROTOCOL)
        print('신호등 데이터를 저장했습니다.', t)

#============================================================================================
if __name__ == '__main__':
    #----------------------------------------------------------------------------------------
    CIRCLE_X = 400              # 신호등 원 최대/최소 크기 설정용 X 좌표 
    CIRCLE_Y = 60               # 신호등 원 최대/최소 크기 설정용 Y 좌표
    MAX_RADIUS = 38             # 신호등의 최대 반지름(상수값)
    MIN_RADIUS = 8              # 신호등의 작은 반지름(상수값)

    outCirDia = 135             # Lead 객체와 Tail 객체를 연결하는 외곽 원 반지름
    centerX = 400               # HSV 기준원 중심 X좌표
    centerY = 335               # HSV 기준원 중심 Y좌표
    mouseX = centerX            # 마우스 X좌표
    mouseY = centerY            # 마우스 Y좌표
    mouseButtenLeft = False     # 마우스 왼쪽 버튼
    mouseButtenToggle = False   # 마우스 오른쪽 버튼
    #----------------------------------------------------------------------------------------
    viewWin = np.zeros((480,800,3),np.uint8)  # 표시되는 윈도우 가로, 세로, 컬러층, 8비트

    cam = cv2.VideoCapture(0,cv2.CAP_V4L)
    cam.set(3,640)
    cam.set(4,480)

    logo = cv2.imread("./_IMAGE/hcir.png",cv2.IMREAD_UNCHANGED)    # BGR + 알파층 까지 읽는다.
    mask = logo[:,:,3]           # 알파층 1개 를 마스크 층으로 사용(1 개 층 이미지) logo[:,:,-1]
    logo = logo[:,:,0:3]         # 0,1,2번 층을 가져온 것이다.(알파층 빼고, 3 개층 이미지가 된다)
    height, width = logo.shape[:2]
    #----------------------------------------------------------------------------------------
    cv2.namedWindow('Out',cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty('Out', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    # Mouse Callback Function ---------------------------------------------------------------
    # Mouse Event ---------------------------------------------------------------------------
    cv2.namedWindow('Out')                               # 윈도우 창을 생성
    cv2.setMouseCallback('Out', controlMain, viewWin)    # 마우스 제어 설정
    #----------------------------------------------------------------------------------------
    main()
    cam.release()                           # 카메라 자원을 반납
    cv2.destroyAllWindows()                 # 열려 있는 모든 윈도우를 닫기
    #----------------------------------------------------------------------------------------
#############################################################################################

