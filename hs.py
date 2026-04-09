#############################################################################################
# HSV 색상 좌표계                                                             <hsvscope.py>
#
# HSV 색상 좌표계를 이해하기 쉽도록 합니다.
#
# 이미지에서 원하는 색상의 이미지를 구하기 위하여  2 개의 Hue 값, Saturation, Value 값을 
# 설정하여 마스크 이미지를 만들고 원래의 이미지와  Bitwise AND 연산하여 추출한 이미지를 디스프레이 합니다.
#
# Hue 값의 범위: 0~179
# Saturation 값의 범위: 0~255
# Value 값의 범위: 0~255
#
# 마우스 왼쪽 버튼을 누르면 객체 값 설정 모드가 되며 가장 가까운 객체가 선택됩니다.
#
# Hue Lead Angle 객체
# Hue Tail Angle 객체
# Saturation 객체
# Value 객체
# 
# 마우스를 천천히 이동하면 객체 위치가 변경됩니다.
# 4 개의 HSV 파라메터에 따라 마스크 가 만들어지며 
# 마우스 오른쪽 버튼을 한번 더 누르거나 오른쪽 마우스 버튼을 누르면 객체 선택 모드가 해지 됩니다.
#
# 2023-12-03
#
# SAMPLE Electronics co.                                        
# http://www.ArduinoPLUS.cc                                     
#                                                               
# Library Import ============================================================================
import numpy as np          # 이미지 배열 생성(OpenCV 에서 사용) 라이브러리
import cv2                  # 영상처리(OpenCV) 라이브러리 
from math import sin, cos, atan2, sqrt, pi
from operator import itemgetter
# Constant ----------------------------------------------------------------------------------
RED     =   0,  0,255       # Red
GREEN   =   0,255,  0       # Green
BLUE    = 255,  0,  0       # Blue
MAGENTA = 255,  0,255       # Magenta(Pink)
CYAN    = 255,255,  0       # Cyan(Sky Blue)
YELLOW  =   0,255,255       # Yellow
WHITE   = 255,255,255       # White
GRAY    =  32, 32, 32       # Gray
BLACK   =   0,  0,  0       # Black
#--------------------------------------------------------------------------------------------
outCirDia = 135             # Lead 객체와 Tail 객체를 연결하는 외곽 원 반지름
centerX = 575               # HSV 기준원 중심 X좌표
centerY = 340               # HSV 기준원 중심 Y좌표
mouseX = centerX            # 마우스 X좌표
mouseY = centerY            # 마우스 Y좌표
mouseButtenLeft = False     # 마우스 왼쪽 버튼
mouseButtenToggle = False   # 마우스 오른쪽 버튼
SIZE_EQU = False            # 디스프레이 모드
#--------------------------------------------------------------------------------------------
viewWin = np.zeros((480,800,3),np.uint8)  # 표시되는 윈도우 가로, 세로, 컬러층, 8비트
cir = cv2.imread('./_IMAGE/cirh256.png',cv2.IMREAD_COLOR)      # Hue 원 이미지 읽기
y, x, _ = cir.shape                                   # 이미지의 크기
cirHueImgSizeY = int(y/2); cirHueImgSizeX = int(x/2)  # 이미지 세로 크기, 가로 크기 
camera = cv2.VideoCapture(0,cv2.CAP_V4L)              # 카메라 객체 생성
camera.set(3,640)                                     # 카메라 영상 X 크기
camera.set(4,480)                                     # 카메라 영상 Y 크기
#--------------------------------------------------------------------------------------------
cv2.namedWindow('Out',cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Out', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# Mouse Callback Function -------------------------------------------------------------------
def controlMain(event,x,y,flags,param):
    '''
    마우스 이벤트 발생시 처리되는 함수 입니다.
    3 개의 마우스 버튼 Up/Down 이벤트와 X,Y 이동 이벤트를 처리합니다.
    '''
    global mouseX, mouseY, mouseButtenLeft, mouseButtenToggle
    
    mouseX = x; mouseY = y              # 마우스 좌표값 X, Y를 전역변수 mouseX, mouseY에 저장

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
#--------------------------------------------------------------------------------------------
def valueBox(hue, width=50):
    '''
    hue 값을 받아 256 그레이드의 value 값이 표현되는 박스 이미지를 만듭니다.
    saturation 값은 255 으로 고정입니다.
    반환되는 박스 이미지의 크기는 width 값을 지정하지 않으면 가로는 50 pixel 이며 
    세로는 256 pixel 고정 입니다.
    hue 값의 범위는 0-179 입니다.
    '''
    hue %= 180                       # 휴값 최대 
    value_steps = 256                # 밸류값 최대 (사각형 Height)
    saturation = 255                 # 채도값 최대
    # HSV 이미지 생성
    hsv_image = np.zeros((value_steps, width, 3), dtype=np.uint8)
    hsv_image[..., 0] = hue          # hue 값 입력 (0 번층)
    hsv_image[..., 1] = saturation   # saturation 값 설정 (1 번층) 
    hsv_image[..., 2] = np.tile(np.arange(255, -1, -1).reshape(-1, 1), (1, width)) # value 값 설정  (2 번층)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return(bgr_image)
# Main Loop =================================================================================
def main():

    global mouseButtenLeft, mouseButtenToggle, SIZE_EQU

    lineLeadA = 60                      # 시작(Lead) 라인 각도(/360)
    lineLeadX = 0                       # 시작(Lead) 라인과 외곽 원 교차 X 좌표
    lineLeadY = 0                       # 시작(Lead) 라인과 외곽 원 교차 Y 좌표
    lineTailA = 30                      # 종료(Tail) 라인 각도(/360)
    lineTailX = 0                       # 종료(Tail) 라인과 외곽 원 교차 X 좌표
    lineTailY = 0                       # 종료(Tail) 라인과 외곽 원 교차 Y 좌표
    saturationHalf = 50                 # Saturation 반지름 값 (/128)

    HSV_hue_Lead = int(lineLeadA/2)     # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
    HSV_hue_Tail = int(lineTailA/2)     # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
    HSV_saturation = saturationHalf*2   # HSV 색상 좌표계에서 Saturation(/256)
    HSV_value  = 50                     # HSV 색상 좌표계에서 Value(/256)

    # Object 리스트에 4 개의 이름과 마우스와의 거리 정보가 들어 갑니다. 
    objectList = [['hueLead',0],
                  ['hueTail',0],
                  ['saturation',0],
                  ['value',0]]
    selectedObject = None

    while True:

        viewWin[:] = BLACK
        #viewWin[centerY-sy:centerY+sy,centerX-sx:centerX+sx] = cir
        viewWin[centerY-cirHueImgSizeY:centerY+cirHueImgSizeY,centerX-cirHueImgSizeX:centerX+cirHueImgSizeX] = cir

        _, frame = camera.read()

        # BGR 색상 좌표계를 HSV 색상 좌표계로 변환  
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Lead Angle 객체 좌표 - 반지름 135 인 원과 Lead Angle 라인 교차 좌표값
        lineLeadX = centerX+int(135*cos(lineLeadA*pi/180.0))
        lineLeadY = centerY-int(135*sin(lineLeadA*pi/180.0)) 

        # Tail Angle 객체 좌표 - 반지름 135 인 원과 Tail Angle 라인 교차 좌표값
        lineTailX = centerX+int(135*cos(lineTailA*pi/180.0))
        lineTailY = centerY-int(135*sin(lineTailA*pi/180.0))

        # SaturationHalf 을 반지름으로 원과 Lead각, Tail각 교차 좌표값
        lineLeadSX = centerX+int(saturationHalf*cos(lineLeadA*pi/180.0))
        lineLeadSY = centerY-int(saturationHalf*sin(lineLeadA*pi/180.0)) 
        lineTailSX = centerX+int(saturationHalf*cos(lineTailA*pi/180.0))
        lineTailSY = centerY-int(saturationHalf*sin(lineTailA*pi/180.0))

        # Saturation 객체 좌표
        if lineLeadA >= lineTailA:
            a = ((lineLeadA+lineTailA)/2)%360
        else:
            a = ((lineLeadA+lineTailA)/2+180)%360
        saturationX = centerX+int(saturationHalf*cos(a*pi/180.0))
        saturationY = centerY-int(saturationHalf*sin(a*pi/180.0)) 

        # Value 객체 좌표
        valueX = 765
        valueY = 210+256-HSV_value

        objectList[0][1] = (lineLeadX-mouseX)**2+(lineLeadY-mouseY)**2
        objectList[1][1] = (lineTailX-mouseX)**2+(lineTailY-mouseY)**2
        objectList[2][1] = (saturationX-mouseX)**2+(saturationY-mouseY)**2
        objectList[3][1] = (valueX-mouseX)**2+(valueY-mouseY)**2
 
        #------------------------------------------------------------------------------------
        if mouseButtenLeft:
            selectedObject = sorted(objectList,key=itemgetter(1))[0][0] # 마우스와 최 단거리 객체
            mouseButtenToggle = not mouseButtenToggle
            mouseButtenLeft = False                                     # 원 슈트(One Shoot)

        if mouseButtenToggle:
            if selectedObject=='value':                # HSV Value
                v = (210+256)-mouseY
                if 0<=v and v<256:
                    HSV_value = v                      # 0~255

            elif selectedObject=='saturation':         # HSV Saturation
                s = int(sqrt((mouseX-centerX)*(mouseX-centerX)+(mouseY-centerY)*(mouseY-centerY))) # circle 중심으로 부터 mouse 거리
                if 0<= s and s < 128:
                    saturationHalf = s
                    HSV_saturation = saturationHalf*2  # HSV Saturation은 0~255 까지

            elif selectedObject=='hueLead':            # 마우스의 위치(원 중심으로 부터)에따른 시작선 각도
                lineLeadA = angle360(centerX,centerY,mouseX,mouseY)  

            elif selectedObject=='hueTail':            # 마우스의 위치(원 중심으로 부터)에따른 종료선 각도
                lineTailA = angle360(centerX,centerY,mouseX,mouseY)  

        HSV_hue_Lead = int(lineLeadA/2)                # 0~180
        HSV_hue_Tail = int(lineTailA/2)                # 0~180

        if lineLeadA >= lineTailA:                     # 두 각이 동일한 영역에 있을 때이며 외곽원 Green 
            centerHue = int((lineLeadA+lineTailA)/4)   # 두 각의 중간값(360도 원)각을 구하고 다시 180 원 값을 구한다.
            cv2.ellipse(viewWin,(centerX,centerY),(outCirDia,outCirDia),0,-lineLeadA,-lineTailA,GREEN,3,cv2.LINE_AA)
            cv2.ellipse(viewWin,(centerX,centerY),(saturationHalf,saturationHalf),0,-lineLeadA,-lineTailA,BLACK,1,cv2.LINE_AA)

            upper_color = np.array([HSV_hue_Lead,255,255])                   # Upper
            lower_color = np.array([HSV_hue_Tail,HSV_saturation,HSV_value])  # Lower
            mask = cv2.inRange(hsv, lower_color, upper_color)                # 단일 구간 마스크
        else:                                          # 두 각이 교차 영역에 있을 때이며 외곽원 Red
            centerHue = (int((lineLeadA+lineTailA)/4)+90)%180   # 두 각의 중간값(360도 원)각을 구하고 다시 180 원 값을 구한다.
            cv2.ellipse(viewWin,(centerX,centerY),(outCirDia,outCirDia),0,-lineLeadA,0,RED,3,cv2.LINE_AA)
            cv2.ellipse(viewWin,(centerX,centerY),(saturationHalf,saturationHalf),0,-lineLeadA,0,BLACK,1,cv2.LINE_AA)

            cv2.ellipse(viewWin,(centerX,centerY),(outCirDia,outCirDia),0,-lineTailA,-360,RED,3,cv2.LINE_AA)
            cv2.ellipse(viewWin,(centerX,centerY),(saturationHalf,saturationHalf),0,-lineTailA,-360,BLACK,1,cv2.LINE_AA)
        
            upper_color = np.array([HSV_hue_Lead,255,255])             
            lower_color = np.array([0,HSV_saturation,HSV_value])       # 최소 0
            maskL = cv2.inRange(hsv, lower_color, upper_color)         # Lead Mask

            upper_color = np.array([179,255,255])                      # 최대 179
            lower_color = np.array([HSV_hue_Tail,HSV_saturation,HSV_value])  
            maskT = cv2.inRange(hsv, lower_color, upper_color)         # Tail Mask

            mask = cv2.bitwise_or(maskL, maskT)                        # L 와 T 마스크를 OR 연산  

        cv2.line(viewWin,(lineLeadSX,lineLeadSY),(lineLeadX,lineLeadY),BLACK,1,cv2.LINE_AA)
        cv2.line(viewWin,(lineTailSX,lineTailSY),(lineTailX,lineTailY),BLACK,1,cv2.LINE_AA)

        viewWin[210:210+256,740:740+50]=valueBox(centerHue)            # Value Box 디스프레이
        cv2.rectangle(viewWin,(740,210),(740+50,210+256),YELLOW,1)     # Value Box 외곽선
        cv2.line(viewWin,(740,210+256-HSV_value),(740+50,210+256-HSV_value),WHITE,1) # Value 레벨

        # Lead Angle 객체 위치 디스프레이 
        cv2.circle(viewWin,(lineLeadX,lineLeadY),8,RED,-1)
        # Tail Angle 객체 위치 디스프레이
        cv2.circle(viewWin,(lineTailX,lineTailY),8,GREEN,-1)
        # Saturation 객체 위치 디스프레이
        cv2.circle(viewWin,(saturationX,saturationY),8,BLACK,-1)
        # Value 객체 위치 디스프레이
        cv2.circle(viewWin,(valueX, valueY),8,WHITE,-1)

        res = cv2.bitwise_and(frame,frame,mask=mask)    # 원 이미지와 마스크 이미지의 AND 처리
        maskC = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # 1 Layer(흙백) 마스크를 3 Layer(컬러)로 변환
        
        if SIZE_EQU:
            viewWin[0+0:0+200,0+0:0+266] = cv2.resize(frame, dsize=(266,200), interpolation=cv2.INTER_LINEAR) # 원본 디스프레이
            viewWin[0+0:0+200,0+266+1:0+266+1+266] = cv2.resize(maskC, dsize=(266,200), interpolation=cv2.INTER_LINEAR) # 마스크 디스프레이
            viewWin[0+0:0+200,0+266+1+266+1+0:0+266+1+266+1+266] = cv2.resize(res, dsize=(266,200), interpolation=cv2.INTER_LINEAR) # 결과 디스프레이
            cv2.putText(viewWin,'HSV Scope',(100,290),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,CYAN)
            cv2.putText(viewWin,f'Hue(Lead): {HSV_hue_Lead}/180',(100,320),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv2.putText(viewWin,f'Hue(Tail): {HSV_hue_Tail}/180',(100,340),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv2.putText(viewWin,f'Saturation: {HSV_saturation}/256',(100,360),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv2.putText(viewWin,f'Value: {HSV_value}/256',(100,380),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv2.putText(viewWin,'[View mode] [eXit]',(100,410),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,YELLOW)

            cv2.rectangle(viewWin,(0+0,0+0),(0+266,0+200),YELLOW,1)
            cv2.rectangle(viewWin,(0+266+1,0+0),(0+266+1+266,0+200),YELLOW,1)
            cv2.rectangle(viewWin,(0+266+1+266+1+0,0+0),(0+266+1+266+1+266-1,0+200),YELLOW,1)

        else:
            viewWin[0+0:0+160,0+0:0+213] = cv2.resize(frame, dsize=(213,160), interpolation=cv2.INTER_LINEAR) # 원본 디스프레이
            viewWin[0+0:0+160,0+213:0+213+213] =  cv2.resize(maskC, dsize=(213,160), interpolation=cv2.INTER_LINEAR) # 마스크 디스프레이
            viewWin[160+0:160+320,0+0:0+426] =  cv2.resize(res, dsize=(426,320), interpolation=cv2.INTER_LINEAR) # 결과 디스프레이

            cv2.putText(viewWin,'HSV Scope',(450,40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,CYAN)
            cv2.putText(viewWin,f'Hue(Lead): {HSV_hue_Lead}/180',(450,70),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv2.putText(viewWin,f'Hue(Tail): {HSV_hue_Tail}/180',(450,90),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv2.putText(viewWin,f'Saturation: {HSV_saturation}/256',(450,110),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv2.putText(viewWin,f'Value: {HSV_value}/256',(450,130),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv2.putText(viewWin,'[View mode] [eXit]',(450,160),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,YELLOW)

            cv2.rectangle(viewWin,(0+0,0+0),(0+213,0+160),YELLOW,1)
            cv2.rectangle(viewWin,(0+213,0+0),(0+213+213,0+160),YELLOW,1)
            cv2.rectangle(viewWin,(0+0,160+0),(0+426,160+320-1),YELLOW,1)

        #------------------------------------------------------------------------------------
        cv2.imshow('Out', viewWin)           # 이미지를 LCD에 표시
        #------------------------------------------------------------------------------------
        keyBoard = cv2.waitKey(1) & 0xFF     # 키보드 입력
        if keyBoard == ord('v') or keyBoard == ord('V'): 
            SIZE_EQU = not SIZE_EQU
        if keyBoard == 0x1B or keyBoard == 0x09 or keyBoard == ord('x') or keyBoard == ord('X'):
            break                            # ESC / TAB / 'x' 프로그램 종료

# Mouse Event -------------------------------------------------------------------------------
cv2.namedWindow('Out')                               # 윈도우 창을 생성
cv2.setMouseCallback('Out', controlMain, viewWin)    # 마우스 제어 설정
#--------------------------------------------------------------------------------------------
main()                                               # HSV 메인 프로그램 실행
#--------------------------------------------------------------------------------------------
camera.release()                                     # 카메라 자원을 반납
cv2.destroyAllWindows()                              # 열려 있는 모든 윈도우를 닫기
#############################################################################################
