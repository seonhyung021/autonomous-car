#############################################################################################
#                                                               
# Auto Drive Car                                                    <gi.py>
#
# gi.py 는  자율 주행 하기 위하여 딥러닝에 필요한 이미지를 수집하여 저장 합니다.
# 카메라에서 입력한 이미지는 NVIDIA 에서 정의한 200 x 66 픽셀 크기로 축소,YUV 색상 좌표계로 변환, 
# Gauss법에 의한 노이즈 제거 후 SD 카드의 지정한 디렉토리에 저장합니다.
#
# A. 차량 조정:
#
#     (1) 커서 키에 의한 조정
#         4 개 커서 키에 의하여 전진, 후진, 좌회전, 우회전이 가능합니다.
#         커서키는 한번 누르면 차량이 동작합니다. 
#         (연속으로 누르면 차가 진행, 멈춤을 반복하여 떨림 현상이 발생합니다.)
#         차량이 동작 상태에서 4 개의 방향키 모두 누르거나 [SPACE]를 누르면 정지합니다.
#
#     (2) 마우스에 의한 조정
#         이미지 수집하기 위한 초기 위치로 차량을 이동 (마우스 커서가 Green 색)하거나 
#         이미지 수집 하면서 주행(마우스 커서가 Red 색)할때 마우스 커서로 차량의 이동을 조정합니다.
#         마우스 커서가 중앙에 있을 때 직진이며 마우스 커서가 좌우에 있을 때 커서의 위치 많큼 
#         차량 회전 각도가 변화합니다.
#         마우스 조정에 의한 차량 조정시 후진은 안됩니다.
#
#
# B. 딥러닝 이미지 파일 이름 규약:
#
#     파일 이름은 일련 번호와 조향각이 포함 된 PNG 이미지 입니다. (확장자가 .png)
#     시작문자 1, 일련 번호 4, 언더바 2, 부호 1, 조향각 2 문자, '.' 'png'로 구성됩니다.
#
#     예) P0001__-45.png   P1234__+12.png
#
#     2 개의 언더바("_") 는 자유롭게 사용할 수 있는 빈 공간이며 딥러닝 조건 설정에서 
#     추가적인 라벨용으로 사용됩니다. 
#     현재 Augmentation 용 라벨로 사용합니다.  
#
#     예) P1234_R-24.png
#
#
# C. 커맨드 창에서 이미지 수집 프로그램 gi.py 의 기동 방법
#
#     $ python gi.py <이미지 디렉터리> <시작번호>
#
#     Example:
#
#     (1) 주행 연습모드
#
#         $ python gi.py  
#                                이미지를 저장할 디렉토리를 지정하지 않았습니다. 
#                                파라메터와 이미지 파일이 저장되지 않으며
#                                차량의 이미지 수집/주행 연습만 가능합니다.
#                                
#     (2) 이미지 수집,주행 모드
#
#         $ python3 gi.py wayA
#                                커맨드 라인에서 수집한 이미지를 저장할 디렉토리를 지정합니다.
#                                디렉터리: wayA  이미지 시작번호:0
#                                수집한 이미지와 4 개의 이미지의 수집 크기 정보 
#                                (NVIDIA 200 x 33 으로 축소하기 전 이미지 크기/카메라의 화면에서 위치정보) 
#                                전조등(Lamp)의 상태, 마지막 이미지 파일 번호에서 1 증가된 id  
#
#     (3) 이미지 수집 번호를 지정하여 수집, 주행 모드
#
#         $ python3 gi.py wayA 2000
#                                디렉터리: wayA  이미지 시작번호: 2000
#                                커맨드 라인에서 수집한 이미지를 저장할 디렉토리와 시작 번호를 같이 지정합니다.
#
# D. gi.py 프로그램 초기 기동
#
#      커맨드 라인에서 디렉토리를 지정하면 디렉토리가 이미 존재하는지 확인 합니다. 
#      디렉토리가 존재하면 디렉토리 안에 pickle 파일이 있는지 확인하고 
#      있으면 pickle 파일을 읽어 4 개의 카메라 영역 데이터, 전조등 데이터, 
#      마지막으로 저장한 파일번호의 다음 번호(1 증가된 번호)를 가져옵니다.
#      pickle 파일이 있어도 커멘드 라인에 시작 번호가 지정되어 있으면 pickle 파일의
#      번호는 무시되며 지정한 번호로 시작합니다.
#      지정한 디렉토리가 존재하지 않으면 새로 만들지를 사용자에게 확인합니다. 
#      "Y", "y" 를 누르면 지정한 이름으로 디렉토리를 생성합니다. 그 외의 모든 키는 
#      연습 모드이며 타이틀에 "Exercise Mode" 표시됩니다.
# 
#
# E. 메뉴 및 키 조작
#
#     (1) [A] 키:
#         주위 조명이 어두워지면 On 되며 밝으면 Off 되는 전조등 자동모드 
#         수동 조작(L)에 의하여 전조등이 On/Off 되는 수동 모드를 선택합니다. 
#         Toggle 모드로 동작합니다.
#
#     (2) [L] 키:
#         전조등 수동 모드시 전조등 On/Off를 선택합니다. 
#         전조등 자동모드에서 동작하지 않습니다.
#         ar.py 에서 지원하는 연출 효과(경찰차, 앰블런스 등)는 없습니다.
#         Toggle 모드로 동작합니다.
#
#     (3) [V] 키:
#         고정 또는 가변 이미지 수집 모드 선택를 선택합니다.
#         1. 가변 이미지 수집 모드 
#            차량 조정시 마우스의 Y 축 데이터를 이용하여 도로의 형태가 급격한 커브이면 마우스 커서를 스크린 
#            아래 방향으로 놓아 차량 속도를 느리게(최소 45) 하고 촬영 시간 간격은 작게(최소 0.1초) 합니다. 
#            도로의 형태가 단순한 직선 구간은 차량의 속도를 빠르게(최대 95) 하고 
#            촬영 시간 간격은 크게(1.0초) 합니다.
#            표시 영역에서 마우스가 박스내 위쪽에 위치하면 차량의 속도가 빠르게 되며 촬영영속도 간격은 커집니다.
#            표시 영역에서 마우스가 박스내 아래쪽에 위치하면 차량의 속도가 느리게 되며 촬영속도는 작아집니다.  
#
#         2. 고정 이미지 수집 모드
#            Y 축 방향으로의 마우스의 위치와 관계없이 차량 속도와 이미지 수집 시간 간격을 고정되어 있습니다.
#
#         default 설정은 차량 속도(defForSpeed)와 이미지 촬영 속도(defRecTime)는 고정입니다.
#         Toggle 모드로 동작 합니다.
#
#     (4) [C] 키:
#            클리어 뷰 모드, 메뉴 뷰 모드를 선택합니다.
#            1. 메뉴 뷰: 키/마우스 버턴의 해당 기능과 파일 이름, 200 x 66 RGB 이미지가 표시됩니다.
#            2. 클리어 뷰: 카메라에서 입력한 영상 전부 보여지며  최소의 정보인 이미지 파일 이름만 표시 됩니다. 
#               딥 러닝 이미지 저장 영역은 박스 표시 됩니다.
#            Toggle 모드로 동작 합니다.
#
#     (5) [^1] 키:
#            좌/우 이미지 영역을 축소 합니다.
#
#     (6) [^2] 키:
#            좌우/이미지 영역을 확대 합니다.
#
#     (7) [^3] 키:
#            이미지 상단을 아래로 내립니다.
#
#     (8) [^4] 키:
#            이미지 상단을 위로 올립니다.
#
#     (9) [^5] 키:
#            이미지 하단을 위로 올립니다.
#
#     (10) [^6] 키:
#            이미지 하단을 아래로 내립니다.
#
#     (11) 마우스 왼쪽 버튼: 
#            주행/촬영(수집) 시작합니다.
#
#     (12) 마우스 중앙 버튼: 
#           1 개 이미지를 촬영(수집)합니다.
#
#     (13) 마우스 오른쪽 버튼: 
#           차량 정지, 이미지 수집 정지 합니다.
#
#     (14) 커서 up: 
#           차량이 전진 합니다.
#
#     (15) 커서 Down: 
#           차량이 후진 합니다.
#
#     (16) 커서 Left: 
#           차량이 좌회전 합니다.
#
#     (17) 커서 Right: 
#           차량이 우회전 합니다.
#
#     (18) [ESC],[TAB],X: 
#           이미지 수집 프로그램 실행을 마치고 커맨드 창으로 돌아갑니다.
#
# F. 프로그램 종료시 pickle 에 저장되는 데이터
#
#     gi.py 에서 pickle 로 저장하는 데이터는 스크린 영역 4 개, 전조등 상태 2 개,
#     다음 이미지 촬영시 부여되는 파일 Id 입니다.
#     스크린 영역 데이터 4 개는 자율 주행 프로그램 ar.py 에서도 pickle 파일을 읽어 사용합니다.
# 
# G. 스크린 구성
#
#     Raspberry Pi 의  DSI 모드로 연결되는 LCD 의 해상도는 800 x 480 픽쎌,
#     donaldCar 에서 사용한 카메라는 광각으로 640 x 480 해상도를 사용합니다. 
#     카메라 해상도를 크게하면 프레임당 처리 시간이 길어지며 다른 카메라로 교체할 때 
#     프로그램/카메라 호환성 문제가 발생할 수 있으므로 640 x 480 표준 VGA를 사용합니다.  
#     카메라에서 640(X) x 480(Y) 크기로 이미지를 가지고 와 LCD 의 800 x 480 영역중
#     [0:480,80:720] 을 사용합니다.
#
#
# FEB 2 2022
#
# SAMPLE Electronics co.                                        
# http://www.ArduinoPLUS.cc                                     
#                                                               
#############################################################################################
#
WIN_GAP_X = 200             # 제어 윈도우 X 최소 폭
WIN_GAP_Y = 20              # 제어 윈도우 Y 최소 폭
WIN_XL_ORG = 0              # 제어 윈도우 왼쪽 X 값
WIN_XR_ORG = 639            # 제어 윈도우 오른쪽 X 값
WIN_YU_ORG = 246            # 제어 윈도우 위쪽 Y 값
WIN_YD_ORG = 459            # 제어 윈도우 아래쪽 Y값
WIN_XL = WIN_XL_ORG         # [0] 제어 윈도우 왼쪽 X 값
WIN_XR = WIN_XR_ORG         # [1] 제어 윈도우 오른쪽 X 값
WIN_YU = WIN_YU_ORG         # [2] 제어 윈도우 위쪽 Y 값
WIN_YD = WIN_YD_ORG         # [3] 제어 윈도우 아래쪽 Y값
#--------------------------------------------------------------------------------------------
AUTO_LIGHT = False          # [4] 전조등 지동 점등 모드 True: 자동 False: 수동
MANUAL_LIGHT = False        # [5] 전조등 수동 점등 모드 True: 점등 False: 소등
fileId = 0                  # [6] 기본(default) 이미지파일 시작 번호
VARIABLE_MODE = False       # 마우스에 의한 주행 속도-촬영 시간 
fileNF = 'P'                # 기본(default) 이미지 파일 이름 문자 (여러 개 문자 가능 예: Project_A)
DIRECTORY_FLAG = False      # 이미지 저장용 디렉토리 상태(True:디렉토리 있음)
imageDir = 'Exercise Mode'  # 이미지 파일 저장 디렉토리 디렉토리를 지정하지 않으면 이미지가 저장되지 않는 연습모드로 실행됩니다.
defForSpeed = 60            # 기본(default) 이미지 저장 차량 전진 속도 (PWM 값)
defMotorAngle = 0.15        # 기본(default) 각도 이득(모터 PWM 좌,우 배분)
defRecTime = 500000000      # 기본(default) 이미지 저장 시간 간격 0.5초
recUnitTime = 100000000     # 가변 이미지 수집 모드에서 이미지 저장 단위 간격 0.1초
recordOn = False            # 이미지 연속 녹화 상태 / mouseCtrlState 가 True이고 마우스 왼쪽 버튼에 의하여 True가 된다.
oneShot = False             # 단일(One Shoot) 이미지 촬영
mouseCtrlState = False      # 마우스에 의한 차량 주행및 조향 제어 상태 (True:제어 가능)
REC_LOW_FORWARD_SPEED = 40  # 가변 차량속도/촬영시간 모드에서 녹화시 최저 모터 속도 (PWM 값)
recTimeList = [recUnitTime*1,recUnitTime*2,recUnitTime*3,recUnitTime*4,
               recUnitTime*5,recUnitTime*6,recUnitTime*7,recUnitTime*8,
               recUnitTime*9,recUnitTime*10,recUnitTime*11] # 가변 차량속도/촬영시간 모드에서 녹화 구간별 녹화 시간 간격
# Library Import ============================================================================
import os                   # 파일, 디렉토리 존재 유무 확인, 디렉토리 생성 라이브러리
import sys                  # 커맨드 라인 인수(이미지 저장 디렉토리, 레코딩 시작번호)처리 라이브러리 
import numpy as np          # 이미지 배열 생성(OpenCV 에서 사용) 라이브러리
import cv2 as cv            # 영상처리(OpenCV) 라이브러리 
import pickle               # python 객체 저장/읽기 라이브러리
import RPi.GPIO as GPIO     # Raspberry Pi 헤터핀 제어 라이브러리
import NEOPIXEL as nx       # NeoPixel 제어 라이브러리
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
# 조향 각도를 7 개 그룹으로 분리 -----------------------------------------------------------------
# 최대 좌회전 명령시 차가 움직이는 방향의 값을 -99, 
# 최대 우회전 명령시 차가 움직이는 방향의 값을 +99 으로 설정합니다.
# 조향 각도 그룹을 세분화 하여 딥러닝 훈련(Training)데이터와 평가(Validation)데이터를 분산
# 
angAL =  -99                # 조향 각 영역 - A Left
angAR =  -47                # 조향 각 영역 - A Right
angBL =  -46                # 조향 각 영역 - B Left
angBR =  -22                # 조향 각 영역 - B Right
angCL =  -21                # 조향 각 영역 - C Left
angCR =  -6                 # 조향 각 영역 - C Right
angDL =  -5                 # 조향 각 영역 - D Left
angDR =  5                  # 조향 각 영역 - D Right
angEL =  6                  # 조향 각 영역 - E Left
angER =  21                 # 조향 각 영역 - E Right
angFL =  22                 # 조향 각 영역 - F Left
angFR =  46                 # 조향 각 영역 - F Right
angGL =  47                 # 조향 각 영역 - G Left
angGR =  99                 # 조향 각 영역 - G Right
#--------------------------------------------------------------------------------------------
VIDEO_X = 640               # Video X Size (960x240 / 960x800 / 1600x240)
VIDEO_Y = 480               # Video Y Size
cursorColor = YELLOW        # 조향 커서 색상 (YELLOW:정지, GREEN:주행, RED:녹화)
mouseX = 400                # 마우스 X 초기 위치
mouseY = 420                # 마우스 Y 초기 위치
mL = 0                      # 왼쪽 모터의 PWM 값이며 절대 값으로 0 부터 100까지(99 아님)
mR = 0                      # 오른쪽 모터의 PWM 값이며 절대 값으로 0 부터 100까지(99 아님)
# GPIO --------------------------------------------------------------------------------------
LIGHT_SENSOR = 25           # GPIO.25  CdS 광 센서 입력
MOTOR_L_PWM = 12            # GPIO.12  왼쪽 모터 펄스 푹 변조(Pulse Width Modulation)
MOTOR_L_DIR = 5             # GPIO.5   원쪽 모터 방향 (Motor Rotation Direction)
MOTOR_R_PWM = 13            # GPIO.13  오른쪽 모터 펄스 폭 변조(Pulse Width Modulation)
MOTOR_R_DIR = 6             # GPIO.6   오른쪽 모터 방향 (Motor Rotation Direction)
# Video Window ------------------------------------------------------------------------------
cam = cv.VideoCapture(0,cv.CAP_V4L)   # 카메라 객체 생성
cam.set(3,VIDEO_X)                    # 카메라 입력 화면 X 크기
cam.set(4,VIDEO_Y)                    # 카메라 입력 화면 Y 크기
# Set up ------------------------------------------------------------------------------------
GPIO.setmode(GPIO.BCM)                # GPIO 핀 번호를 BCM 방식으로 설정 
GPIO.setwarnings(False)               # 부팅시 경고 메시지 출력(터미널 창에서)안보기로 설정
GPIO.setup(MOTOR_L_PWM,GPIO.OUT)      # 왼쪽 모터 PWM 신호 출력 핀 
GPIO.setup(MOTOR_L_DIR,GPIO.OUT)      # 왼쪽 모터 방향 신호 출력 핀
GPIO.setup(MOTOR_R_PWM,GPIO.OUT)      # 오른쪽 모터 PWM 신호 출력 핀 
GPIO.setup(MOTOR_R_DIR,GPIO.OUT)      # 오른쪽 모터 방향 신호 출력 핀
MOTOR_L = GPIO.PWM(MOTOR_L_PWM,500)   # 왼쪽 모터 PWM 주파수 500Hz
MOTOR_R = GPIO.PWM(MOTOR_R_PWM,500)   # 오른쪽 모터 PWM 주파수 500Hz
MOTOR_L.start(0)                      # 왼쪽 모터 펄스폭 변조(PWM)값 0 으로 시작
MOTOR_R.start(0)                      # 오른쪽 모터 펄스폭 변조(PWM)값 0 으로 시작
GPIO.setup(LIGHT_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP) # CdS 빛감지 센서 핀 방향(입력), 형태(풀업)설정
#--------------------------------------------------------------------------------------------
viewWin = np.zeros((480,800,3),np.uint8)  # 표시되는 윈도우 가로, 세로, 컬러층, 8비트
msgBoxL = cv.imread('./_IMAGE/gimL.png',cv.IMREAD_COLOR)
msgBoxR = cv.imread('./_IMAGE/gimR.png',cv.IMREAD_COLOR)
#--------------------------------------------------------------------------------------------
cv.namedWindow('Out',cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty('Out', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
# Motor Run Function ------------------------------------------------------------------------
def motorRun(leftMotor, rightMotor):
    '''
    2 개의 정수 값을 인수로 받아 모터 드라이버에 신호를 주어 왼쪽 모터와 오른쪽 모터가 동작합니다.
    왼쪽 모터 2 개와 오른쪽 모터 2 개는 각각 +(빨간색)선과 -(검정색)선이 
    묶여져 하나의 모터처럼 동시에 동작합니다. 
    두개의 입력 값(왼쪽 모터, 오른쪽 모터)의 범위는 각각 -100 에서 +100까지 입니다(99 아님).  
    입력 값의 부호에 따라 모터의 회전 방향을 결정합니다. 
    - 값은 차량 후진, + 값은 차량 전진 방향으로 모터가 회전합니다. 
    두 입력 값은 절대 값으로 만든후 PWM 신호로 사용하며 0 일때 모터 정지, 
    100일때 최고 속도로 모터가 회전합니다
    '''

    #return
    if leftMotor>100: leftMotor = 100        # 왼쪽 모터 전진 최대 PWM 값 (100: 최고속도)
    if leftMotor<-100: leftMotor = -100      # 왼쪽 모터 후진 최대 PWM 값 (100: 최고속도)

    if (leftMotor>=0): GPIO.output(MOTOR_L_DIR,GPIO.HIGH)   # 왼쪽 모터 전진
    else: GPIO.output(MOTOR_L_DIR,GPIO.LOW)  # 왼쪽 모터 후진
    #----------------------------------------------------------------------------------------
    if rightMotor>100: rightMotor = 100      # 오른쪽 모터 전진 최대 PWM 값 (100: 최고속도)
    if rightMotor<-100: rightMotor = -100    # 오른쪽 모터 후진 최대 PWM 값 (100: 최고속도)

    if  rightMotor >= 0: GPIO.output(MOTOR_R_DIR,GPIO.HIGH)   # 오른쪽 모터 전진
    else: GPIO.output(MOTOR_R_DIR,GPIO.LOW)  # 오른쪽 모터 후진
    #----------------------------------------------------------------------------------------
    MOTOR_L.ChangeDutyCycle(abs(leftMotor))  # 왼쪽 모터 PWM 값 지정
    MOTOR_R.ChangeDutyCycle(abs(rightMotor)) # 오른쪽 모터 PWM 값 지정
# Mouse Callback Function -------------------------------------------------------------------
def controlMain(event,x,y,flags,param):
    '''
    마우스(Mouse)가 동작(움직임, 버튼 클릭)하면 인터럽트가 발생하며 이때 상황에 따른 
    마우스 동작을 실현 합니다. 마우스가 움직임으로 인터럽트가 발생하면 마우스 좌표값 X, Y는 
    전역변수 mouseX, mouseY 에 저장하여 메인 프로그램에서 자동차의 핸들에 해당하는 
    수직막대 커서 디스프레이 위치 설정에 사용됩니다.
    마우스 왼쪽 버튼은 한번 누르면 주행 시작하며 두번째(더블 클릭 아님)는 레코딩 모드를 
    설정하여 메인 프로그램에서 이미지 저장 가능하도록 합니다. 
    오른쪽 마우스 버튼을 클릭하면 레코딩 모드와 주행 모드를 취소하여 이미지 저장이 금지되며 
    차량이 정지합니다.
    마우스 중앙 버튼은 원샷 이미지 촬영 가능 상태로 하여 메인 프로그램에서 1 장의 
    이미지를 저장 합니다.
    '''

    global mouseX, mouseY
    global recordOn, cursorColor
    global mouseCtrlState, oneShot

    mouseX = x; mouseY = y             # 마우스가 움직임으로 인터럽트가 발생하여 
                                       # 마우스 좌표값 X, Y를 전역변수 mouseX, mouseY에 저장
    if event == cv.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 이벤트 발생
        if mouseCtrlState == False:    # 마우스에 의한 차량 제어 
            mouseCtrlState = True      # 마우스 왼쪽 클릭에서 마우스 조정 주행상태 수동 조작에 의하여 False 가 된다.
            cursorColor = GREEN        # 조향 커서키, 모터PWM 막대 그래프 색상
        else:
            if DIRECTORY_FLAG:         # 이미지 저장용 디렉토리 상태(True:디렉토리 있음)
                recordOn = True        # 이미지 레코딩 가능
                cursorColor = RED      # 조향 커서키, 모터 PWM 막대 그래프 색상

    if event == cv.EVENT_RBUTTONDOWN:  # 마우스 오른쪽 버튼 이벤트 발생
        mouseCtrlState = False         # 마우스에 의한 차량 제어 금지
        recordOn = False               # 이미지 저장 금지
        cursorColor = YELLOW           # 차량 정지 상태에서 조향 커서키, 모터PWM 막대 그래프 색상
        MOTOR_L.ChangeDutyCycle(0)     # 왼쪽 모터 정지
        MOTOR_R.ChangeDutyCycle(0)     # 오른쪽 모터 정지

    if event == cv.EVENT_MBUTTONDOWN:  # 마우스 가운데 버튼 이벤트 발생
        oneShot = True                 # 원 샷 (One shot) 촬영 가능상태

# Main Loop =================================================================================
def main():

    global WIN_XL, WIN_XR, WIN_YU, WIN_YD, AUTO_LIGHT, MANUAL_LIGHT, fileId
    global VARIABLE_MODE, viewWin, oneShot, recordOn, mL, mR, cursorColor, mouseCtrlState

    angle = 0                          # 마우스 핸들로 만들어진 조향 각도 임시 저장 변수로 사용
    preKey = ord(' ')                  # 초기 키 값으로 space 를 지정합니다.
    MENU_VIEW = True                   # True:메뉴 뷰 모드 회색 바탕위에 기본적인 키에 관한 설명
                                       # False:클리어 뷰 모드 카메라 입력 영상을 전부 볼수 있으며 이미지 파일 이름만 표시
    recordTime = defRecTime            # 고정(기본) 이미지 저장 시간 간격
    forwardSpeed = 60                  # 차량 속도 (최고 속도 = 100)
    shotPeriode = 0                    # 촬영 시간 간격 
    timeMemory = cv.getTickCount()     # 시스템 부팅 시 부터 1 mS 간격으로 카운트 업됩니다.
    #----------------------------------------------------------------------------------------
    while(cam.isOpened()):             # 카메라 활성화 되어 있므면 무한 반복

        ret, frame = cam.read()        # 카메라로부터 한장의 정지화면 데이터를 가져 옵니다.
        viewWin[0:480,0:80] = msgBoxL 
        viewWin[0:480,720:800] = msgBoxR 
        #------------------------------------------------------------------------------------
        viewWin[0:480,80:720] = frame[0:480,0:640]  # 카메라 전체 이미지 
        #------------------------------------------------------------------------------------
        c = GPIO.input(LIGHT_SENSOR)   # 주위가 밝으면  Low(False) 어두우면 High(True)
        if (AUTO_LIGHT and c) or (not AUTO_LIGHT and MANUAL_LIGHT):
            nx.lamp(255,255,255)       # 전조등 On 
            if MENU_VIEW: cv.rectangle(viewWin,(80+216,36),(80+423,109),WHITE,2)
        else: nx.lamp(0,0,0)           # 전조등 Off
        #------------------------------------------------------------------------------------
        roadImg = viewWin[WIN_YU:WIN_YD,80+WIN_XL:80+WIN_XR]  # frame 화면이므로 +80
        roadImg = cv.resize(roadImg, (200, 66))               # NVIDIA 형식으로 변환
        if MENU_VIEW: viewWin[40:(40+66), 80+220:(80+220+200)] = roadImg # 학습 RGB 이미지 표시
        roadImg = cv.cvtColor(roadImg, cv.COLOR_BGR2YUV)      # RGB를 YUV 좌표계로 변환
        roadImg = cv.GaussianBlur(roadImg, (3,3), 0)          # 가우시안법 노이즈 제거
        #------------------------------------------------------------------------------------
        cv.rectangle(viewWin,(80+WIN_XL,WIN_YU),(80+WIN_XR,WIN_YD),CYAN,1) # 딥러닝 영역 박스처리
        viewWin[WIN_YD-WIN_GAP_Y:WIN_YD-WIN_GAP_Y+1,80+WIN_XL:80+WIN_XR:4]=CYAN
        #------------------------------------------------------------------------------------
        angle = int(mouseX-(80+320))    # 중앙을 0 으로 왼쪽을 - 값, 오른쪽을 + 값으로 각도 값 정의
        s = int(angle*defMotorAngle)
        mL = forwardSpeed + s           # 각도 값에 의하여 왼쪽 모터의 속도(PWM)값 계산
        mR = forwardSpeed - s           # 각도 값에 의하여 오른쪽 모터의 속도(PWM)값 계산
        if mL > 100: mL = 100           # 왼쪽 모터 최대 PWM 값 100
        if mR > 100: mR = 100           # 오른쪽 모터 최대 PWM 값 100
        if mouseCtrlState: motorRun(mL, mR)
        g = int(angle/3.6)              # 이미지 파일의 붙여질 각도값, 핸들이 직선 움직임으로 적절한 상수값 보정
        cv.rectangle(viewWin,(angle+80+320-10,WIN_YU),(angle+80+320+10,WIN_YD),cursorColor,2)  #
        if WIN_YU < mouseY < WIN_YD:
            cv.line(viewWin,(80+WIN_XL,mouseY),(mouseX-15, mouseY), cursorColor, 1)
            cv.line(viewWin,(80+WIN_XR,mouseY),(mouseX+15, mouseY), cursorColor, 1)
        t = ' '
        if angAL <= g <= angAR: l = angAL; r = angAR; t = 'A'  # 좌회전
        if angBL <= g <= angBR: l = angBL; r = angBR; t = 'B'
        if angCL <= g <= angCR: l = angCL; r = angCR; t = 'C'
        if angDL <= g <= angDR: l = angDL; r = angDR; t = 'D'  # 직진
        if angEL <= g <= angER: l = angEL; r = angER; t = 'E'
        if angFL <= g <= angFR: l = angFL; r = angFR; t = 'F'
        if angGL <= g <= angGR: l = angGL; r = angGR; t = 'G'  # 우회전
        if MENU_VIEW:
            u = int(l*3.6)+80+320; v = int(r*3.6)+80+320; w = int((u+v)/2)  # 빨간색 배경 좌표값
            z = mouseY
            cv.rectangle(viewWin,(u,z+5),(v,z+17+5),RED,-1)           # 빨간색 블럭
            cv.putText(viewWin,t,(w-7,z+17+5-2),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE) # 블럭문자 표시
        #------------------------------------------------------------------------------------
        # 마우스 위치에 따라 차량의 기준 속도(직진)와 이미지 촬영 간격 시간을 구한다.
        # Y 방향 구간 내에서 마우스 Y 위치를 10 구간으로 구한다. 
        m = mouseY
        if mouseY > WIN_YD: m = WIN_YD                 # Y 방향 마우스 / 전진 최소 속도  
        if mouseY < WIN_YU: m = WIN_YU                 # Y 방향 마우스 / 전진 최대 속도

        if mouseY > (WIN_YD-WIN_GAP_Y): n = 0          # 컨트롤 박스 하단 GAP 영역일 때 0 지정
        else: n = 10-int(10*(m-(WIN_YU))/(WIN_YD-WIN_GAP_Y-WIN_YU))    # Y 에 의하여 0~10 까지 11개의 정수 생성
        if VARIABLE_MODE:                              # 촬영 속도 가변 모드
            recordTime = recTimeList[n]                # recordTime = recUnitTime*abs(n)
            forwardSpeed = REC_LOW_FORWARD_SPEED + n*5 # 양쪽 모터 기준속도 (직진)
        else: recordTime = defRecTime; forwardSpeed = defForSpeed
        #------------------------------------------------------------------------------------
        u = 'Manu'; v = 'Off'; w = 'Day   '; m = 'Fix R'
        if c: w = 'Night'
        if AUTO_LIGHT: u = 'Auto'
        if MANUAL_LIGHT: v = 'On '
        if VARIABLE_MODE: m = 'Var R'
        if not AUTO_LIGHT: cv.putText(viewWin,f'[L] {v}',(3, 100),cv.FONT_HERSHEY_PLAIN,1,YELLOW) 
        cv.putText(viewWin,f'{w}',(3, 20),cv.FONT_HERSHEY_PLAIN,1,MAGENTA) 
        cv.putText(viewWin,f'[A] {u}',(3, 80),cv.FONT_HERSHEY_PLAIN,1,YELLOW) 
        cv.putText(viewWin,f'[V] {m}',(3, 60),cv.FONT_HERSHEY_PLAIN,1,YELLOW) 

        if MENU_VIEW: # 메뉴 모드에서 스크린에 키 설명과 차량의 상태 정보를 표시
            if DIRECTORY_FLAG: cv.putText(viewWin,imageDir,(85, 20),cv.FONT_HERSHEY_PLAIN,1,GREEN)
            cv.putText(viewWin,f'{n}',(mouseX-7,mouseY-25),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE) # 실측한 이미지 촬영 저장 시간 간격
            cv.putText(viewWin,f'  {shotPeriode/1000000000:3.2f}  {forwardSpeed}',(mouseX-83,mouseY-5),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE) # 실측한 이미지 촬영 저장 시간 간격
            cv.putText(viewWin,f'{WIN_XL} {WIN_YU}',(85+WIN_XL, WIN_YU+15),cv.FONT_HERSHEY_PLAIN,1,MAGENTA) 
            cv.putText(viewWin,f'{WIN_XR} {WIN_YD}',(WIN_XR, WIN_YD-5),cv.FONT_HERSHEY_PLAIN,1,MAGENTA) 
            cv.putText(viewWin,f'{mL:3d}                     {mR}',(80+156,105),
                       cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)            # 모터의 PWM값(0-100)
            cv.rectangle(viewWin,(80+195,105),(80+205,105-mL),cursorColor,-1)  # 왼쪽 모터
            cv.rectangle(viewWin,(80+435,105),(80+445,105-mR),cursorColor,-1)  # 오른쪽 모터
        #------------------------------------------------------------------------------------
        v = abs(g); w = '+'
        if g < 0: w = '-'                            # 각도 값이 음수일 때 '-' 기호를 붙인다.
        saveFile = f'{fileNF}{fileId:04d}__{w}{v:02d}.png'  
        if MENU_VIEW:
            cv.putText(viewWin,f'{w}{v}',(80+250, 93),cv.FONT_HERSHEY_COMPLEX_SMALL,3,WHITE)
        # 이미지 레코딩 ----------------------------------------------------------------------- 
        if DIRECTORY_FLAG and (recordOn or oneShot): # 연속 레코드 모드 또는 원샷 촬영 모드
            t = cv.getTickCount()
            q = t - timeMemory
            if q > recordTime or oneShot:            # 촬영 간격 시간에 도달 또는 원샷 촬영
                timeMemory = t; 
                #----------------------------------------------------------------------------
                s = f'{filePath}{saveFile}'
                cv.imwrite(s, roadImg)               # 이미지를 저장
                #----------------------------------------------------------------------------
                shotPeriode = q
                #print(saveFile, shotPeriode)
                fileId += 1                          # 이미지 인덱스 번호 증가
                oneShot = False
        # 이미지 파일 이름을 스크린에 BLUE 박스에 표시한다.
        cv.rectangle(viewWin,(80+212,7),(80+426,7+25),BLUE,-1)
        cv.putText(viewWin,saveFile[-14:],(80+212+8,7+18),cv.FONT_HERSHEY_COMPLEX_SMALL,1,CYAN)
        #------------------------------------------------------------------------------------
        cv.imshow('Out', viewWin)           # 이미지를 LCD에 표시합니다.
        #------------------------------------------------------------------------------------
        keyBoard = cv.waitKey(1) & 0xFF     #print(hex(keyBoard)) # 키보드의 키 값 확인  
        if keyBoard == 0x1B or keyBoard == 0x09 or keyBoard == ord('x') or keyBoard == ord('X'):
            break                           # ESC / TAB / 'x' 프로그램 종료
        # 카메라 입력 영역 조정 -----------------------------------------------------------------
        if keyBoard == ord('!'):                             # X(가로)방향 줄임
            if not fileId and (WIN_XL <= WIN_GAP_X): WIN_XL += 2; WIN_XR -= 2
        if keyBoard == ord('@'):                             # X(가로)방향 늘림
            if not fileId and (WIN_XL >= 2): WIN_XL -= 2; WIN_XR += 2
        if keyBoard == ord('#'):                             # Y-Up 내리기
            if not fileId and ((WIN_YD-WIN_YU) > (WIN_GAP_Y+4)): WIN_YU += 2
        if keyBoard == ord('$'):                             # Y-Up 올리기
            if not fileId and (WIN_YU > 152): WIN_YU -= 2
        if keyBoard == ord('%'):                             # Y-Down 올리기
            if not fileId and ((WIN_YD-WIN_YU) > (WIN_GAP_Y+4)): WIN_YD -= 2
        if keyBoard == ord('^'):                             # Y-Down 내리기
            if not fileId and (WIN_YD <= 477): WIN_YD += 2
        if keyBoard == ord(')'):                             # X, Y 원위치
            if not fileId:
                WIN_XL = WIN_XL_ORG                          # 제어 윈도우 왼쪽 X 값
                WIN_XR = WIN_XR_ORG                          # 제어 윈도우 오른쪽 X 값
                WIN_YU = WIN_YU_ORG                          # 제어 윈도우 위쪽 Y 값
                WIN_YD = WIN_YD_ORG                          # 제어 윈도우 아래쪽 Y값
        if keyBoard == ord('c') or keyBoard == ord('C'):     # 클리어 View/메뉴 View
            MENU_VIEW = not MENU_VIEW
        if keyBoard == ord('a') or keyBoard == ord('A'):     # 전조등 자동/수동 점등 
            AUTO_LIGHT = not AUTO_LIGHT
        if keyBoard == ord('l') or keyBoard == ord('L'):     # 전조등 ON/OFF 점등 
            MANUAL_LIGHT = not MANUAL_LIGHT
        if keyBoard == ord('v') or keyBoard == ord('V'):     # 속도-시간 가변/고정
            VARIABLE_MODE = not VARIABLE_MODE
        # 수동 차량 조작 ----------------------------------------------------------------------
        if keyBoard == 82:                                   # 전진 Arrow Up
            recordOn = False; mouseCtrlState = False
            if preKey == ord(' '): preKey = keyBoard; mL = 80; mR = 80; cursorColor = GREEN
            else: preKey = ord(' '); mL = 0; mR = 0; cursorColor = YELLOW
            motorRun(mL, mR)
        #------------------------------------------------------------------------------------
        if keyBoard == 84:                                   # 후진 Arrow Down
            recordOn = False; mouseCtrlState = False
            if preKey == ord(' '): preKey = keyBoard; mL = -70; mR = -70; cursorColor = GREEN
            else: preKey = ord(' '); mL = 0; mR = 0; cursorColor = YELLOW
            motorRun(mL, mR)
        #------------------------------------------------------------------------------------
        if keyBoard == 81:                                   # 좌 회전 Arrow Left
            recordOn = False; mouseCtrlState = False
            if preKey == ord(' '): preKey = keyBoard; mL = -70; mR = 70; cursorColor = GREEN
            else: preKey = ord(' '); mL = 0; mR = 0; cursorColor = YELLOW
            motorRun(mL, mR)
        #------------------------------------------------------------------------------------
        if keyBoard == 83:                                   # 우 회전 Arrow Right
            recordOn = False; mouseCtrlState = False
            if preKey == ord(' '): preKey = keyBoard; mL = 70; mR = -70; cursorColor = GREEN
            else: preKey = ord(' '); mL = 0; mR = 0; cursorColor = YELLOW
            motorRun(mL, mR)
        #------------------------------------------------------------------------------------
        if keyBoard == ord(' '):                             # Space 정지
            recordOn = False; mouseCtrlState = False; cursorColor = YELLOW
            preKey = ord(' '); mL = 0; mR = 0
            motorRun(mL, mR)
    #----------------------------------------------------------------------------------------
    nx.lamp(0,0,0)                          # 전조등 Off
    MOTOR_L.stop()                          # 왼쪽 모터 PWM(펄스 폭 변조) 정지
    MOTOR_R.stop()                          # 오른쪽 모터 PWM(펄스 폭 변조) 정지
    GPIO.cleanup()                          # GPIO 초기화
    cam.release()                           # 카메라 자원을 반납
    cv.destroyAllWindows()                  # 열려 있는 모든 윈도우를 닫기

#============================================================================================
if __name__ == '__main__':
    '''
    커맨드 라인에서 직접 gi.py 를 입력할 때 실행 됩니다. 
    라이브러리 형태로 gi.py 가 호출될 때 이 부분은 실행되지 않습니다. 
    이미지 편집 및 딥러닝 프로그램 dl.py 에서 조향각 범위 데이터를 가져오기 위하여 
    gi.py 를 라이브러리 형식으로 호출합니다.

    커맨드 라인에서 gi.py 실행 방법과 argv 관계

    pi@raspberrypi:~/donaldCar $                      # 
    pi@raspberrypi:~/donaldCar $ gi.py                #  argv = 1 이며 연습
    pi@raspberrypi:~/donaldCar $ gi.py wayA           #  argv = 2 이며 이미지 수집 가능
    pi@raspberrypi:~/donaldCar $ gi.py wayA 2000      #  argv = 3 이며 지정한 번호부터 이미지 수집

    '''
    # OpenCV 라이브러리 버젼 표시
    print('\n')
    print('OpenCV Version: ', cv.__version__)
    print('\n')

    # 연속 촬영 이미지 번호 지정 됬으면 시작 번호 저장 ex) python3 gi wayA 2000
    if len(sys.argv) >= 3: 
        fileId = int(sys.argv[2])   # 이미지 시작 번호 지정

    # 이미지 저장 디렉토리 지정 되었는지 확인
    if len(sys.argv) >= 2:
        t = sys.argv[1]             # 이미지 저장 디렉토리

        DIRECTORY_FLAG = True
        if not os.path.exists(t):
            print(f'이미지 저장 디렉터리 {t} 가 없습니다.')
            b = input('새로 생성 할까요? < Y: Yes>  <Any Key: No>')
            if b == 'y' or b == 'Y':       # 대문자 'Y', 소문자 'y' 모두 사용 가능
                os.mkdir(t)                # 디렉토리 생성 Sudo 모드로 생성
                imageDir = t
            else:
                DIRECTORY_FLAG = False     # 디렉토리 생성을 취소 했을 때 
        else:
            imageDir = t

        if DIRECTORY_FLAG:
            filePath = './'+imageDir+'/'
            print('File Path: ', filePath)

            pathParmeter = filePath+'_'+imageDir+'_track.pickle'  # pickle 파일 이름 

            if os.path.exists(pathParmeter):                      # pickle 파일 있으면 읽어들임
                with open(pathParmeter, 'rb') as fr:
                    d = pickle.load(fr)
                    WIN_XL = d[0]; WIN_XR = d[1]; WIN_YU = d[2]; WIN_YD = d[3] # 카메라 영역 정보
                    AUTO_LIGHT = d[4]; MANUAL_LIGHT = d[5]        # 전조등 정보 
                    if len(sys.argv) < 3: fileId = d[6]   # 커맨드 라인에서 파일 시작번호가 없을 때
                print('차선 검출 영역 데이터를 읽었습니다:', pathParmeter)

    # Mouse Event ---------------------------------------------------------------------------
    cv.namedWindow('Out')                              # 윈도우 창을 생성합니다.
    cv.setMouseCallback('Out', controlMain, viewWin)   # 마우스 제어 설정
    #----------------------------------------------------------------------------------------
    main()                                             # 이미지 수집 프로그램 실행
    import subprocess
    if DIRECTORY_FLAG:        
        subprocess.call(['sudo', 'chown', 'pi:pi', imageDir])     # 디렉토리를 root 계정에서 pi 계정으로 전환 
        if len(sys.argv) >= 2:
            w = [WIN_XL, WIN_XR, WIN_YU, WIN_YD, AUTO_LIGHT, MANUAL_LIGHT, fileId]  # ar.py 호환성
            with open(pathParmeter, 'wb') as fw:       # 디렉토리가 지정된 경우 차선 검출 영역 데이터 저장
                pickle.dump(w, fw)
            print('카메라 영역, 램프 상태, 파일 일련 번호를 저장했습니다:', pathParmeter)

        print('WIN_XL:', WIN_XL, 'WIN_XR:', WIN_XR,'WIN_YU:', WIN_YU, 'WIN_YD:', WIN_YD)
        print('file Id:', fileId)

#############################################################################################


