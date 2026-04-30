#############################################################################################
#
# Auto Drive Car  V2.0                                                          <ar.py>
#
# PyTorch 딥러닝 프레임 웍에서 학습된 모델파일에 의하여 자유주행 합니다.
#
# $ python ar.py <이미지 디렉터리>
#
# ★ V2.0 추가 기능 (fa.py 모듈 연동)
#   - 앞 차 추종: ToF 거리에 따라 자동 속도 조절 (감속/정지)
#   - 장애물 회피: 급격한 거리 감소 감지 → 자동 회피 기동
#   - 차선 복귀:  회피 후 딥러닝 조향각으로 자동 복귀
#
# G, [Home]: 자율 주행을 시작합니다.
# X, [Esc]: 주행을 종료하고 시스템(Command Window)으로 복귀합니다.
# C: 화면의 보여지는 정보를 최소화 하거나 최대로 보여 줍니다.
# D: ToF 거리감지 센서의 동작을 선택하거나 해지 합니다.
# H: 경적 소리 나게 합니다.
# L: 전조등(Neopixel)의 동작 상태를 선택합니다.
#    ON, Off, Police 1, Police 2, Police 3, Ambulance, Fire Truck, Patrol
#        gi.py 에서 Lamp = On 설정하면 On 으로 동작합니다.
# A: 전조등이 자동으로 점등 하거나 수동으로 점등 하도록 선택합니다.
#        gi.py 에서 Auto = True 설정하면 자동 점등 모드로 동작합니다.
#        자동 모드에서 CdS 빛감지 센서에 의하여 어두우면 전조등 On 밝으면 전조등 Off 됩니다.
# [End]: 자율주행 종료합니다.
# [Up]: 전진
# [Left]: 제자리 좌회전
# [Right]: 제자리 우회전
# [Down]: 후진
#
#
# 학습 모델파일(*.pt)을 지정하지 않거나 없으면 차량은 수동 조작만 가능합니다.
#
# Control Key
# 80: Home
# 81: Left Arrow
# 82: Up Arrow
# 83: Right Arrow
# 84: Down Arrow
# 85: Page Up
# 86: Page Down
# 87: End
#
# 파이토치 최신 버젼에서 view 함수가 reshape 로 변경
#
# FEB 2 2022 / APR 2026 (V2.0 fa 모듈 추가)
#
# SAMPLE Electronics co.
# http://www.ArduinoPLUS.cc
#
#############################################################################################
HORN_DISTANCE = 350                   # 경적이 울리는 거리
STOP_DISTANCE = 110                   # 차량 정지 거리 (차량 정지, 자율 주행과 수동 전진에서 해당)
RUN_SPEED = 70                        # 60, 70, 80, 90  차량 주행 속도(speed)
ANGLE_GAIN = 0.8                      # 0.7, 0.8, 0.9, 1.0, 1.1  조향각 이득(gain)
downLoads = 'Downloads'               # 디폴트 Down Load 폴더
#--------------------------------------------------------------------------------------------
imageDir = ''                         # 기본 디렉토리 이름
modelName = '_model_check.pt'         # 모델 파일 
model = None                          # 딥러닝 모델
modelFile = None                      # 딥러닝 생성 모델 파일 이름
MODEL_FILE = False                    # 딥러닝 모델 파일
lightMode = 'MANU'                    # Head Light 동작모드 Auto:자동, Manual:수동 
lightCar = ' '                        # 자동차 종류(램프 동작)
#--------------------------------------------------------------------------------------------
RED     =   0,  0,255                 # Red
GREEN   =   0,255,  0                 # Green
BLUE    = 255,  0,  0                 # Blue
MAGENTA = 255,  0,255                 # Magenta(Pink)
CYAN    = 255,255,  0                 # CYAN(Sky Blue)
YELLOW  =   0,255,255                 # Yellow
WHITE   = 255,255,255                 # White
BLACK   =   0,  0,  0                 # Black
GRAY    = 128,128,128                 # Gray
#--------------------------------------------------------------------------------------------
import VL53L0X                        # ToF Ranger Library
import ts                             # Traffic Signal Library
import NEOPIXEL as nx                 # Adafruit Neopixel Library
import fa                             # ★ V2.0 앞 차 추종 + 장애물 회피 모듈
#--------------------------------------------------------------------------------------------
import os                             # File 읽기(Read) 쓰기(Write)와 관련된 라이브러리
import sys                            # 커맨드 라인 처리를 위한 라이브러리
import time                           # 시간(시계) 정보 라이브러리
import glob                           # 검사조건을 통과한 화일 이름 리스트 생성 라이브러리
import pickle                         # 파이썬 객채 저장과 읽기 라이브러리
import cv2 as cv                      # OpenCV 그래픽 라이브러리 version 4.5.1
import numpy as np                    # OpenCV의 이미지 데이터 저장 배열변수 라이브러리
import RPi.GPIO as GPIO               # Raspberry Pi GPIO 핀 라이브러리
import torch                          # Pytorch
import torch.nn as nn                 # Torch CNN
# Traffic Signal Parameter ------------------------------------------------------------------
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
# Green -------------------------------------------------------------------------------------
lineLeadGREENX = 0                          # 시작(Lead) 라인과 외곽 원 교차 X 좌표
lineLeadGREENY = 0                          # 시작(Lead) 라인과 외곽 원 교차 Y 좌표
lineTailGREENX = 0                          # 종료(Tail) 라인과 외곽 원 교차 X 좌표
lineTailGREENY = 0                          # 종료(Tail) 라인과 외곽 원 교차 Y 좌표
HSV_hueLeadGREEN = int(lineLeadAngGREEN/2)  # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
HSV_hueTailGREEN = int(lineTailAngGREEN/2)  # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
HSV_saturationGREEN = saturationHalfGREEN*2 # HSV 색상 좌표계에서 Saturation(/256)
centerHueGREEN = 60
# Red ----------------------------------------------------------------------------------------
lineLeadREDX = 0                            # 시작(Lead) 라인과 외곽 원 교차 X 좌표
lineLeadREDY = 0                            # 시작(Lead) 라인과 외곽 원 교차 Y 좌표
lineTailREDX = 0                            # 종료(Tail) 라인과 외곽 원 교차 X 좌표
lineTailREDY = 0                            # 종료(Tail) 라인과 외곽 원 교차 Y 좌표
HSV_hueLeadRED = int(lineLeadAngRED/2)      # HSV 색상 좌표계에서 시작(Lead) 라인 각도(/180)
HSV_hueTailRED = int(lineTailAngRED/2)      # HSV 색상 좌표계에서 종료(Tail) 라인 각도(/180)
HSV_saturationRED = saturationHalfRED*2     # HSV 색상 좌표계에서 Saturation(/256)
centerHueRED = 0
#--------------------------------------------------------------------------------------------
# 차선 감지 영역 
WIN_XL = 0                            # [0] 제어 윈도우 왼쪽 X 값
WIN_XR = 639                          # [1] 제어 윈도우 오른쪽 X 값
WIN_YU = 246                          # [2] 제어 윈도우 위쪽 Y 값
WIN_YD = 459                          # [3] 제어 윈도우 아래쪽 Y값
#--------------------------------------------------------------------------------------------
# Raspberry Pi GPIO Pin Number
MOTOR_L_PWM = 12                      # GPIO.12    왼쪽 모터 펄스폭 변조
MOTOR_L_DIR = 5                       # GPIO.5     원쪽 모터 방향
MOTOR_R_PWM = 13                      # GPIO.13    오른쪽 모터 펄스폭 변조
MOTOR_R_DIR = 6                       # GPIO.6     오른쪽 모터 방향
BUZZER = 23                           # GPIO.23    경적
MUSIC = 24                            # GPIO.24    후진 사운드
LAMP_R_YELLOW = 20                    # Right Yellow Lamp
LAMP_L_YELLOW = 26                    # Left Yellow Lamp
LAMP_BRAKE = 21                       # 브레이크 Lamp
LIGHT_SENSOR = 25                     # 빛 감지 센서(CdS)
# Raspberry Pi GPIO Pin 설정
GPIO.setwarnings(False)               # GPIO 관련 경고 메시지 출력 금지
GPIO.setmode(GPIO.BCM)                # BCM 핀 번호
GPIO.setup(MOTOR_L_PWM,GPIO.OUT)      # 왼쪽 모터 펄스폭
GPIO.setup(MOTOR_L_DIR,GPIO.OUT)      # 왼쪽 모터 방향
GPIO.setup(MOTOR_R_PWM,GPIO.OUT)      # 오른쪽 모터 펄스폭
GPIO.setup(MOTOR_R_DIR,GPIO.OUT)      # 오른쪽 모터 방향
GPIO.setup(BUZZER,GPIO.OUT)           # 경적 
GPIO.setup(MUSIC,GPIO.OUT)            # 후진 사운드 
GPIO.setup(LAMP_R_YELLOW,GPIO.OUT)    # 우회전 깜빡이 램프
GPIO.setup(LAMP_L_YELLOW,GPIO.OUT)    # 좌회전 깜빡이 램프
GPIO.setup(LAMP_BRAKE,GPIO.OUT)       # 브레이크 램프
GPIO.setup(LIGHT_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP) # 빛감지 센서(CdS) 입력, 풀-업 모드
# DC Motor 설정 ------------------------------------------------------------------------------
MOTOR_L = GPIO.PWM(MOTOR_L_PWM,500)   # 왼쪽 모터 PWM(펄스폭 변조) 주파수 500Hz
MOTOR_R = GPIO.PWM(MOTOR_R_PWM,500)   # 오른쪽 모터 PWM(펄스폭 변조) 주파수 500Hz
MOTOR_L.start(0)                      # 왼쪽 모터 PWM(펄스폭 변조) 값 0 으로 시작
MOTOR_R.start(0)                      # 오른쪽 모터 PWM(펄스폭 변조) 값 0 으로 시작
#--------------------------------------------------------------------------------------------
tof = VL53L0X.VL53L0X()               # 거리 감지 센서 객체 설정
#--------------------------------------------------------------------------------------------
viewWin = np.zeros((480,800,3),np.uint8)  # 표시되는 윈도우 가로, 세로, 컬러층, 8비트
msgBoxL = cv.imread('./_IMAGE/armL.png',cv.IMREAD_COLOR)
msgBoxR = cv.imread('./_IMAGE/armR.png',cv.IMREAD_COLOR)
#--------------------------------------------------------------------------------------------
cv.namedWindow('Out',cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty('Out', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
#--------------------------------------------------------------------------------------------
fileList = []             # os.listdir(dataDir)  # 데이터 디렉토리 내의 이미지 파일 이름을 리스트로 반환
#--------------------------------------------------------------------------------------------
class NvidiaModel(nn.Module):
    def __init__(self):
        super(NvidiaModel, self).__init__()
        # elu=Expenential Linear Unit, similar to leaky Relu
        # skipping 1st hiddel layer (nomralization layer), as we have normalized the data
        # Convolution Layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3)),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ELU(inplace=True)
        )
        # Fully Connected Layers
        self.layer2 = nn.Sequential(
            # nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(in_features=18 * 64, out_features=100),
            nn.ELU(inplace=True),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(inplace=True),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(inplace=True)
        )
        # Output Layer
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=10, out_features=1)
        )
    def forward(self, x):
        x = self.layer1(x)
        #x = x.view(x.shape[0], -1)
        x = x.reshape(x.shape[0], -1)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
# Motor Run Function ------------------------------------------------------------------------
def motorRun(leftMotor, rightMotor):
    '''
    왼쪽 모터와 오른쪽 모터에 PWM 값을 설정합니다.
    부호가 음수(-)이면 후진, 양수(+)이면 전진합니다.
    PWM 값의 범위는 0~100 까지 입니다.
    '''
    amotorL = abs(leftMotor)                   # 왼쪽 모터 PWM 절대값
    if amotorL>100:                            # PWM 값이 100 보다 크면 100 으로 조정
        amotorL = 100
    amotorR = abs(rightMotor)                  # 오른쪽 모터 PWM 절대값
    if amotorR>100:                            # PWM 값이 100 보다 크면 100 으로 조정
        amotorR = 100
    #----------------------------------------------------------------------------------------
    if leftMotor >= 0:                         # 왼쪽 모터의 방향 설정
        GPIO.output(MOTOR_L_DIR,GPIO.HIGH)     # 왼쪽 모터 전진
    else:
        GPIO.output(MOTOR_L_DIR,GPIO.LOW)      # 왼쪽 모터 후진
    MOTOR_L.ChangeDutyCycle(amotorL)           # 왼쪽 모터 PWM 값 설정

    if rightMotor >= 0:                        # 오른쪽 모터의 방향 설정
        GPIO.output(MOTOR_R_DIR,GPIO.HIGH)     # 오른쪽 모터 전진
    else:
        GPIO.output(MOTOR_R_DIR,GPIO.LOW)      # 오른쪽 모터 후진
    MOTOR_R.ChangeDutyCycle(amotorR)           # 오른쪽 모터 PWM 값 설정
#============================================================================================
def main():

    global lightMode, lightCar

    # 상수 값 설정
    RESTART_NUM = 33             # Red 신호 유지 시간(frame단위)
    YM = 240                     # 모터 바 그래프 중앙 위치 (왼쪽, 오른쪽 공통 적용)
    nx.lamp(0,0,0)               # 전조등 끄기
    camera=cv.VideoCapture(0,cv.CAP_V4L) # 카메라 객체 생성
    camera.set(3, 640)           # 카메라 비디오 X(가로) 크기, 표준 VGA
    camera.set(4, 480)           # 카메라 비디오 Y(세로) 크기, 표준 VGA
    # 변수 초기 설정
    VIEW_FLAG = True             # 화면 표시 모드
    distance  = 9999             # ★ V2.0 ToF 거리 초기값 (fa 모듈 공유 변수)
    # ★ V2.0 앞 차 추종 + 장애물 회피 모듈 초기화
    #    avoidRight=True  → 오른쪽으로 회피 (반대 차선이 오른쪽이면 False 로 변경)
    fa.init(baseSpeed=RUN_SPEED, avoidRight=True)
    autoRun = False              # 자동차 동작 상태 (True:자율주행 False:정지)
    BLINK_LEFT = False           # 왼쪽 깜빡이
    BLINK_RIGHT = False          # 오른쪽 깜빡이
    TOF_FLAG = False             # 거리 감지 센서 동작 상태
    YUV = False                  # NVIDIA Cell 디스프레이 모드
    CDS = 'NIGHT'                # CdS 의 상태기록 Day:밝음, Night:어두움 
    preKey = ord(' ')            # 차량 초기 상태를 정지 상태로 설정
    PWM_COLOR = GREEN            # 모터 PWM 그래프 색상, GREEN: 수동 조작, RED: 자율 주행
    mL = 0                       # 왼쪽 모터 속도, PWM 값 -100 ~ 0 ~ +100
    mR = 0                       # 오른쪽 모터 속도, PWM 값 -100 ~ 0 ~ +100
    signRedDelay = 0             # 적색 신호가 감지 됬을 때 0 보다 큰 수로 설정, 0 이면 주행 
    angle = 0                    # 조향 각도
    timeCycle = 0                # 실행 주기 시간 (카메라 frame) 
    horn = 0                     # 경적 트리거 (1=울리기 시작, 0=정지)
    hornTimer = 0                # 경적 3회 패턴 프레임 카운터
    blinkCount = 0               # 깜빡이 카운트

    #----------------------------------------------------------------------------------------
    while(camera.isOpened()):
        startCycle = time.time()                       # 프레임 시작 시각
        if GPIO.input(LIGHT_SENSOR): CDS = 'NIGHT'     # 빛 감지 센서 
        else: CDS = 'DAY'
        if lightMode == 'AUTO' and CDS == 'NIGHT': nx.lamp(255,255,255)  # 전조등 자동 On
        elif lightCar == 'ON':  nx.lamp(255,255,255)   # 전조등 수동 On
        elif lightCar == 'POLICE 1': nx.policeCar1()   # Police Car 1
        elif lightCar == 'POLICE 2': nx.policeCar2()   # Police Car 2
        elif lightCar == 'POLICE 3': nx.policeCar3()   # Police Car 3
        elif lightCar == 'AMBULANCE': nx.ambulance()   # Ambulance Car
        elif lightCar == 'FIRE TRUCK': nx.fireTruck()  # Fire Truck
        elif lightCar == 'PATROL': nx.patrol()         # Patrol Car
        elif lightCar == 'OFF' or lightCar == ' ': nx.lamp(0,0,0)        # 전조등 Off
        # 경적 (3회 삐-삐-삐 패턴: 5프레임 ON + 4프레임 OFF × 3회) -------------------------
        if horn:
            hornTimer += 1
            phase = (hornTimer - 1) % 9               # 9프레임 1사이클 (5 ON + 4 OFF)
            if phase < 5:
                GPIO.output(BUZZER, GPIO.HIGH)
            else:
                GPIO.output(BUZZER, GPIO.LOW)
            if hornTimer >= 27:                        # 9프레임 × 3회 = 27프레임 완료
                horn = 0; hornTimer = 0
                GPIO.output(BUZZER, GPIO.LOW)
        else:
            hornTimer = 0
            GPIO.output(BUZZER, GPIO.LOW)              # 경적 끄기
        # 카메라로 부터 1 프레임 영상 가져오기 ---------------------------------------------------
        _, frame = camera.read() #frame = cv.flip(frame,-1) # 입력 이미지를 필요하면 상하 반전
        viewWin[0:480,80:720] = frame[0:480,0:640]     # 480 x 640 VGA
        viewWin[0:480,0:0+80] = msgBoxL                # 왼쪽 메시지 박스
        viewWin[0:480,720:720+80] = msgBoxR            # 오른쪽 메시지 박스
        image = viewWin[WIN_YU:WIN_YD,WIN_XL+80:WIN_XR+80]   # 트랙 이미지
        image = cv.resize(image,(200,66))              # NVIDEA Image 포맷 (X:200, Y:66)
        imageYUV = cv.cvtColor(image, cv.COLOR_BGR2YUV)# RGB 색상을 YUV 색상으로 변환
        imageYUVG = cv.GaussianBlur(imageYUV, (3,3), 0) # Gauss 노이즈 제거
        procImg = imageYUVG/255            # 0-255 범위(정수)를 0.000-0.999 범위(유리수)로 변환
        # 신호등 인식 -------------------------------------------------------------------------
        # 적색 신호등이 인식되면 signRedDelay 에 non zero(RESTART_NUM) 를 저장한다.
        # signRedDelay 이 0 이 아니면 차량은 정지하고 0 이 되면 주행 시작한다.
        # signRedDelay 은 1 프레임 에서 1 씩 감소하여 0 이 된다. 
        v = ts.trafficSign(viewWin,          # Y480 x X800 전체 LCD 스크린
                        VIEW_FLAG,            # 신호등 데이터 디스프레이
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

        if v == 'G': signRedDelay = 0                  # 녹색 신호등
        if v == 'R' and preKey == 80: 
            signRedDelay = RESTART_NUM                 # 자율 주행 모드에서 적색 신호등이면 정지
        if signRedDelay > 0:                           # 적색 신호등 이후에 일정시간 지나면 
            signRedDelay -= 1                          # 녹색 신호등이 없어도 주행 시작
        if signRedDelay > 0:                           # 적색 신호등 소등 이후 카운트 표시
            cv.putText(viewWin,f'{signRedDelay:2d}',(750,110),cv.FONT_HERSHEY_PLAIN,1,YELLOW)
        # 방향 지시등 -------------------------------------------------------------------------
        blinkCount += 1
        blinkCount %= 10                               # 10 frame 주기
        if blinkCount > 5:
            if BLINK_LEFT: GPIO.output(LAMP_L_YELLOW,GPIO.HIGH); viewWin[350:380,10:70] = YELLOW
            if BLINK_RIGHT: GPIO.output(LAMP_R_YELLOW,GPIO.HIGH); viewWin[350:380,730:790] = YELLOW
        else:
            GPIO.output(LAMP_R_YELLOW,GPIO.LOW); GPIO.output(LAMP_L_YELLOW,GPIO.LOW) # 노란색 끄기
        # 전방 거리 측정, 경적 발생, 차량 정지 ---------------------------------------------------
        if TOF_FLAG:
            distance = tof.get_distance()                # VL53L0X TOF 거리 센서 측정 값 가져오기
            fa.measureDistance(distance)                 # ★ V2.0 fa 모듈에 거리 전달
            if VIEW_FLAG:
                if distance < 2000:
                    cv.putText(viewWin, f'{distance:3d} MM', (320,165),cv.FONT_HERSHEY_COMPLEX_SMALL,2,WHITE)
                else:
                    cv.putText(viewWin, f'--- MM', (320,165),cv.FONT_HERSHEY_COMPLEX_SMALL,2,WHITE)
            # 경적 구간에 들어오면 스크린에 정지 표시하고 경적 소리 출력 -----------------------------
            if 20 < distance < HORN_DISTANCE:            # 20mm 이하는 센서 노이즈(사거리 등) 무시
                cv.circle(viewWin, (400,160), 100, RED, 25)  # 정지 표시
                cv.line(viewWin, (400+70,160-70), (400-70,160+70), RED, 25)
                GPIO.output(LAMP_BRAKE,GPIO.HIGH)        # 브레이크 적색 램프 점등
                if preKey == 80 or preKey == 82:         # 자율 주행(G, Home) 또는 전진(Arrow Up)
                    if not horn: horn = 1                # 아직 울리지 않는 경우에만 3회 시작
                if distance < STOP_DISTANCE:             # 정지 거리
                    horn = 0; hornTimer = 0              # 경적 Off
                    if (preKey == 80 or preKey == 82):   # 자율 주행(80) 또는 수동 주행(82) 확인
                        signRedDelay = RESTART_NUM       # 주행 정지
                        BLINK_LEFT = True; BLINK_RIGHT = True
            else:
                GPIO.output(LAMP_BRAKE,GPIO.LOW)         # 브레이크 적색 램프 끄기
        #------------------------------------------------------------------------------------
        if VIEW_FLAG:
            if MODEL_FILE:
                cv.putText(viewWin, f'{imageDir}',(90,20),cv.FONT_HERSHEY_COMPLEX_SMALL,1,MAGENTA)
            cv.putText(viewWin, f'{timeCycle:4d} mS',(340,20),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
            cv.putText(viewWin, f'{lightMode} {CDS} {lightCar}',(510,475),cv.FONT_HERSHEY_PLAIN,1,GREEN)
            cv.putText(viewWin, 'Lane Detect Area', (WIN_XL+90,WIN_YU+18),cv.FONT_HERSHEY_PLAIN,1,YELLOW)
            cv.rectangle(viewWin,(WIN_XL+80,WIN_YU),(WIN_XR+80,WIN_YD),YELLOW,1)  # 차선 검색 영역 표시
            cv.putText(viewWin, 'Traffic Siginal Area', (TSWIN_XL+90,TSWIN_YU+18),cv.FONT_HERSHEY_PLAIN,1,CYAN)
            cv.rectangle(viewWin,(TSWIN_XL+80,TSWIN_YU),(TSWIN_XR+80,TSWIN_YD),CYAN,1)  # 신호등 검색 영역 표시

            if YUV:
                viewWin[407:407+66,300:300+200] = imageYUVG
            else:
                viewWin[407:407+66,300:300+200] = image
            if (lightMode == 'AUTO' and CDS == 'NIGHT') or lightCar == 'ON':
                cv.rectangle(viewWin,(300-2,407-2),(300+200+2,407+66+2),WHITE,3)  # Lamp On
            # ★ V2.0 fa 모듈: 추종/회피 상태 오버레이 (왼쪽 패널)
            fa.drawStatus(viewWin, distance, mL, mR)
        # 자율 주행 모드 조향 각도 검출 ---------------------------------------------------------
        if MODEL_FILE and autoRun:    # 학습 파일이 존재하고 자율주행 모드일 때 실행
            X = np.asarray([procImg])
            X = torch.Tensor(X).permute(0, 3, 1, 2)
            angle = 0
            model.eval()
            with torch.no_grad():
                angle = int(model(X)[0] * ANGLE_GAIN)
            if VIEW_FLAG:
                cv.putText(viewWin, f'{angle:3d}', (260,380),cv.FONT_HERSHEY_COMPLEX_SMALL,6,WHITE)
            # ★ V2.0 fa 모듈: 앞 차 추종 속도 조절 + 장애물 회피 + 차선 복귀
            #    fa.update() 가 최종 mL, mR, 조향각을 모두 계산하여 반환합니다.
            mL, mR, angle = fa.update(distance, angle, autoRun)
        # 왼쪽 모터와 오른쪽 모터의 속도 크가 값(PWM)을 바 그래프로 표시 -----------------------------
        if autoRun: 
            PWM_COLOR = RED            # 자율 주행에서 그래프 색상
        else: 
            PWM_COLOR = GREEN          # 수동 조정에서 그래프 색상
        # 왼쪽 모터 PWM 그래프 ------------------------------------------------------------------
        if mL >= 0: 
            ys = YM - mL
            ye = YM
        else: 
            ys = YM
            ye = YM - mL  
        cv.rectangle(viewWin,(30,ys),(50,ye),PWM_COLOR,-1)
        # 오른쪽 모터 PWM 그래프 ----------------------------------------------------------------
        if mR >= 0: 
            ys = YM - mR
            ye = YM
        else: 
            ys = YM
            ye = YM - mR  
        cv.rectangle(viewWin,(750,ys),(770,ye),PWM_COLOR,-1)
        # 모터 PWM 값 표시 --------------------------------------------------------------------
        cv.putText(viewWin, f'{mL:3d}', (15,340),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        cv.putText(viewWin, f'{mR:3d}', (735,340),cv.FONT_HERSHEY_COMPLEX_SMALL,1,WHITE)
        # 자율 주행 모드에서 Red 신호(0 보다 큰 수)일 때 정지 --------------------------------------
        if signRedDelay > 0:
            if autoRun or preKey == 82:           # 자율주행 모드일 또는 커서 Up 키
                mL = 0; mR = 0
        # 왼쪽 모터, 오른쪽 모터 동작 -----------------------------------------------------------
        motorRun(mL, mR)                          # PWM 값 지정 
        # 깜빡이 설정 -------------------------------------------------------------------------
        if preKey == 84: pass                     # Backward Key
        else:
            if abs(mL - mR) > 9:
                if (mL - mR) > 0: 
                    BLINK_LEFT = False; BLINK_RIGHT = True
                else: 
                    BLINK_LEFT = True; BLINK_RIGHT = False 
            else: 
                BLINK_LEFT = False; BLINK_RIGHT = False
        # 윈도우 디스프레이 --------------------------------------------------------------------
        cv.imshow('Out', viewWin)
        # 수동 조작 모드, 파라메터 설정 ---------------------------------------------------------
        keyBoard = cv.waitKey(1)                              # 1 mSec 시간 지연하고 키값 읽기 
        # print(hex(keyBoard), keyBoard)
        if keyBoard == 0x1B or keyBoard == 0x09 or keyBoard == ord('x') or keyBoard == ord('X'):
            break                                             # ESC, TAB, X 프로그램 종료
        elif keyBoard == 80 or keyBoard == ord('g') or keyBoard == ord('G'): # Home, G 자율 주행 시작
            autoRun = True; signRedDelay = 0; preKey = 80
        elif keyBoard == ord('c') or keyBoard == ord('C'):    # 카메라, 신호등 영역 표시
            VIEW_FLAG = not VIEW_FLAG
        elif keyBoard == ord('h') or keyBoard == ord('H'):    # 경적
            horn = 1; hornTimer = 0                           # 3회 경적 시작
        elif keyBoard == ord('l') or keyBoard == ord('L'):    # 전조등 매뉴얼 모드
            if lightCar == ' ': lightCar = 'ON'
            elif lightCar == 'ON': lightCar = 'OFF'
            elif lightCar == 'OFF': lightCar = 'POLICE 1'
            elif lightCar == 'POLICE 1': lightCar = 'POLICE 2'
            elif lightCar == 'POLICE 2': lightCar = 'POLICE 3'
            elif lightCar == 'POLICE 3': lightCar = 'AMBULANCE'
            elif lightCar == 'AMBULANCE': lightCar = 'FIRE TRUCK'
            elif lightCar == 'FIRE TRUCK': lightCar = 'PATROL'
            elif lightCar == 'PATROL': lightCar = ' '
        elif keyBoard == ord('a') or keyBoard == ord('A'):    # 전조등 점등 모드
            if lightMode == 'MANU': 
                lightMode = 'AUTO'                            # 전조등 자동 점등 
            elif lightMode == 'AUTO': 
                lightMode = 'MANU'                            # 전조등 수동 점등
        elif keyBoard == ord('d') or keyBoard == ord('D'):    # ToF Range Sensor
            TOF_FLAG = not TOF_FLAG
            if TOF_FLAG: 
                tof.start_ranging(VL53L0X.VL53L0X_HIGH_SPEED_MODE) # 거리센서 동작 시작
            else: 
                tof.stop_ranging()                            # 거리 센서 정지
        elif keyBoard == ord('y') or keyBoard == ord('Y'):    # NVIDIA YUV 디스프레이
            YUV = not YUV                                     # RGB <-> YUV 전환
        # 수동 차량 조정 ----------------------------------------------------------------------
        elif keyBoard == 82:                                  # 전진 Arrow Up(82)
            autoRun = False 
            if preKey == ord(' '): 
                mL = 80; mR = 80; signRedDelay = 0; preKey = keyBoard
            else: 
                preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC,GPIO.LOW)
        elif keyBoard == 84:                                  # 후진 Arrow Down(84)
            autoRun = False 
            if preKey == ord(' '): 
                mL = -70; mR = -70; signRedDelay = 0; preKey = keyBoard
                GPIO.output(MUSIC,GPIO.HIGH)                  # 후진 멜로디
                BLINK_LEFT = True; BLINK_RIGHT = True         # 비상등
            else: 
                preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC,GPIO.LOW)
        elif keyBoard == 81:                                  # 좌회전 Left(81)
            autoRun = False 
            if preKey == ord(' '): 
                mL = -70; mR = 70; signRedDelay = 0; preKey = keyBoard
            else: 
                preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC,GPIO.LOW)
        elif keyBoard == 83:                                  # 우회전 Right(83)
            autoRun = False 
            if preKey == ord(' '): 
                mL = 70; mR = -70; signRedDelay = 0; preKey = keyBoard
            else: 
                preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC,GPIO.LOW)
        elif keyBoard == ord(' ') or keyBoard == 87:          # [Spacd] [End]
            autoRun = False; preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC,GPIO.LOW)
            
        endCycle = time.time()                                # 프레임 종료 시각
        timeCycle = int((endCycle - startCycle)*1000)         # 프레임 소요 시간
    #----------------------------------------------------------------------------------------
    if TOF_FLAG: 
        tof.stop_ranging()            # 거리센서 사용중이면 사용정지
    motorRun(0, 0)                    # 모터 정지
    nx.lamp(0,0,0)                    # 전조등 소등
    GPIO.cleanup()                    # GPIO 모듈의 점유 리소스 해제
    cv.destroyAllWindows()            # 열려진 모든 윈도우 닫기
#============================================================================================
if __name__ == '__main__':

    print('\n')
    print('Python Version:     ', sys.version)
    print('OpenCV Version:     ', cv.__version__)
    print('Pytorch Version:    ', torch.__version__)
    print('\n')

    if len(sys.argv) >= 2:
        imageDir = sys.argv[1]        # Track Image 저장 디렉토리
    
    print('imageDir', imageDir)

    ################################################################
    # 사용자 이름
    username = os.getlogin()
    # Downloads 안에 있는 모델 *.pt 파일 리스트 생성
    file_listD = glob.glob(f'/home/{username}/{downLoads}/_{imageDir}_*.pt')
    current_directory = os.getcwd()  # 현재 작업 디렉토리
    # 프로젝트 디렉토리에 있는 모델 *.pt 파일 리스트 생성
    file_listC = glob.glob(os.path.join(current_directory,f'{imageDir}/_{imageDir}_*.pt'))
    # Downloads 디렉토리의 파일과 프로젝트 디렉토리 내의 화일을 병합
    file_list = file_listD + file_listC
    # 파일의 수정 시간을 가져와서 (파일 경로, 수정 시간) 형태로 저장
    file_times = [(file, os.path.getmtime(file)) for file in file_list]
    # 수정 시간을 기준으로 파일을 정렬 - 최근 만들어진 파일이 마지막에 오도록 정열한다.
    sorted_files = sorted(file_times, key=lambda x: x[1], reverse=False)
    # 파일을 LCD 에 디스프레이 한다. ------------------------------------------------------------
    def listingFile(sorted_files, startN, count, cline):
        viewWin[:,:] = BLACK
        i = 0
        while count:
            color = GRAY
            if i == cline:
                color = YELLOW
            cv.putText(viewWin, f'{startN+1:2d}: {sorted_files[startN][0]}',(10,20+20*i),cv.FONT_HERSHEY_COMPLEX_SMALL,1,color)
            startN += 1
            count -= 1
            i += 1
        cv.imshow('Out', viewWin)
    #----------------------------------------------------------------------------------------
    c = len(sorted_files)               # 전체 파일(프로젝트 디렉토리 + Downloads 디렉토리) 개수
    if c:
        maxRows = 15                    # 화면에 리스팅되는 최대 파일 개수 설정
        startN = 0
        cline = 0
        if c >= maxRows:
            startN = c - maxRows 
            count = maxRows
            cline = maxRows - 1
        else:
            startN = 0 
            count = c
            cline = c - 1

        while True:
            if c==1:                # 프로젝트 디렉토리 또는 Downloads 디렉토리에 1 개의 모델파일이 
                break               # 존재하면 즉시 실행한다.

            listingFile(sorted_files, startN, count, cline)
            keyBoard = cv.waitKey(0) & 0xFF                       # 1 mSec 시간 지연하고 키값 읽기 
            if keyBoard == 13:                                    # Enter Key
                break
            elif keyBoard == 82:                                  # Arrow Up(82)
                if c > startN+cline:
                    if cline > 0:
                        cline -= 1
                    else:
                        if startN > 0:
                            startN -= 1
            elif keyBoard == 84:                                  # Arrow Down(84)
                if c <= maxRows:                                  #
                    if (cline+1) < c:
                        cline += 1
                else:
                    if (cline+1) < maxRows:
                        cline += 1
                    elif startN+maxRows < c:
                        startN += 1

        modelFile = sorted_files[startN+cline][0]    # 선택한 모델파일
        model = NvidiaModel()
        model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))

        MODEL_FILE = True
        print('\n학습 모델 파일 ', modelFile, ' 을 읽었습니다.\n')
    else:
        print('\n학습 모델 파일 ', modelFile, ' 이 없습니다.\n')
        modelFile = 'Manual Mode'
    # 트랙 윈도우 파일
    t = './'+imageDir+'/_'+imageDir+'_track.pickle'
    if os.path.exists(t):
        with open(t, 'rb') as f:
            d = pickle.load(f)
            WIN_XL = d[0]; WIN_XR = d[1]; WIN_YU = d[2]; WIN_YD = d[3]  # 800 x 480 스크린 모드
            if d[4]: 
                lightMode = 'AUTO'
            elif d[5]: 
                lightCar = 'ON'
        print('WIN_XL:',WIN_XL,'WIN_XR:',WIN_XR, 'WIN_YU:',WIN_YU,'WIN_YD:', WIN_YD)
        print('트랙 윈도우 파일', t, ' 을 읽었습니다.')
    else:
        print('트랙 윈도우 파일', t, ' 이 없습니다.')

    # 신호등 파라메터 읽기
    fn = './'+imageDir+'/_'+imageDir+'_TS.pickle'
    if os.path.exists(fn):
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
    else:
        print('\n신호등 데이터 파일이 없습니다.')
    main()

#############################################################################################

'''
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



'''
