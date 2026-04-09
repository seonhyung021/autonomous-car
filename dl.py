############################################################################################
#                                                               
# Auto Drive Car
#
# 도로 이미지를 딥러닝하여 학습 데이터 파일을 생성합니다.
# 딥러닝 프레임은 PyTorch 입니다.
#
# Example:
# $ python dp.py myWay           디렉터리: myWay
# 
#
# E: 테스트 이미지 이며 딥러닝에 관여하지 않습니다. 
# V: 평가 이미지 이며 딥러닝에 사용됩니다.
# T: 훈련 이미지 이며 딥러닝에 사용됩니다.
#
# 
#
# FEB 2 2022
#
# SAMPLE Electronics co.                                        
# http://www.ArduinoPLUS.cc                                     
#                                                               
############################################################################################
import cv2 as cv                  # OpenCV 그래픽 라이브러리 version 4.5.1
import os                         # File 일기 쓰기와 관련된 라이브러리
import sys                        # 커맨드 라인 처리를 위한 라이브러리
import time                       # 시간 정보 라이브러리
import glob                       # 검사조건을 통과한 화일 이름 리스트 생성 라이브러리
import pickle                     # 파이썬 개채 저장과 읽기 라이브러리
import random                     # 랜덤(난수) 라이브러리
import datetime                   # 날짜와 시간 라이브러리
import numpy as np                # OpenCV의 이미지 데이터 저장 배열변수 라이브러리
import zipfile as zf              # 압축 파일 생성 라이브러리

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import mean_squared_error, r2_score
import sklearn
#===========================================================================================
RED     =   0,  0,255     # Red
GREEN   =   0,255,  0     # Green
BLUE    = 255,  0,  0     # Blue
MAGENTA = 255,  0,255     # Magenta(Pink)
CYAN    = 255,255,  0     # Cyan(Sky Blue)
YELLOW  =   0,255,255     # Yellow
WHITE   = 255,255,255     # White
GRAY    =  86, 86, 86     # Gray
BLACK   =   0,  0,  0     # Black
RED_DN  =   0,  0,100
BLUE_DN = 100,  0,  0
GRAY_DN =  50, 50, 50

############################################################################################
EPOCHS = 300              #[0] 80 ~ 300
LEARN_RATE = 0.001        #[1] Learn Rate 
VT_RATE = 0.2             #[2] V / T 
AUG_SHIFT = 15            #[3] Augmentation Shift Pixel
AUG_ANGLE = 10            #[4] Augmentation Angle Offset
EVT = 'E'                 #[5] V: V에 E 포함, T: T에 E 포함, E: 딥러닝에 관여하지 않음

YUV_MODE = False          # YUV, BGR
AUG_MODE = False          # True: 오그멘트 디스프레이  False: 표준 디스프레이모드
AUG_ANGLE_TEMP = 10       # 메시지 패널에 표시되고 AUG 실행할 때 AUG_ANGLE 로 전달된다. 

trainLineNum = 0
epCount = 0               # 대시보드에 학습 회수 표시
timeLearn = 0             # 학습에 소요된 시간

# 데이터 불러오기
imageDir = ''             # 트랙 이미지 데이터 디렉토리
dataDir = None            # './'+imageDir
fileList = []             # os.listdir(dataDir)  # 데이터 디렉토리 내의 이미지 파일 이름을 리스트로 반환

SCREEN_WIDTH = 800        # 디스프레이 Width
SCREEN_HEIGHT = 480       # 디스프레이 Height

CELL_H_AUG = 30           # AUG 모드 1 개 Cell 의 높이
CELL_H_STD = 55           # STD 모드 1 개 Cell 의 높이 
ROW_N_AUG = 13            # AUG 모드 Y 방향 Cell 의 개수 
ROW_N_STD = 7             # STD 모드 Y 방향 Cell 의 개수
rowNum = ROW_N_STD        # STD 모드로 부팅

outImg = np.zeros((SCREEN_HEIGHT,SCREEN_WIDTH,3),np.uint8)  # 모드에 관계없이 디스프레이 크기로 고정된다.
msgBox = cv.imread('./_IMAGE/controlBox.png',cv.IMREAD_COLOR)

# 각도를 7 개 그룹으로 분리 --------------------------------------------------------------------
aAL =  -99                # 조향 각 영역 - A Left
aAR =  -47                # 조향 각 영역 - A Right
aBL =  -46                # 조향 각 영역 - B Left
aBR =  -22                # 조향 각 영역 - B Right
aCL =  -21                # 조향 각 영역 - C Left
aCR =  -6                 # 조향 각 영역 - C Right
aDL =  -5                 # 조향 각 영역 - D Left
aDR =  5                  # 조향 각 영역 - D Right
aEL =  6                  # 조향 각 영역 - E Left
aER =  21                 # 조향 각 영역 - E Right
aFL =  22                 # 조향 각 영역 - F Left
aFR =  46                 # 조향 각 영역 - F Right
aGL =  47                 # 조향 각 영역 - G Left
aGR =  99                 # 조향 각 영역 - G Right


file_AE = []; file_AV = []; file_AT = []  # A Example, Valid, Train
file_BE = []; file_BV = []; file_BT = []  # B Example, Valid, Train
file_CE = []; file_CV = []; file_CT = []  # C Example, Valid, Train
file_DE = []; file_DV = []; file_DT = []  # D Example, Valid, Train
file_EE = []; file_EV = []; file_ET = []  # E Example, Valid, Train
file_FE = []; file_FV = []; file_FT = []  # F Example, Valid, Train
file_GE = []; file_GV = []; file_GT = []  # G Example, Valid, Train

x_test_Image = []; y_test_Angle = []      # 확인(Test) 이미지, 각도
x_valid_Image = []; y_valid_Angle = []    # 평가(Valid) 이미지, 각도
x_train_Image = []; y_train_Angle = []    # 훈련(Train) 이미지, 각도

x_test_Image_Aug = []                     # Augmentation 확인(Test) 이미지
x_valid_Image_Aug = []                    # Augmentation 평가(Valid) 이미지
x_train_Image_Aug = []                    # Augmentation 훈련(Train) 이미지

x_test_Image_Result_AUG = []; y_test_Angle_Result_AUG = []
#===========================================================================================
def fileNameExt(v):
    '''
    pass(파일 이름 경로)가 붙어있는 파일 이름 집합을 입력으로 받고 
    pass를 제거한 이름을 리스트 형태로 반환합니다.
    입력한 스트링을 구분기호 "/" 로 분리하여 리스트 데이터로 만들고 
    리스트의 마지막(오른쪽) 데이터(파일 이름)를 리스트 구조로 반환합니다.
    '''
    c = []
    for i in v:
        u = []
        u = i.split('/')         # 패쓰에 포함된 '/' 기호로 분리
        m = u[-1]                # 리스트의 마지막 요소인 이미지 파일 이름
        c.append(m)
    return(c)

#-------------------------------------------------------------------------------------------
def fileSep():
    '''
    이미지 파일을 7 개의 조향 각 그룹으로 분리합니다.
    7 개 그룹은 시험(E), 평가(V), 훈련(T) 이미지로 VT_RATE 비율로 분리합니다.
    이미지를 여러(7개)그룹으로 분리한 이유는 딥러닝시 훈련 데이터셋과 평가 데이터셋을
    분리합니다. 자율주행 자동차의 특성상 급격하게 우회전 하거나 급격하게 좌회전하는
    이미지의 개수는 다른 조향각을 가진 이미지의 개수보다 상대적으로 작습니다.
    단순히 확율적으로만 훈련셋과 평가셋을 분리하게 되면 개수가 적은 급격한 좌회전 이미지와
    급격한 우회전 이미지 셋은 훈련데이터 셋에만 들어가고 평가 데이터 셋에 포함되지 않을
    확율이 생깁니다. 강제로 조향 각도별로 이미지를 나누어 그룹화하고 훈련 데이터셋과 
    평가 데이터 셋을 VT_RATE 비율로 나누어 딥러닝에 들어가기 위한 의도 입니다.  

    '''
    global trainLineNum

    tA=[]; tB=[]; tC=[]; tD=[]; tE=[]; tF=[]; tG=[]

    file_AE.clear(); file_AV.clear(); file_AT.clear()  # A (E, V, T) Clear
    file_BE.clear(); file_BV.clear(); file_BT.clear()  # B (E, V, T) Clear
    file_CE.clear(); file_CV.clear(); file_CT.clear()  # C (E, V, T) Clear
    file_DE.clear(); file_DV.clear(); file_DT.clear()  # D (E, V, T) Clear
    file_EE.clear(); file_EV.clear(); file_ET.clear()  # E (E, V, T) Clear
    file_FE.clear(); file_FV.clear(); file_FT.clear()  # F (E, V, T) Clear
    file_GE.clear(); file_GV.clear(); file_GT.clear()  # G (E, V, T) Clear

    def split(S, T, V, E):

        for i, j in enumerate(S):
            #print(i)
            if i == 0:
                T.append(j)
            elif i == 1:
                V.append(j)
            elif i == 2:
                E.append(j)
            else:
                if i < (int(VT_RATE*len(S))+2):
                    V.append(j)
                else:
                    T.append(j)
    #---------------------------------------------------------------------------------------
    for fN in fileList:                 # 조향각 범위에 따라 7개 구룹으로 분류합니다.

        a = int(fN[-7:-4])              # 이미지 파일 이름의 각도 값을 추출
                                        # P0123__+12.png
        if aAL <= a <= aAR:             # A 그룹
            tA.append(fN)
        if aBL <= a <= aBR:             # B 그룹
            tB.append(fN)
        if aCL <= a <= aCR:             # C 그룹
            tC.append(fN)
        if aDL <= a <= aDR:             # D 그룹
            tD.append(fN)
        if aEL <= a <= aER:             # E 그룹
            tE.append(fN)
        if aFL <= a <= aFR:             # F 그룹
            tF.append(fN)
        if aGL <= a <= aGR:             # G 그룹
            tG.append(fN)

    split(tA, file_AT, file_AV, file_AE) 
    split(tB, file_BT, file_BV, file_BE)
    split(tC, file_CT, file_CV, file_CE)
    split(tD, file_DT, file_DV, file_DE)
    split(tE, file_ET, file_EV, file_EE)
    split(tF, file_FT, file_FV, file_FE)
    split(tG, file_GT, file_GV, file_GE)

    # 최대 개수를 가진 그룹의 파일 개수와 화면의 Row 개수를 더한 크기의 바탕 이미지 생성 

    a = len(file_AE)+len(file_AV)+len(file_AT)
    b = len(file_BE)+len(file_BV)+len(file_BT)
    c = len(file_CE)+len(file_CV)+len(file_CT)
    d = len(file_DE)+len(file_DV)+len(file_DT)
    e = len(file_EE)+len(file_EV)+len(file_ET)
    f = len(file_FE)+len(file_FV)+len(file_FT)
    g = len(file_GE)+len(file_GV)+len(file_GT)

    trainLineNum = max([a, b, c, d, e, f, g])    # 그룹중 최대 개수

def makeAugFileNameL(f):
    '''입력한 파일이름을 Left 오그멘테이션 파일 이름으로 변환합니다. '''
    k = f[:-9]                  # P1234 입력한 파일이름에서 오른쪽 9 개의 문자를 버린다. 
    g = int(f[-7:-4])-AUG_ANGLE # 입력한 파일이름에서 각도값 영역 문자를 정수로 변환하고 오그멘트 각도 감산
    if g<-99: g = -99           # 2 자리를 넘지 않도록 제한 
    v = abs(g)                  # 
    s = '+'
    if g<0: s = '-'
    n = f'{k}_L{s}{v:02d}.png'  # P1234_L-56.png
    return(n)

def makeAugFileNameR(f):
    '''입력한 파일이름을 Right 오그멘테이션 파일 이름으로 변환합니다. '''
    k = f[:-9]                  # P1234 입력한 파일이름에서 오른쪽 9 개의 문자를 버린다. 
    g = int(f[-7:-4])+AUG_ANGLE # 입력한 파일이름에서 각도값 영역 문자를 정수로 변환하고 오그멘트 각도 합산
    if g>99: g = 99             # 2 자리를 넘지 않도록 제한 
    v = abs(g)                  #
    s = '+'
    if g<0: s = '-'
    n = f'{k}_R{s}{v:02d}.png'  # P1234_R+56.png
    return(n)
#-------------------------------------------------------------------------------------------
def outDisplay(IDX, Result = False):
    '''
    수집한 이미지를 7 개의 조향각 그룹으로 나누어 엑셀 형식으로 표시합니다.
    텍스트 이미지(E 그룹)을 상단에 고정 배치 합니다.
    STD 모드에서 100x33 픽셀 이미지를 7 by 6 행열 형태로 배열 합니다.
    AUG 모드에서 50x16 픽셀 이미지를 15 by 11 행열 형태로 배열 합니다.
    '''

    def colDisplay_AUG(x, t, m, n, E, V, T):     
        '''
        x: Cell Start Position 
        t: _, L, R
        m: Angle Position
        n: FileName Position
        E: Test Group
        V: Valitation Group
        T: Train Group
        '''
        y = 20
        h = ROW_N_AUG
        #-----------------------------------------------------------------------------------
        if len(E)>=1:                                # 첫번째 셀은 고정 디스프레이 (IDX와 관련 없다)
            e = E[0]
            c = GRAY_DN
            if t=='_': c = GRAY
            outImg[y+12-10:(y+12+16+2), x-1:(x+50+1)] = c   # 배경색 
            if n == -61:        # D 그룹 에서 _ 파일 일때 위치이며 파일 이름에 각도 같이 표기  D 그룹일때 L , Center 이미지 다 뿌리고 R 처리할 때 Center 파일 이름 쓴다는 뜻임 
                cv.putText(outImg,e[-14:-9]+e[-7:-4],(x+n,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE)
            elif n == -21:      # L 또는 R 그룹 STD 화일 이름 표시 위치
                cv.putText(outImg,e[-14:-9],(x+n,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE)
            if t != '_':
                g = int(e[-7:-4])
                if t=='L': g -= AUG_ANGLE              # AUG 각도 Subtract
                if t=='R': g += AUG_ANGLE              # AUG 각도 Addition
                if g<-99: g = -99
                if g>99: g = 99                      # 2 자리를 넘지 않도록 제한 
                s = '+'
                if g<0: s = '-'

                p = f'{e[:-9]}_{t}{s}{abs(g):02d}.png'
                f = f'./{imageDir}/{p}' 
            else:
                p = e
                f = f'./{imageDir}/{e}' 
            img = cv.imread(f, cv.IMREAD_COLOR)      # 이미지 파일 읽기
            if not YUV_MODE: img = cv.cvtColor(img, cv.COLOR_YUV2BGR) 
            img = cv.resize(img, (50, 16))           # 이미지를 1/4 축소
            outImg[y+12:(y+12+16), x:(x+50)] = img
            if m!=0: cv.putText(outImg, f[-7:-4],(x+m,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE) # 각도 표시

            if Result: 
                x_test_Image_Result_AUG.append(p) 
                y_test_Angle_Result_AUG.append(int(p[-7:-4]))
                return
        if Result: 
            return
        y += CELL_H_AUG
        h -= 1
        #-----------------------------------------------------------------------------------
        i = IDX                                 # 평가 이미지 Row 번호
        while i<len(V) and h:
            e = V[i]
            c = RED_DN
            if t=='_': c = RED
            outImg[y+12-10:(y+12+16+2), x-1:(x+50+1)] = c   # 배경색 
            if n == -61:        # D 그룹 에서 _ 파일 일때 위치이며 파일 이름에 각도 같이 표기  D 그룹일때 L , Center 이미지 다 뿌리고 R 처리할 때 Center 파일 이름 쓴다는 뜻임 
                cv.putText(outImg,e[-14:-9]+e[-7:-4],(x+n,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE)
            elif n == -21:      # L 또는 R 그룹 STD 화일 이름 표시 위치
                cv.putText(outImg,e[-14:-9],(x+n,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE)
            if t != '_':
                g = int(e[-7:-4])
                if t=='L': g -= AUG_ANGLE              # AUG 각도 Subtract
                if t=='R': g += AUG_ANGLE              # AUG 각도 Addition
                if g<-99: g = -99
                if g>99: g = 99                      # 2 자리를 넘지 않도록 제한 
                s = '+'
                if g<0: s = '-'
                f = f'./{imageDir}/{e[:-9]}_{t}{s}{abs(g):02d}.png' 
            else:
                f = f'./{imageDir}/{e}' 
            img = cv.imread(f, cv.IMREAD_COLOR)      # 이미지 파일 읽기
            if not YUV_MODE: img = cv.cvtColor(img, cv.COLOR_YUV2BGR) 
            img = cv.resize(img, (50, 16))           # 이미지를 1/4 축소
            outImg[y+12:(y+12+16), x:(x+50)] = img
            if m!=0: cv.putText(outImg, f[-7:-4],(x+m,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE) # 각도 표시
            y += CELL_H_AUG
            h -= 1
            i += 1
        #-----------------------------------------------------------------------------------
        i -= len(V)                                  # 평가 파일 개수 제거
        while i<len(T) and h:
            e = T[i]                                 # 훈련 파일 이름
            c = BLUE_DN
            if t=='_': c = BLUE
            outImg[y+12-10:(y+12+16+2), x-1:(x+50+1)] = c   # 배경색 
            if n == -61:        # D 그룹 에서 _ 파일 일때 위치이며 파일 이름에 각도 같이 표기  D 그룹일때 L , Center 이미지 다 뿌리고 R 처리할 때 Center 파일 이름 쓴다는 뜻임 
                cv.putText(outImg,e[-14:-9]+e[-7:-4],(x+n,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE)
            elif n == -21:      # L 또는 R 그룹 STD 화일 이름 표시 위치
                cv.putText(outImg,e[-14:-9],(x+n,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE)
            if t != '_':
                g = int(e[-7:-4])
                if t=='L': g -= AUG_ANGLE              # AUG 각도 Subtract
                if t=='R': g += AUG_ANGLE              # AUG 각도 Addition
                if g<-99: g = -99
                if g>99: g = 99                      # 2 자리를 넘지 않도록 제한 
                s = '+'
                if g<0: s = '-'
                f = f'./{imageDir}/{e[:-9]}_{t}{s}{abs(g):02d}.png' 
            else:
                f = f'./{imageDir}/{e}' 
            img = cv.imread(f, cv.IMREAD_COLOR)      # 이미지 파일 읽기
            if not YUV_MODE: img = cv.cvtColor(img, cv.COLOR_YUV2BGR) 
            img = cv.resize(img, (50, 16))           # 이미지를 1/4 축소
            outImg[y+12:(y+12+16), x:(x+50)] = img
            if m!=0: cv.putText(outImg, f[-7:-4],(x+m,y+11),cv.FONT_HERSHEY_PLAIN,0.8,WHITE) # 각도 표시
            y += CELL_H_AUG
            h -= 1
            i += 1
    #---------------------------------------------------------------------------------------
    def colDisplay_STD(p, E, V, T):
        
        h = ROW_N_STD                                # STD 모드 Row 
        i = IDX                                      # 화일 위치 지정
        x = 25+110*p                                 # 그룹에 따른 Cell의 X 시작위치
        y = 20                                       # 첫번째 Cell Y 시작값
        #-----------------------------------------------------------------------------------
        if len(E)>=1:                                # 첫번째 셀은 고정 디스프레이
            e = E[0]                                 # 테스트 파일 이름
            k = './'+imageDir+'/'+ e                 # 경로 + 이름
            img = cv.imread(k, cv.IMREAD_COLOR)
            if not YUV_MODE: img = cv.cvtColor(img, cv.COLOR_YUV2BGR) 
            outImg[y+20-14:(y+20+33+5), x-3:(x+100+3)] = GRAY
            outImg[y+20:(y+20+33), x:(x+100)] = cv.pyrDown(img)
            cv.putText(outImg, e[-14:-4],(x,y+18),cv.FONT_HERSHEY_PLAIN,1,WHITE)
        y+=CELL_H_STD                                # 박스를 포함한 쎌의 높이
        h-=1
        if Result: return
        #-----------------------------------------------------------------------------------
        while i<len(V) and h:
            e = V[i]                                 # 평가 파일 이름
            k = './'+imageDir+'/'+ e                 # 경로 + 이름
            img = cv.imread(k, cv.IMREAD_COLOR)
            if not YUV_MODE: img = cv.cvtColor(img, cv.COLOR_YUV2BGR) 
            outImg[y+20-14:(y+20+33+5), x-3:(x+100+3)] = RED
            outImg[y+20:(y+20+33), x:(x+100)] = cv.pyrDown(img)
            cv.putText(outImg, e[-14:-4],(x,y+18),cv.FONT_HERSHEY_PLAIN,1,WHITE)
            y+=CELL_H_STD                            # 박스를 포함한 쎌의 높이
            h-=1
            i+=1
        #-----------------------------------------------------------------------------------
        i -= len(V)                                  # 평가 파일 수 제거
        while i<len(T) and h:
            e = T[i]                                 # 훈련 파일 이름
            k = './'+imageDir+'/'+ e                 # 경로 + 이름
            img = cv.imread(k, cv.IMREAD_COLOR)
            if not YUV_MODE: img = cv.cvtColor(img, cv.COLOR_YUV2BGR) 
            outImg[y+20-14:(y+20+33+5), x-3:(x+100+3)] = BLUE
            outImg[y+20:(y+20+33), x:(x+100)] = cv.pyrDown(img)
            cv.putText(outImg, e[-14:-4],(x,y+18),cv.FONT_HERSHEY_PLAIN,1,WHITE)
            y+=CELL_H_STD                             # 박스를 포함한 쎌의 높이
            h-=1
            i+=1
    #---------------------------------------------------------------------------------------
    if AUG_MODE:
        '''
        x: Cell Start Position 
        t: _, L, R
        m: Angle Position
        n: FileName Position
        E: Test Group
        V: Valitation Group
        T: Train Group
        '''
        #-----------------------------------------------------------------------------------
        if not Result:                       # Result 모드가 아니면 라인번호 표시
            a = 1                            # IDX 가 0 일때 첫번째 라인 번호는 1
            for i in range(ROW_N_AUG-1):                   
                cv.putText(outImg, str(IDX+a),(1,i*CELL_H_AUG+70),cv.FONT_HERSHEY_PLAIN,0.8,WHITE)
                a += 1
        #-----------------------------------------------------------------------------------
        x = 2; xp = 53
        colDisplay_AUG( x, 'L',  1,   0, file_AE, file_AV, file_AT); x += xp
        colDisplay_AUG( x, '_', 24, -21, file_AE, file_AV, file_AT); x += xp
        colDisplay_AUG( x, 'L',  1,   0, file_BE, file_BV, file_BT); x += xp
        colDisplay_AUG( x, '_', 24, -21, file_BE, file_BV, file_BT); x += xp
        colDisplay_AUG( x, 'L',  1,   0, file_CE, file_CV, file_CT); x += xp
        colDisplay_AUG( x, '_', 24, -21, file_CE, file_CV, file_CT); x += xp

        
        colDisplay_AUG( x, 'L',  1,   0, file_DE, file_DV, file_DT); x += xp
        colDisplay_AUG( x, '_',  0,   0, file_DE, file_DV, file_DT); x += xp
        colDisplay_AUG( x, 'R', 24, -61, file_DE, file_DV, file_DT); x += xp
        
        colDisplay_AUG( x, '_',  1,   0, file_EE, file_EV, file_ET); x += xp
        colDisplay_AUG( x, 'R', 24, -21, file_EE, file_EV, file_ET); x += xp
        colDisplay_AUG( x, '_',  1,   0, file_FE, file_FV, file_FT); x += xp
        colDisplay_AUG( x, 'R', 24, -21, file_FE, file_FV, file_FT); x += xp
        colDisplay_AUG( x, '_',  1,   0, file_GE, file_GV, file_GT); x += xp
        colDisplay_AUG( x, 'R', 24, -21, file_GE, file_GV, file_GT); x += xp
    else:   # STD 모드 
        #-----------------------------------------------------------------------------------
        if not Result:                       # Result 모드가 아니면 라인번호 표시
            a = 1                            # IDX 가 0 일때 첫번째 라인 번호는 1
            for i in range(ROW_N_STD):                   
                cv.putText(outImg, str(IDX+a),(1,i*CELL_H_STD+115),cv.FONT_HERSHEY_PLAIN,1,WHITE)
                a += 1
        #-----------------------------------------------------------------------------------
        colDisplay_STD(0, file_AE, file_AV, file_AT)
        colDisplay_STD(1, file_BE, file_BV, file_BT)
        colDisplay_STD(2, file_CE, file_CV, file_CT)
        colDisplay_STD(3, file_DE, file_DV, file_DT)
        colDisplay_STD(4, file_EE, file_EV, file_ET)
        colDisplay_STD(5, file_FE, file_FV, file_FT)
        colDisplay_STD(6, file_GE, file_GV, file_GT)
#-------------------------------------------------------------------------------------------
def preData():
    global x_train_Image, x_valid_Image, x_test_Image
    global y_train_Angle, y_valid_Angle, y_test_Angle

    # 각 그룹의 훈련용 이미지를 모아 훈련용 이미지 리스트를 만든다.
    x_train_Image.extend(file_AT)
    x_train_Image.extend(file_BT)
    x_train_Image.extend(file_CT)
    x_train_Image.extend(file_DT)
    x_train_Image.extend(file_ET)
    x_train_Image.extend(file_FT)
    x_train_Image.extend(file_GT)

    # 각 그룹의 평가용 이미지를 모아 평가용 이미지 리스트를 만든다.
    x_valid_Image.extend(file_AV)
    x_valid_Image.extend(file_BV)
    x_valid_Image.extend(file_CV)
    x_valid_Image.extend(file_DV)
    x_valid_Image.extend(file_EV)
    x_valid_Image.extend(file_FV)
    x_valid_Image.extend(file_GV)

    # 각 그룹의 시험 이미지를 모아 시험 이미지 리스트를 만든다.
    x_test_Image.extend(file_AE)
    x_test_Image.extend(file_BE)
    x_test_Image.extend(file_CE)
    x_test_Image.extend(file_DE)
    x_test_Image.extend(file_EE)
    x_test_Image.extend(file_FE)
    x_test_Image.extend(file_GE)

    # 시험용 이미지를 훈련용 데이터 리스트에 포함할지 평가용 리스트에 포함할지 그냥 둘건지 처리한다.
    if EVT == 'T':                           # Test 데이터 셋을 훈련 데이터 셋에 포함시킨다.
        x_train_Image.extend(x_test_Image)
    elif EVT == 'V':                         # Test 데이터 셋을 평가 데이터 셋에 포함시킨다.
        x_valid_Image.extend(x_test_Image)
    x_train_Image.extend(x_train_Image_Aug)
    x_valid_Image.extend(x_valid_Image_Aug)
    # 이미지 파일에 라벨링된 각도 값을 추출한다.
    for f in x_train_Image:                  # 훈련용 이미지 파일 이름에서 각도값 추출하여 훈련용 라벨 리스트
        y_train_Angle.append(int(f[-7:-4]))
    for f in x_valid_Image:                  # 평가용 이미지 파일 이름에서 각도값 추출하여 평가용 라벨 리스트
        y_valid_Angle.append(int(f[-7:-4]))
    for f in x_test_Image:                   # 시험용 이미지 파일 이름에서 각도값 추출하여 시험용 라벨 리스트
        y_test_Angle.append(int(f[-7:-4]))
#-------------------------------------------------------------------------------------------
def augment_del():
    '''
    SD 카드에 저장되어 있는 Augmentation 이미지를 전부 삭제(delete)합니다.
    '''
    global x_test_Image_Aug  
    global x_valid_Image_Aug 
    global x_train_Image_Aug  

    x_test_Image_Aug.clear()    # Augmentation 테스트(Test) 이미지 리스트 삭제
    x_valid_Image_Aug.clear()   # Augmentation 평가(Valid) 이미지 리스트 삭제
    x_train_Image_Aug.clear()   # Augmentation 훈련(Train) 이미지 리스트 삭제

    # SD 카드에 저장 되어있는 Augmentation 이미지 파일 *.L* 을 전부 삭제 합니다.
    k = './'+imageDir+'/'+'?????_L???.png'
    k = glob.glob(k)
    for f in k:
        os.remove(f)

    # SD 카드에 저장 되어있는 Augmentation 이미지 파일 *.R* 을 전부 삭제 합니다.
    k = './'+imageDir+'/'+'?????_R???.png' 
    k = glob.glob(k)
    for f in k:
        os.remove(f)
#-------------------------------------------------------------------------------------------
def augmentation():
    ''' Augmentation 이미지 데이터를 생성하고 SD 카드에 저장합니다. '''
    global x_test_Image_Aug, x_valid_Image_Aug, x_train_Image_Aug

    def imgRotateShift(f, ANGLE, SHIFT):
        '''
        NVIDIA 정의된 66x200 의 이미지를 받아 왼쪽 50 픽쎌, 오른쪽 50 픽셀, 위쪽 33 픽쎌, 아래쪽 33 픽쎌을
        추가한 작업용 이미지(132 x 300)를 만들어 중앙에 배치합니다. 그리고 각 4 개의 변 1 픽쎌 라인을 
        각 방향으로 복사하여 4 개의 빈 영역을 채웁니다. 그리고 중심 기준으로 ANGLE 값으로 회전하고  
        SHIFT 값으로 이동시킵니다. 차량이 왼쪽으로 이탈 했을때 오른쪽으로 가도록 각도 보정(Puls ANGLE
        Minus SHIFT)하며 차량이 오른쪽으로 이탈 했을때 왼쪽으로 가도록 보정(Minus ANGLE Plus SHIFT) 합니다.
        ANGLE 값은 0 ~ 15(Degree), SHIFT 값은 0 ~ 20(Pixel) 범위 입니다.

        '''
        workImg = np.zeros((33+66+33,50+200+50,3),np.uint8)  # 작업 영역 이미지 위,아래 33 좌우 50 추가
        img = cv.imread('./'+imageDir+'/'+f,cv.IMREAD_COLOR) # SD 카드로 부터 이미지(66x200)를 읽어온다.
        workImg[33:33+66,50:50+200] = img                    # 66x200 이미지를 132x300 의 중앙으로 복사
        # Y 방향으로 이미지 확장
        for i in range(33):                             # 위, 아래 1개 라인을 33 번 복사하여 채운다.
            workImg[99+i:99+i+1,50:50+200] = workImg[99+i-1:99+i,50:50+200]  # 아래쪽 채우기
            workImg[33-i-1:33-i,50:50+200] = workImg[33-i:33-i+1,50:50+200]  # 윗쪽 채우기
        # X 방향으로 이미지 확장
        for i in range(50):                             # 좌, 우 50 Line 복사하여 채운다.
            workImg[0:132,250+i:250+i+1] = workImg[0:132,250+i-1:250+i]      # 오른쪽 채우기
            workImg[0:132,50-i-1:50-i] = workImg[0:132,50-i:50-i+1]          # 왼쪽 채우기

        a = cv.getRotationMatrix2D((150, 66),ANGLE,1)   # 중심정 기준으로 회전 
        dr = cv.warpAffine(workImg, a,(300, 132))       # (+)ANGLE -> ABCD, (-)ANGLE -> DEFG

        img = dr[33:33+66, 50-SHIFT:50-SHIFT+200]       # SHIFT>0 일때 ABCD, SHIFT<0 일 때 DEFG

        if SHIFT < 0: n = makeAugFileNameL(f)           # (-)*(-) = (+) 이므로 L 라벨(ABCD)
        if SHIFT > 0: n = makeAugFileNameR(f)           # (+)*(-) = (-) 이므로 R 라벨(DEFG)

        m = f'./{imageDir}/{n}' 
        cv.imwrite(m, img)                              # 이미지를 저장
        return(n)                                       # AUG 파일 이름을 반환합니다.
    #---------------------------------------------------------------------------------------
    x_test_Image_Aug.clear();  x_valid_Image_Aug.clear(); x_train_Image_Aug.clear()
    #---------------------------------------------------------------------------------------
    print('왼쪽 조향각 이미지 그룹(A, B, C)과 D 그룹의 Aug 이미지 생성합니다.')  
    tL = []
    tL.extend(file_AE); tL.extend(file_BE); tL.extend(file_CE); tL.extend(file_DE)
    for f in tL: x_test_Image_Aug.append(imgRotateShift(f, AUG_ANGLE, -AUG_SHIFT))
    tL = []
    tL.extend(file_AV); tL.extend(file_BV); tL.extend(file_CV); tL.extend(file_DV)
    for f in tL: x_valid_Image_Aug.append(imgRotateShift(f, AUG_ANGLE, -AUG_SHIFT))
    tL = []
    tL.extend(file_AT); tL.extend(file_BT); tL.extend(file_CT); tL.extend(file_DT)
    for f in tL: x_train_Image_Aug.append(imgRotateShift(f, AUG_ANGLE, -AUG_SHIFT))
    #---------------------------------------------------------------------------------------
    print('D 그룹과 오른쪽 조향각 이미지 그룹(E, F, G)의 Aug 이미지 생성합니다.')  
    tR = []
    tR.extend(file_DE); tR.extend(file_EE); tR.extend(file_FE); tR.extend(file_GE)
    for f in tR: x_test_Image_Aug.append(imgRotateShift(f, -AUG_ANGLE, AUG_SHIFT))
    tR = []
    tR.extend(file_DV); tR.extend(file_EV); tR.extend(file_FV); tR.extend(file_GV)
    for f in tR: x_valid_Image_Aug.append(imgRotateShift(f, -AUG_ANGLE, AUG_SHIFT))
    tR = []
    tR.extend(file_DT); tR.extend(file_ET); tR.extend(file_FT); tR.extend(file_GT)
    for f in tR: x_train_Image_Aug.append(imgRotateShift(f, -AUG_ANGLE, AUG_SHIFT))

    print('AUG_Test:',len(x_test_Image_Aug),'AUG_Valid:',len(x_valid_Image_Aug),'AUG_Train:',len(x_train_Image_Aug))
#-------------------------------------------------------------------------------------------
def augPro():
    global rowNum, AUG_ANGLE

    AUG_ANGLE = AUG_ANGLE_TEMP     # 새 값으로 변경한다.
    rowNum = ROW_N_AUG             # AUG 모드에서  Row 개수
    cv.destroyAllWindows()         # 열려 있는 모든 윈도우를 닫기
    augment_del()                  # SD 카드에 남아있는 AUG 파일 삭제
    augmentation()                 # AUG 파일 생성
#-------------------------------------------------------------------------------------------
def proc():

    global trainLineNum, rowNum, indexLine, AUG_MODE
    global LEARN_RATE, EVT, EPOCHS, VT_RATE, YUV_MODE ,AUG_SHIFT, AUG_ANGLE, AUG_ANGLE_TEMP

    indexLine = 0
    cursorX = 3
    cursorY = 0
    ret = False

    # AUG 모드에서 커서 박스 X 시작, 종점 좌표
    X_START = [0, 106, 212, 318, 424+53, 530+53, 636+53]
    X_END = [0+106, 106+106, 212+106, 424+53, 530+53, 636+53, 689+53+53]

    #---------------------------------------------------------------------------------------
    def te2v(file_XE, file_XV, file_XT): 
        '''
        V 키를 누르면 T 또는 E 가 V 로 이동합니다.
        '''
        if cursorY==0:                         # E 를 V 로 이동한다.
            if len(file_XE)!=0:
                file_XV.append(file_XE.pop(0))
                if AUG_MODE: augPro()
        else:
            i = indexLine+(cursorY-1)-len(file_XV)
            if i>=0 and i<len(file_XT):        # T 를 V 로 이동한다.
                file_XV.append(file_XT.pop(i))
                if AUG_MODE: augPro()
    #---------------------------------------------------------------------------------------
    def ve2t(file_XE, file_XV, file_XT):
        '''
        T 키를 누르면 V 또는 E 가 T 로 이동합니다.
        '''
        if cursorY==0:                         # E 를 T 로 이동한다.
            if len(file_XE)!=0:
                file_XT.append(file_XE.pop(0))
                if AUG_MODE: augPro()
        else:
            i = indexLine+(cursorY-1)
            if i>=0 and i<len(file_XV):        # V 를 T 로 이동한다.
                file_XT.append(file_XV.pop(i))
                if AUG_MODE: augPro()
    #---------------------------------------------------------------------------------------
    def vtse(file_XE, file_XV, file_XT):
        '''
        V 또는 T 를 E 와 교환합니다.
        '''
        if cursorY>0:                                # 커서가 E 에 있으면 아무일도 안한다.
            i = indexLine+(cursorY-1)
            if len(file_XE)!=0:                      # E가 있을 때 교환 한다.
                if i<len(file_XV):                   # V 영역 이므로 E와 V를 교환한다.
                    file_XE.append(file_XV.pop(i))   # V를 제거하고 E에 추가한다.
                    file_XV.insert(i,file_XE.pop(0)) # 첫번째 E를 제거하고 V[i]에 삽입한다.
                    if AUG_MODE: augPro()
                else:
                    i -= len(file_XV)
                    if i<len(file_XT):
                        file_XE.append(file_XT.pop(i))  # V를 제거하고 E에 추가한다.
                        file_XT.insert(i,file_XE.pop(0)) # 첫번째 E를 제거하고 V[i]에 삽입한다.
                        if AUG_MODE: augPro()
            else:                                    # V또는 T를 E로 이동
                if i<len(file_XV):                   # V 영역 이므로 V를 비어있는 E로 이동한다.
                    file_XE.append(file_XV.pop(i))
                    if AUG_MODE: augPro()
                else:                                # T 영역 이므로 T를 비어있는 E로 이동한다.
                    i -= len(file_XV)
                    if i<len(file_XT):
                        file_XE.append(file_XT.pop(i))
                        if AUG_MODE: augPro()
    #---------------------------------------------------------------------------------------
    def delCell(file_XE, file_XV, file_XT):
        '''
        커서 위치의 쎌을 삭제합니다. (SD Card 의 파일은 보존)

        '''
        global x_test_Image_Aug                     # Augmentation 확인(Test) 이미지
        global x_valid_Image_Aug                    # Augmentation 평가(Valid) 이미지
        global x_train_Image_Aug                    # Augmentation 훈련(Train) 이미지

        f = None
        i = indexLine
        if cursorY>0: i=i+cursorY-1

        if cursorY==0:
            if len(file_XE)!=0:                  # E 영역에서 파일이 있으면
                f = file_XE[0]
                file_XE.pop(0)                   # 화일 목록 제거
                print(f'테스트 영역의 {f}를 삭제 하였습니다.')
        else:
            if i<len(file_XV):                   # i 가 validation 영역에 있는지 확인
                f=file_XV[i]                     # 파일 이름 가져온다.
                file_XV.pop(i)                   # 평가(validation) 영역에서 i 파일 제거
                print(f'평가 영역의 {f}를 삭제 하였습니다.')
            else:
                i-=len(file_XV)
                if i<len(file_XT):
                    f=file_XT[i]
                    file_XT.pop(i)
                    print(f'훈련 영역의 {f}를 삭제 하였습니다.')
                else: print('해당 커서 위치에 파일이 없습니다.')

        if f != None:                            # STD 파일이 삭제되었으면 AUG 파일도 삭제한다.
            p = makeAugFileNameL(f); q = makeAugFileNameR(f)
            if p in x_test_Image_Aug: x_test_Image_Aug.remove(p)
            if p in x_valid_Image_Aug: x_valid_Image_Aug.remove(p)
            if p in x_train_Image_Aug: x_train_Image_Aug.remove(p)
            if q in x_test_Image_Aug: x_test_Image_Aug.remove(q)
            if q in x_valid_Image_Aug: x_valid_Image_Aug.remove(q)
            if q in x_train_Image_Aug: x_train_Image_Aug.remove(q)
            if f in fileList: fileList.remove(f)   # fileList 에서도 제거한다.
    #---------------------------------------------------------------------------------------
    while True:

        outImg[:] = BLACK          # 전체 영역을 지운다(검정색으로) * 이미지 객체는 Global 데이터
        outDisplay(indexLine)
        outImg[0:20,:] = GRAY                    # Y:20, X:796 Angle bOX
        t = 'A        B       C          D         E        F        G'
        cv.putText(outImg,t,(47,17),cv.FONT_HERSHEY_COMPLEX_SMALL,1,CYAN)
        t = f'{aAL:3d}   {aAR:3d}   {aBL:3d}   {aBR:3d}  {aCL:3d}  {aCR:3d}      {aDL:3d}   {aDR:3d}      {aEL:3d}   {aER:3d}    {aFL:3d}  {aFR:3d}     {aGL:3d}  {aGR:3d}'
        cv.putText(outImg,t,(10,19),cv.FONT_HERSHEY_PLAIN,1,YELLOW) # 타이틀 조향각 표시

        if AUG_MODE:                             # AUG 모드에서 커서 박스 표시
            xs = X_START[cursorX]; xe = X_END[cursorX]
            cv.rectangle(outImg,(xs,20+CELL_H_AUG*cursorY),(xe,20+CELL_H_AUG*cursorY+30-1),YELLOW,1)
        else:                                    # STD 모드에서 커서 박스 표시
            cv.rectangle(outImg,(25+110*cursorX,41+CELL_H_STD*cursorY),(25+110*cursorX+99,41+CELL_H_STD*cursorY+31),YELLOW,2)

        outImg[429:429+50,0:800] = msgBox               # Y:50, X:796  message bOX

        e = len(file_AE)+len(file_BE)+len(file_CE)+len(file_DE)+len(file_EE)+len(file_FE)+len(file_GE)
        v = len(file_AV)+len(file_BV)+len(file_CV)+len(file_DV)+len(file_EV)+len(file_FV)+len(file_GV)
        t = len(file_AT)+len(file_BT)+len(file_CT)+len(file_DT)+len(file_ET)+len(file_FT)+len(file_GT)

        print('ES:',e,'VS:',v,'TS:',t)           # STD 파일 개수를 출력합니다.

        if AUG_MODE:
            e += len(x_test_Image_Aug)
            v += len(x_valid_Image_Aug)
            t += len(x_train_Image_Aug)
        '''
        print('\nEA List:',len(x_test_Image_Aug),'\n',x_test_Image_Aug)
        print('VA List:',len(x_valid_Image_Aug),'\n',x_valid_Image_Aug)
        print('TA List:',len(x_train_Image_Aug),'\n',x_train_Image_Aug)
        '''
        if EVT == 'E':                                   # E 는 V 또는 T 에 포함되지 않는다.
            p = v; q = t
        elif EVT == 'V':                                 # E 는 V 에 포함 시킨다.
            p = v + e; q = t
        elif EVT == 'T':                                 # E 는 T 에 포함 시킨다.
            p = v; q = t + e

        v = int(100*p/(p+q+1))                           # divide by 0 에러를 피하기 위하여 분모에 1 추가
        s2 = f'{e:2d}        {p:<4d}        {q:<4d}      {v:3d}%     {EVT}      {LEARN_RATE:5.4f}        {EPOCHS}'
        cv.putText(outImg,s2,(70,449),cv.FONT_HERSHEY_COMPLEX_SMALL,0.9,CYAN)
        s3 = f'{AUG_SHIFT:2d}   {AUG_ANGLE_TEMP:2d}'
        cv.putText(outImg,s3,(70,472),cv.FONT_HERSHEY_COMPLEX_SMALL,0.9,CYAN)
        cv.putText(outImg,imageDir,(495,472),cv.FONT_HERSHEY_COMPLEX_SMALL,0.9,MAGENTA)
        #-----------------------------------------------------------------------------------
        cv.namedWindow('Out',cv.WND_PROP_FULLSCREEN)
        cv.setWindowProperty('Out', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
        #-----------------------------------------------------------------------------------
        cv.imshow('Out', outImg)

        #-----------------------------------------------------------------------------------
        keyBoard = cv.waitKey(0)    # 커맨드 키 입력을 기다린다.

        if keyBoard == 0x1B or keyBoard == 0x09 or keyBoard == ord('x') or keyBoard == ord('X'):
            break                   # ESC/TAB/'x' 프로그램 종료

        elif keyBoard == 80:        # Home 
            indexLine = 0

        elif keyBoard == 87:        # End
            indexLine = trainLineNum-rowNum + 2

        elif keyBoard == 86:        # Page Down
            if (indexLine+rowNum) < (trainLineNum-rowNum):
                indexLine = indexLine+rowNum 
            else:
                indexLine = trainLineNum-rowNum + 2

        elif keyBoard == 85:        # Page up
            if (indexLine-rowNum) >= 0:
                indexLine = indexLine-rowNum
            else:
                indexLine = 0

        elif keyBoard == 82:        # Arrow Up
            if cursorY > 0:
                cursorY -= 1
            elif cursorY == 0:
                if indexLine > 0:
                    indexLine -= 1

        elif keyBoard == 84:        # Arrow Down
            if cursorY < (rowNum-1):     # STD 에서 5 AUG 에서 10
                cursorY += 1
            elif indexLine < (trainLineNum-rowNum+1):
                indexLine += 1

        elif keyBoard == 81:        # Arrow Left
            if cursorX > 0:
                cursorX -= 1
            else:
                cursorX = 6       

        elif keyBoard == 83:        # Arrow Right
            if cursorX < 6:
                cursorX += 1
            else:
                cursorX = 0
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('h') or keyBoard == ord('H'): # 학습 회수 설정
            if EPOCHS < 300:
                EPOCHS += 20
            else:
                EPOCHS = 60

        elif keyBoard == ord('l') or keyBoard == ord('L'): # 학습율 설정
            if LEARN_RATE < 0.0025:
                LEARN_RATE += 0.0005
            else:
                LEARN_RATE = 0.0005

        elif keyBoard == ord('c') or keyBoard == ord('C'):         #  E 를 T 또는 V 로 합침
            if EVT == 'E':
                EVT = 'T'
            elif EVT == 'T':
                EVT = 'V'
            elif EVT == 'V':
                EVT = 'E'
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('v') or keyBoard == ord('V'):   # 선택한 T 또는 E 를 V 에 합친다.
            if cursorX == 0: te2v(file_AE, file_AV, file_AT)
            if cursorX == 1: te2v(file_BE, file_BV, file_BT)
            if cursorX == 2: te2v(file_CE, file_CV, file_CT)
            if cursorX == 3: te2v(file_DE, file_DV, file_DT)
            if cursorX == 4: te2v(file_EE, file_EV, file_ET)
            if cursorX == 5: te2v(file_FE, file_FV, file_FT)
            if cursorX == 6: te2v(file_GE, file_GV, file_GT)
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('t') or keyBoard == ord('T'):   # 선택한 V 또는 E 를 T 에 합친다.
            if cursorX == 0: ve2t(file_AE, file_AV, file_AT)
            if cursorX == 1: ve2t(file_BE, file_BV, file_BT)
            if cursorX == 2: ve2t(file_CE, file_CV, file_CT)
            if cursorX == 3: ve2t(file_DE, file_DV, file_DT)
            if cursorX == 4: ve2t(file_EE, file_EV, file_ET)
            if cursorX == 5: ve2t(file_FE, file_FV, file_FT)
            if cursorX == 6: ve2t(file_GE, file_GV, file_GT)
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('e') or keyBoard == ord('E'):   # V 또는 T 를 E 와 교환한다.
            if cursorX == 0: vtse(file_AE, file_AV, file_AT)
            if cursorX == 1: vtse(file_BE, file_BV, file_BT)
            if cursorX == 2: vtse(file_CE, file_CV, file_CT)
            if cursorX == 3: vtse(file_DE, file_DV, file_DT)
            if cursorX == 4: vtse(file_EE, file_EV, file_ET)
            if cursorX == 5: vtse(file_FE, file_FV, file_FT)
            if cursorX == 6: vtse(file_GE, file_GV, file_GT)
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('d') or keyBoard == ord('D'):   # E 또는 V 또는 T 를 지운다
            if cursorX == 0: delCell(file_AE, file_AV, file_AT)
            if cursorX == 1: delCell(file_BE, file_BV, file_BT)
            if cursorX == 2: delCell(file_CE, file_CV, file_CT)
            if cursorX == 3: delCell(file_DE, file_DV, file_DT)
            if cursorX == 4: delCell(file_EE, file_EV, file_ET)
            if cursorX == 5: delCell(file_FE, file_FV, file_FT)
            if cursorX == 6: delCell(file_GE, file_GV, file_GT)
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('s'):                          #  랜덤으로 파일 섞기
            random.shuffle(fileList)
            fileSep()                                       # 그룹별 파일 분류
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('r'):                          # V/T 비율 변경
            VT_RATE += 0.05
            if VT_RATE > 0.3: VT_RATE = 0.1
            print('V/T 비율을 변경 합니다!')
            fileSep()                                       # 그룹별 파일 분류
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('y') or keyBoard == ord('Y'): #  RGB 색상 / YUV 색상 변환
            YUV_MODE = not YUV_MODE
            print('색상 모드를 변경합니다.(RGB <->YUV)')
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('a') or keyBoard == ord('A'): # Augmentation 이미지 생성
            AUG_MODE = not AUG_MODE
            if AUG_MODE: 
                rowNum = ROW_N_AUG   
                cv.destroyAllWindows()         # 열려 있는 모든 윈도우를 닫기
                augment_del()                  # SD 카드에 남아있는 AUG 파일 삭제
                AUG_ANGLE = AUG_ANGLE_TEMP     # 새로 지정한 값을 사용한다.
                augmentation()                 # AUG 파일 생성
            else:
                rowNum = ROW_N_STD
                cursorY = 0
                augment_del()
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('o'):    # AUG 옵셋 각도값은 즉시 표시되나 AUG 실행시 파일 변경 된다.   
            if AUG_ANGLE_TEMP > 1:
                AUG_ANGLE_TEMP -= 1   # AUG 옵셋 각도 값 감소
        elif keyBoard == ord('O'):
            if AUG_ANGLE_TEMP < 15:
                AUG_ANGLE_TEMP += 1   # AUG 옵셋 각도 값 증가
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('b'):    # 오그멘트 쉬프트 이미지 비트 감소
            if AUG_SHIFT > 1:
                AUG_SHIFT -= 1
        elif keyBoard == ord('B'):    # 오그멘트 쉬프트 이미지 비트 증가
            if AUG_SHIFT < 20:        
                AUG_SHIFT += 1
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('p'):    # R Pi Stand Alone Deep Learning
            preData()                 # 데이터 준비
            if len(x_train_Image) and len(x_valid_Image):
                ret = True            # Raspberry Pi 스탠드 얼론 딥러닝 모드로 진입한다.
            else:
                print('Valid(평가)데이터와 Train(훈련)데이터는 최소 1 개 이상이어야 합니다.')
            break
        #-----------------------------------------------------------------------------------
        elif keyBoard == ord('z') or keyBoard == ord('Z'):  # 디렉토리 내의 모든 파일을 압축한다.
            preData()                               # 데이터 준비

            print(x_valid_Image)
            print(x_test_Image)

            p = [EPOCHS, LEARN_RATE]
            n = f'./{imageDir}/_{imageDir}.pickle'  # CoLab 에서 사용
            with open(n, 'wb') as f:
                pickle.dump(x_test_Image, f)        # 시험(Test)
                pickle.dump(x_valid_Image, f)       # 평가(Valid)
                pickle.dump(x_train_Image, f)       # 훈련(Train)
                pickle.dump(p, f)                   # EPOCHS, LEARN_RATE

            u = f'./{imageDir}/*.*'
            t = glob.glob(u)                        # 디렉토리 안의 모든 파일 이름
            v = f'{imageDir}.zip'

            with zf.ZipFile(v, 'w') as z:
                for f in t:
                    z.write(f)
            print(f'\n디렉토리 <{imageDir}> 를 Zip 으로 압축하여 저장 하였습니다.')

            break
    cv.destroyAllWindows()                          # 열려 있는 모든 윈도우를 닫기
    return(ret)
#===========================================================================================
# Pytorch device 설정
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'{device} is available')

# Nvidia CNN 모델 구성 -----------------------------------------------------------------------
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
      x = x.view(x.shape[0], -1)
      x = self.layer2(x)
      x = self.layer3(x)

      return x
#-------------------------------------------------------------------------------------------
def imgPaths(fGroup):
    r = []
    for filename in fGroup:
        r.append(os.path.join(dataDir, filename)) # 파일 경로(path)를 화일 이름 앞에 부착하여 리스트에 추가
    return(r)
# 학습 데이터 생성 ---------------------------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, imageList, angleList):
        self.imageList = imgPaths(imageList)
        self.angleList = angleList

        for i in range(len(imageList)):
            # print(i, self.imageList[i])
            image = cv.imread(self.imageList[i])
            image = image / 255

            self.imageList[i] = image

    def __len__(self):
        return len(self.imageList)

    def __getitem__(self, index):
        images = torch.FloatTensor(self.imageList[index]).permute(2,0,1)
        angles = torch.FloatTensor([self.angleList[index]])

        return images, angles
#-------------------------------------------------------------------------------------------
class EarlyStopping:

    """주어진 patience 이후로 validation loss가 개선되지 않으면 학습을 조기 중지"""
    def __init__(self, patience=7, verbose=False, delta=0, path=None):
        """
        Args:
            patience (int): validation loss가 개선된 후 기다리는 기간
                            Default: 7
            verbose (bool): True일 경우 각 validation loss의 개선 사항 메세지 출력
                            Default: False
            delta (float): 개선되었다고 인정되는 monitered quantity의 최소 변화
                            Default: 0
            path (str): checkpoint저장 경로
                            Default: 'checkpoint.pt'
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'Early Stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''validation loss가 감소하면 모델을 저장한다.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        # torch.save(model, self.path)
        self.val_loss_min = val_loss
# 모델 학습 ---------------------------------------------------------------------------------
def learnProc():

  global timeLearn, epCount

  startLearn = time.time()   #  학습 시작 시각

  model = NvidiaModel().to(device)

  train_dataset = CustomDataset(x_train_Image, y_train_Angle)
  train_loader = DataLoader(train_dataset, batch_size=len(x_train_Image))

  valid_dataset = CustomDataset(x_valid_Image, y_valid_Angle)
  valid_loader = DataLoader(valid_dataset, batch_size=len(x_valid_Image))

  optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
  criterion = nn.MSELoss().to(device)

  early_stopping = EarlyStopping(patience=30, verbose=True, path=f'./{imageDir}/_{imageDir}_model_check.pt')

  train_losses = []
  valid_losses = []
  avg_train_losses = []
  avg_valid_losses = []

  for epoch in range(1, EPOCHS + 1):
      # 학습
      epCount = epoch    # 대시보드에 학습회수 표시

      model.train()

      for batch_idx, (data, target) in enumerate(train_loader):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()

          output = model(data)

          loss = criterion(output.to(torch.float32), target.to(torch.float32))
          loss.backward()
          optimizer.step()

          train_losses.append(loss.item())

      # 평가
      model.eval()

      with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output.to(torch.float32), target.to(torch.float32))
            valid_losses.append(loss.item())

      train_loss = np.average(train_losses)
      valid_loss = np.average(valid_losses)
      avg_train_losses.append(train_loss)
      avg_valid_losses.append(valid_loss)

      epoch_len = len(str(EPOCHS))

      print_msg = (f'[{epoch:>{epoch_len}}/{EPOCHS:>{epoch_len}}] ' +
                      f'train_loss: {train_loss:.6f} ' +
                      f'valid_loss: {valid_loss:.6f}')

      print(print_msg)

      train_losses = []
      valid_losses = []

      early_stopping(valid_loss, model)

      if early_stopping.early_stop:
          print("Early stopping")
          break

  endLearn = time.time()                   # 학습 종료 시각
  timeLearn = int(endLearn - startLearn)   # 학습 소요 시간
  return {'loss':avg_train_losses, 'val_loss':avg_valid_losses}
# 결과 확인 ---------------------------------------------------------------------------------
def resultShow():
    '''
    학습이 종료되면 학습 손실 값과 평가 손실 값 변화 과정이 딕셔너리 데이터 형태로 반환되며
    훈련 손실(train loss) 키는 'loss' 이며 평가 손실(valid loss) 키는 'val_loss'입니다.
    loss 키와 val_loss 키의 데이터는 리스트 데이터 입니다.
    loss 와 val_loss 값의 변화 과정을 그래프로 플롯합니다.

    '''
    model = NvidiaModel()                  # NvidiaModel 의 Summary를 Terminal에 표시합니다.

    outImg[:,:] = cv.imread('./_IMAGE/dashBoard.png',cv.IMREAD_COLOR) #
    outDisplay(0, Result = True)

    # Plot Box: pltXs, pltYs, pltXe, pltYe 힌색 바탕 그래프 박스의 위치값을 지정합니다.
    pltXs = 50; pltYs = 118+75; pltXe = 370; pltYe = 372+75

    lossMax = max(history['loss'])         # 딕셔너리 history의 'loss'키 데이터는 리스트, 최대값
    val_lossMax = max(history['val_loss']) # 딕셔너리 history의 'val_loss'키 데이터는 리스트, 최대값

    if lossMax > val_lossMax: historyMax = lossMax  # 학습 손실과 평가 손실 둘중 큰 것으로 최대 손실값 선택
    else: historyMax = val_lossMax

    yRate = (pltYe-pltYs)/historyMax       # Y 축 플롯 비율 설정

    hislenX = len(history['loss'])         # 딕셔너리 history의 'loss'키 데이터의 개수 
    stepX = int((pltXe-pltXs)/(hislenX-1)) # X 축 플롯 스텝 수
    stepY = int((historyMax)/5)            # Y 축 플롯 스텝 수
    incY = 0

    initX = 0
    initValid = pltYe-int((history['val_loss'][0])*yRate)
    initTrain = pltYe-int((history['loss'][0])*yRate)

    for i in range(4):                     # X 축 라벨 4 개 
        cv.putText(outImg,f'{int(hislenX*(i+1)/4)}',(int((pltXe-pltXs)*(i+1)/4+20),pltYe+19),cv.FONT_HERSHEY_PLAIN,1,WHITE)

    for i in range(5):                     # Y 축 라벨 5 개 
        cv.putText(outImg,f'{i*stepY:4d}',(pltXs-47,pltYe-int(i*(pltYe-pltYs)/5)),cv.FONT_HERSHEY_PLAIN,1,WHITE)

    for i in range(hislenX-1):             # 딕셔너리에 저장된 손실 데이터 그래프
        destValid = pltYe-int((history['val_loss'][i+1])*yRate)     # 평가(Valid) 손실 Y 좌표
        destTrain = pltYe-int((history['loss'][i+1])*yRate)         # 훈련(Train) 손실 Y 좌표
        cv.line(outImg,(initX+pltXs,initValid),((initX+pltXs+stepX),destValid),RED,1,cv.LINE_AA) 
        cv.line(outImg,(initX+pltXs,initTrain),((initX+pltXs+stepX),destTrain),BLUE,1,cv.LINE_AA)
        initValid = destValid; initTrain = destTrain                # 그래프 시작점 갱신
        initX += stepX                                              # X축 갱신
    #---------------------------------------------------------------------------------------
    if len(x_test_Image):          # Test(시험) 데이터가 1 개 이상이면 통계자료 표시한다.

        if AUG_MODE: 
            test_dataset = CustomDataset(x_test_Image_Result_AUG, y_test_Angle_Result_AUG)
            bs = len(x_test_Image_Result_AUG)
        else: 
            test_dataset = CustomDataset(x_test_Image, y_test_Angle)
            bs = len(x_test_Image)

        test_loader = DataLoader(test_dataset, batch_size = bs)
        model = NvidiaModel().to(device)
        model.load_state_dict(torch.load(f'./{imageDir}/_{imageDir}_model_check.pt'))
        model.eval()
  
        y_pred = []
        y_target = []

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                print('입력 조향각', target.to(device).numpy())
                print('예측 조향각', output.to(device).numpy())

                for output_data, target_data in zip(output.to(device).numpy(), target.to(device).numpy()):
                    y_pred.append(output_data)
                    y_target.append(target_data)

        mse = mean_squared_error(y_target, y_pred)     # 표준 편차
        cv.putText(outImg,f'{mse:0.2f}',(390,75+278),cv.FONT_HERSHEY_PLAIN,2,WHITE)
        r2s = r2_score(y_target, y_pred)               # 회귀 결정 계수
        cv.putText(outImg,f'{r2s:.2%}',(390,75+338),cv.FONT_HERSHEY_PLAIN,2,WHITE)

        # 시험 이미지의 예측 각도 표시 ----------------------------------------------------------
        def aCellDsp(x, y):
            nonlocal index

            t = y_pred[index]
            t = str(int(t[0]))
            t = f'{t:>3}'
            cv.putText(outImg, t,(x,y),cv.FONT_HERSHEY_COMPLEX_SMALL,1,CYAN)
            index += 1

        index = 0

        if AUG_MODE:
            if len(file_AE): aCellDsp(5,70); aCellDsp(58,70)
            if len(file_BE): aCellDsp(111,70); aCellDsp(164,70)
            if len(file_CE): aCellDsp(217,70); aCellDsp(270,70)
            if len(file_DE): aCellDsp(323,70); aCellDsp(376,70); aCellDsp(429,70)
            if len(file_EE): aCellDsp(482,70); aCellDsp(535,70)
            if len(file_FE): aCellDsp(588,70); aCellDsp(641,70)
            if len(file_GE): aCellDsp(694,70); aCellDsp(747,70)
        else:
            if len(file_AE): aCellDsp(50,100)
            if len(file_BE): aCellDsp(160,100)
            if len(file_CE): aCellDsp(270,100)
            if len(file_DE): aCellDsp(380,100)
            if len(file_EE): aCellDsp(490,100)
            if len(file_FE): aCellDsp(600,100)
            if len(file_GE): aCellDsp(710,100)
    #---------------------------------------------------------------------------------------
    t = f'{(int(len(x_valid_Image)*100/(len(x_valid_Image)+len(x_train_Image)+1))):3d}%'
    cv.putText(outImg,t,(700,75+180),cv.FONT_HERSHEY_PLAIN,2,WHITE)   # V/T 비율 표시
    t = f'{len(x_valid_Image):<4d}        {len(x_train_Image):<4d}'
    cv.putText(outImg,t,(520,75+147),cv.FONT_HERSHEY_PLAIN,2,WHITE)   # Train, Valid 개수 표시
    cv.putText(outImg,str(epCount),(585,75+180),cv.FONT_HERSHEY_PLAIN,2,WHITE)  # Epoch 수 표시
    cv.putText(outImg,str(LEARN_RATE),(670,75+210),cv.FONT_HERSHEY_PLAIN,2,WHITE) # Learn Rate 표시

    lossMin = min(history['loss'])         # 딕셔너리 history의 'loss'키 데이터는 리스트, 최소값
    t = f'{lossMin:0.2f}'
    cv.putText(outImg,t,(295,223),cv.FONT_HERSHEY_COMPLEX_SMALL,1,BLUE,2)
    val_lossMin = min(history['val_loss']) # 딕셔너리 history의 'val_loss'키 데이터는 리스트, 최소값
    t = f'{val_lossMin:0.2f}'
    cv.putText(outImg,t,(295,247),cv.FONT_HERSHEY_COMPLEX_SMALL,1,RED,2)

    t = f'{int(timeLearn/60):4d}   {int(timeLearn%60):2d}' # 학습 소요 시간 
    cv.putText(outImg,t,(500,75+368),cv.FONT_HERSHEY_PLAIN,2,WHITE)
    cv.putText(outImg,imageDir,(50,180),cv.FONT_HERSHEY_COMPLEX_SMALL,1,GREEN) # 프로젝트 이름
    #---------------------------------------------------------------------------------------
    cv.namedWindow('Out',cv.WND_PROP_FULLSCREEN)
    cv.setWindowProperty('Out', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
    #---------------------------------------------------------------------------------------
    cv.imshow('Out', outImg)
    cv.imwrite(f'./{imageDir}/_{imageDir}_RESULT.png', outImg)
    #---------------------------------------------------------------------------------------
    cv.waitKey(0)                          # 어떤 키든 눌려질 때까지 대기
    cv.destroyAllWindows()                 # 열려 있는 모든 윈도우를 닫기
#==========================================================================================
if __name__ == '__main__':

    print('\n\n')
    print('OpenCV:       ', cv.__version__)
    print('NumPy:        ', np.__version__)
    print('Pytorch:      ', torch.__version__)     # Pytorch version
    print('Scikit-learn: ', sklearn.__version__)
    print('\n\n')

    if len(sys.argv) >= 2:
        imageDir = sys.argv[1]              # Video 저장 디렉토리
    #---------------------------------------------------------------------------------------
        dataDir = './'+imageDir
        pathPar = dataDir+'/_'+imageDir+'_DeepLearn.pickle'
        print('File Path: ', pathPar)

        if os.path.exists(pathPar):
            with open(pathPar, 'rb') as f:
                d = pickle.load(f)              # Pickle 파일이 없으면 다음의 기본값으로 설정된다.
                EPOCHS = d[0]                   # 300    80~300 Max Epoch, PyTorch 에서의 Epoch
                LEARN_RATE = d[1]               # 0.001  Learn Rate 
                VT_RATE = d[2]                  # 0.2    V/T(평가파일 훈련파일 비율) 
                AUG_SHIFT = d[3]                # 40     Augmentation Shift Pixel
                AUG_ANGLE = d[4]                # 20     Augmentation Angle Offset
                EVT = d[5]                      # 'E'    V: E를 V에 포함, T: E를 T에 포함, E: E는 V 또는 T에 포함되지 않음 

                AUG_ANGLE_TEMP = AUG_ANGLE      # Augmentation Angle Offset
            print('딥러닝 파라메터를 읽었습니다:', pathPar)

        if  os.path.exists(dataDir):            # 커맨드 라인에 주어진 디렉터리가 있는지 검사
            augment_del()                       # SD 카드에 남아 있는 AUgmentation 이미지를 삭제합니다.
            fileList = fileNameExt(glob.glob(f'{dataDir}/*__?[0-9][0-9].png')) # 검사 조건에 만족하는 파일 리스트 생성
            random.shuffle(fileList)            # 읽은 파일을 셔플 (deBug 필요시 주석처리)
            fileSep()                           # 그룹별 파일 분류

            if proc():                          # 이미지 데이터 편집

                print('\n')
                print("시험(Test)  데이터:", len(x_test_Image))
                print("평가(Valid) 데이터:", len(x_valid_Image))
                print("학습(Train) 데이터:", len(x_train_Image))
                print('\n')
            
                history = learnProc()                  # 학습 시작
                resultShow()                           # 대시보드에 학습 결과 표시 

            w = [EPOCHS, LEARN_RATE, VT_RATE, AUG_SHIFT, AUG_ANGLE, EVT]
            with open(pathPar, 'wb') as f:
                pickle.dump(w, f)
                print('딥러닝 파라메터를 저장 했습니다:', pathPar)
        else:
            print(f'이미지 저장 디렉터리 {imageDir} 가 없습니다.')
    else:
        print('이미지 저장 디렉터리를 지정 하여야 합니다.')
#==========================================================================================
###########################################################################################



