# ToF 센서 기반 차량 추종 자율주행 시스템

딥러닝 기반 차선 인식 자율주행에 **앞 차량 추종**과 **장애물 회피** 기능을 추가한 라즈베리파이 RC카 프로젝트입니다.

---

## 프로젝트 개요

| 항목 | 내용 |
|------|------|
| **하드웨어** | Raspberry Pi + RC카 (차폭 약 11cm) |
| **센서** | VL53L0X ToF 거리 센서, USB 카메라 |
| **딥러닝 모델** | NVIDIA End-to-End CNN (PyTorch) |
| **개발 환경** | Python 3.9, OpenCV 4.5.1, PyTorch 2.0 |

---

## 주요 추가 기능

### 1. 앞 차량 추종 (Car Following)
- ToF 센서로 실시간 거리 측정
- 거리에 따라 자동 속도 조절 (감속 / 정지)

### 2. 장애물 회피 (Obstacle Avoidance)
- **이중 감지 로직**으로 동적/정적 장애물 모두 대응
- 4단계 상태머신 (AVOID → STRAIGHT → RETURN → NORMAL)
- 부드러운 모터 블렌딩으로 자연스러운 회피 동작

### 3. 차선 복귀 (Lane Return)
- 회피 후 CNN 조향각으로 자동 복귀
- 정상 자율주행 모드로 자동 전환

---

## 파일 구성

```
project/
├── ar1.py              # 메인 자율주행 프로그램 (CNN 차선 인식)
├── fa.py              # 앞 차량 추종 + 장애물 회피 모듈
├── gi.py              # 학습 데이터 수집용 수동 조종 프로그램
├── ts.py              # 신호등 인식 모듈
├── NEOPIXEL.py        # LED 제어
└── VL53L0X.py         # ToF 센서 드라이버
```

---

## fa.py 모듈 상세

### 핵심 파라미터

```python
# 거리 기준 (mm)
FOLLOW_SAFE_DIST  = 500   # 정상 속도 (50cm 이상)
FOLLOW_SLOW_DIST  = 280   # 감속 시작 (28cm)
FOLLOW_STOP_DIST  = 110   # 완전 정지 (11cm, 차폭 수준)

# 장애물 판단
OBSTACLE_DELTA       = 80    # 한 프레임에 80mm 이상 급감 시 장애물 의심
AVOID_TRIGGER_DIST   = 350   # 35cm 이내에서 회피 판단

# 회피 동작 (FPS 15 기준)
AVOID_FRAMES   = 35   # 회피 꺾기 ~2.3초
STRAIGHT_MAX   = 1000 # 직진 (거리 조건으로 종료)
RETURN_FRAMES  = 37   # 복귀 꺾기 ~2.5초

# 모터 PWM (0~100)
AVOID_ML  = 5,  AVOID_MR  = 90   # 왼쪽 회피
RETURN_ML = 95, RETURN_MR = 10   # 오른쪽 복귀
```

### 장애물 감지 로직 (이중 조건)

**조건 1: 급감 감지 (동적 장애물)**
```python
delta = 이전거리 - 현재거리
if delta >= 80mm and distance < 350mm:
    → 동적 장애물 의심
```

**조건 2: 정지 타임아웃 (정적 장애물)**
```python
if 280mm 이내에서 8프레임(~0.5초) 이상 머묾:
    → 정적 장애물 판단
```

**오탐 방지:**
- 2프레임 연속 감지되어야 회피 시작 (`_obsCnt >= 2`)
- 회피 후 30프레임(~2초) 쿨다운으로 재감지 차단

### 상태머신

```
[NORMAL] 정상 자율주행 (CNN 조향)
    ↓ 장애물 감지
[AVOID] 강한 꺾기 (좌5/우90, ~2.3초)
    ↓
[STRAIGHT] 직진 통과 (양쪽 55, 최소 1.4초)
    ↓ 거리 안전 확보
[RETURN] 반대 꺾기 (좌95/우10, ~2.5초)
    ↓
[NORMAL] 정상 자율주행 복귀
```

### 부드러운 모터 블렌딩
각 단계의 진입/종료 25% 구간에서 모터값을 선형 보간으로 전환하여 차체 흔들림 최소화

---

## ar1.py 통합 방법

기존 `ar1.py`에 5곳만 추가하면 됩니다.

### 1. 모듈 import
```python
import fa
```

### 2. 초기화
```python
fa.init(baseSpeed=RUN_SPEED, avoidRight=True)
distance = 9999
```

### 3. 거리 전달
```python
if TOF_FLAG:
    distance = tof.get_distance()
    fa.measureDistance(distance)
```

### 4. 회피 중 정지 우회 (핵심)
```python
if distance < STOP_DISTANCE and not fa.isAvoiding():
    signRedDelay = RESTART_NUM
```

### 5. 모터값 계산 교체
```python
# 기존: mL = RUN_SPEED + angle, mR = RUN_SPEED - angle
mL, mR, angle = fa.update(distance, angle, autoRun)
```

### 6. 상태 화면 표시 (선택)
```python
fa.drawStatus(viewWin, distance, mL, mR)
```

---

## 실행 방법

### 1. 모델 학습 (별도 PC에서 수행)
```bash
# gi.py로 학습 데이터 수집 후 PyTorch 모델 학습
python gi.py <프로젝트명>
```

### 2. 자율주행 실행
```bash
sudo python ar1.py <프로젝트명>
```

### 3. 키보드 조작
| 키 | 기능 |
|----|------|
| **G** / Home | 자율주행 시작 |
| **D** | ToF 거리 센서 ON/OFF |
| **C** | 화면 정보 표시 토글 |
| **H** | 경적 |
| **↑↓←→** | 수동 조종 |
| **Space / End** | 정지 |
| **X / ESC** | 프로그램 종료 |

---

## 화면 표시 정보

| 위치 | 표시 내용 |
|------|----------|
| 좌측 패널 | 거리 게이지 바 (초록/노랑/빨강) |
| 좌측 하단 | `DIST: 350mm` 거리값 |
| 좌측 하단 | 상태 라벨 (FOLLOW / AVOID / RETURN) |
| 좌우 모터 | PWM 값 막대 그래프 |
| 차량 LED | 회피 시 깜빡이 자동 점등 |

---

## 튜닝 가이드

| 증상 | 조정 파라미터 |
|------|--------------|
| 회피가 약함 | `AVOID_ML` 낮추기, `AVOID_MR` 높이기 |
| 회피가 너무 김 (한바퀴 돔) | `AVOID_FRAMES` 줄이기 |
| 너무 늦게 회피함 | `AVOID_TRIGGER_DIST` 늘리기 |
| 오감지 많음 | `_obsCnt` 임계값 늘리기 |
| 회피 중 정지함 | `not fa.isAvoiding()` 조건 확인 |
| 복귀가 짧음 | `RETURN_FRAMES` 늘리기 |

---

## 향후 개선 방향

- [ ] 화면 상태 표시 가시성 향상 (폰트 크기 확대)
- [ ] 감속 구간 시각 효과 강화
- [ ] 추월 기능 구현 (옆 차선 상태 감지 추가)
- [ ] 다중 장애물 연속 회피 시나리오 처리
- [ ] 동역학 기반 회피 경로 계획 알고리즘 도입

---

## 참고 문헌

- 반영준 외 (2025). "자율주행 레이싱을 위한 장애물 회피 경로 계획 및 추종 알고리즘". *Transactions of KSAE*, 33(6), 407-417.
- NVIDIA End-to-End Self-Driving Car CNN 아키텍처

---


- **소속:** 숙명여자대학교 인공지능공학과
- **프로젝트:** 사물인터넷 수업 자율주행 차량 프로젝트
