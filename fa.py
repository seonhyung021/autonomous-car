#############################################################################################
#
# Follow & Obstacle Avoidance Module  V1.0                       <fa.py>
#
# 앞 차량 추종 (Car Following) + 장애물 감지/회피 + 차선 복귀 모듈
#
# ● 기능 요약
#   1. ToF 거리 센서 값을 이용하여 앞 차와의 거리에 따라 속도를 자동 조절
#   2. 장애물을 감지하면 지정 방향으로 회피 기동 수행
#   3. 회피 후 딥러닝 조향각으로 부드럽게 차선 복귀
#
# ● ar.py 통합 방법  (★ 표시된 4곳에 코드 삽입)
#
#   ★ [1] 파일 상단 import 영역 끝에 추가:
#         import fa
#
#   ★ [2] main() 안 변수 초기화 블록에 추가:
#         fa.init(baseSpeed=RUN_SPEED, avoidRight=True)
#         distance = 9999          # ToF 거리 초기값
#
#   ★ [3] "전방 거리 측정" if TOF_FLAG 블록 안에 추가 (기존 정지 로직 아래):
#         (수정 전)
#             if distance < STOP_DISTANCE:
#                 signRedDelay = RESTART_NUM
#                 BLINK_LEFT = True; BLINK_RIGHT = True
#         (수정 후)
#             if distance < STOP_DISTANCE:
#                 signRedDelay = RESTART_NUM
#                 BLINK_LEFT = True; BLINK_RIGHT = True
#             # ★ fa 모듈로 추종 속도 업데이트
#             fa.measureDistance(distance)
#
#   ★ [4] "자율 주행 모드 조향 각도 검출" 블록의 mL/mR 계산 부분 교체:
#         (수정 전)
#             mL = RUN_SPEED + angle
#             mR = RUN_SPEED - angle
#         (수정 후)
#             mL, mR, angle = fa.update(distance, angle, autoRun)
#
#   ★ [5] (선택) VIEW_FLAG 표시 블록 안에 상태 오버레이 추가:
#         if VIEW_FLAG:
#             fa.drawStatus(viewWin, distance, mL, mR)
#
# SAMPLE Electronics co.
# http://www.ArduinoPLUS.cc
#
# APR 2026
#############################################################################################

import cv2 as cv
import numpy as np

# ===========================================================================================
# ★ 파라메터 – 필요에 따라 자유롭게 조정하세요
# ===========================================================================================

# ─── 앞 차 추종 거리 (mm) ────────────────────────────────────────────────────────────────────
FOLLOW_SAFE_DIST  = 500   # 이 거리 이상이면 기본 속도로 정상 주행
FOLLOW_SLOW_DIST  = 280   # 감속 시작 거리
FOLLOW_STOP_DIST  = 110   # 완전 정지 거리  (ar.py 의 STOP_DISTANCE 와 동일 권장)

# ─── 장애물 판단 기준 ─────────────────────────────────────────────────────────────────────────
# 앞 차(서행 추종)와 장애물(급감)을 구분하는 델타 방식 유지
# 부저(350mm) → 감속(280mm) → 급감 감지 시 회피(180mm 이내에서만) 순서로 동작
OBSTACLE_DELTA    = 100   # 한 프레임 내 거리 감소(mm)가 이 이상이면 장애물 의심
OBSTACLE_CONFIRM  = 3     # 장애물 확정을 위한 연속 감지 프레임 수
AVOID_TRIGGER_DIST = 180  # 급감이 감지될 수 있는 최대 거리 (mm) — 이 거리 이내에서만 회피 판단

# ─── 회피 동작 설정 ──────────────────────────────────────────────────────────────────────────
# 논문 권장: 조향각 25~35 PWM, 회피 0.5~0.7s, 직진 장애물 이탈까지, 복귀 0.7~1.0s
AVOID_STEER       = 35    # 회피 조향각 (논문 권장 25~35, 중간값 적용)
AVOID_FRAMES      = 8     # 회피 조향 유지 프레임 수  (FPS ≈ 15 → 약 0.5초)
STRAIGHT_MAX      = 45    # 직진 최대 프레임 수 (약 3초 타임아웃)
RETURN_FRAMES     = 12    # 차선 복귀 조향 유지 프레임 수 (약 0.8초)

# ─── 속도 설정 ───────────────────────────────────────────────────────────────────────────────
MIN_SPEED_RATIO   = 0.15  # 서행 시 최솟값 (기본 속도의 15%)
AVOID_SPEED_RATIO = 0.85  # 회피·복귀 중 속도 비율 (너무 느리면 기동 불안정)

# ===========================================================================================
# 내부 상태 상수
# ===========================================================================================
_ST_NORMAL   = 0   # 정상 주행 (앞 차 없음 또는 안전 거리 이상)
_ST_SLOW     = 1   # 앞 차 감지 – 감속 중
_ST_STOP     = 2   # 앞 차 감지 – 정지
_ST_AVOID    = 3   # 장애물 회피 중 (1단계: 옆으로 벗어남)
_ST_STRAIGHT = 5   # 장애물 통과 직진 (2단계: 장애물 지나칠 때까지 직진)
_ST_RETURN   = 4   # 차선 복귀 중  (3단계: 반대 방향 조향으로 레인 복귀)

_ST_LABEL = {
    _ST_NORMAL   : 'FOLLOW: OK',
    _ST_SLOW     : 'FOLLOW: SLOW',
    _ST_STOP     : 'FOLLOW: STOP',
    _ST_AVOID    : 'AVOID >>>',
    _ST_STRAIGHT : '-- STRAIGHT --',
    _ST_RETURN   : '<<< RETURN',
}

# ===========================================================================================
# 모듈 내부 변수 (외부에서 직접 변경하지 마세요)
# ===========================================================================================
_state      = _ST_NORMAL
_avoidDir   = +1          # +1: 오른쪽 회피,  -1: 왼쪽 회피
_actLeft    = 0           # 현재 동작(회피/복귀) 잔여 프레임
_prevDist   = 9999        # 이전 프레임 거리 (델타 계산용)
_obsCnt     = 0           # 장애물 연속 감지 카운트
_baseSpeed  = 70          # ar.py 의 RUN_SPEED
_latestDist = 9999        # measureDistance() 로 저장된 최신 거리 값
_stop_tick  = 0

# ===========================================================================================
# 공개 API
# ===========================================================================================

def init(baseSpeed: int = 70, avoidRight: bool = True):
    """
    모듈을 초기화합니다.
    ar.py 의 main() 함수 시작 부분에서 한 번 호출하세요.

    Parameters
    ----------
    baseSpeed  : ar.py 의 RUN_SPEED 값 (기본 주행 속도 PWM)
    avoidRight : True → 오른쪽으로 회피,  False → 왼쪽으로 회피
    """
    global _state, _avoidDir, _actLeft, _prevDist, _obsCnt, _baseSpeed, _latestDist
    _state      = _ST_NORMAL
    _avoidDir   = +1 if avoidRight else -1
    _actLeft    = 0
    _prevDist   = 9999
    _obsCnt     = 0
    _baseSpeed  = baseSpeed
    _latestDist = 9999
    print(f'[fa] 초기화 완료  baseSpeed={baseSpeed}  avoidDir={"RIGHT" if avoidRight else "LEFT"}')


def measureDistance(distance: int):
    """
    ToF 거리 값을 모듈에 저장합니다.
    ar.py 의 TOF_FLAG 블록 안에서 매 프레임 호출하세요.

    Parameters
    ----------
    distance : tof.get_distance() 반환값 (mm)
    """
    global _latestDist
    _latestDist = distance


def update(distance: int, modelAngle: int, autoRun: bool):
    global _state, _actLeft, _prevDist, _obsCnt, _stop_tick

    if not autoRun:
        _reset()
        _stop_tick = 0
        mL = _baseSpeed + modelAngle
        mR = _baseSpeed - modelAngle
        return _clamp(mL), _clamp(mR), modelAngle

    # ------------------------------------------------------------------
    # STEP 1 : 장애물 감지 로직 강화
    # ------------------------------------------------------------------
    delta = _prevDist - distance
    _prevDist = distance

    # 장애물 판단 조건 1: 갑자기 튀어남 (Delta)
    is_sudden = (20 < distance < 300) and (delta >= OBSTACLE_DELTA)
    
    # 장애물 판단 조건 2: 앞에 뭐가 있는데 안 비킴 (Timeout)
    # 정지 거리 근처에서 10프레임(약 0.7초) 이상 멈춰있으면 장애물로 간주
    if 20 < distance < FOLLOW_STOP_DIST + 30:
        _stop_tick += 1
    else:
        _stop_tick = 0

    if _state not in (_ST_AVOID, _ST_STRAIGHT, _ST_RETURN):
        # 갑자기 나타나거나, 너무 오래 서 있거나!
        if is_sudden or _stop_tick > 10:
            _obsCnt += 1
        else:
            _obsCnt = max(0, _obsCnt - 1)

        if _obsCnt >= 2: # 확정 카운트를 2로 낮춰서 반응 속도 향상
            _obsCnt = 0
            _stop_tick = 0
            _state = _ST_AVOID
            _actLeft = AVOID_FRAMES
            print(f'[fa] !!! 장애물 회피 기동 !!! dist={distance}mm')

    # ------------------------------------------------------------------
    # STEP 2 : 상태 머신 (기존과 동일하되 블렌딩 유지)
    # ------------------------------------------------------------------
    if _state == _ST_AVOID:
        steerAngle = AVOID_STEER * _avoidDir
        _actLeft -= 1
        if _actLeft <= 0:
            _state = _ST_STRAIGHT
            _actLeft = STRAIGHT_MAX

    elif _state == _ST_STRAIGHT:
        steerAngle = modelAngle
        _actLeft -= 1
        # 장애물이 시야에서 사라지면 즉시 복귀
        if (distance > FOLLOW_SLOW_DIST) or (distance <= 20) or (_actLeft <= 0):
            _state = _ST_RETURN
            _actLeft = RETURN_FRAMES

    elif _state == _ST_RETURN:
        steerAngle = -AVOID_STEER * _avoidDir
        _actLeft -= 1
        # 복귀 끝부분에서 딥러닝 조향과 섞기
        blend = max(1, RETURN_FRAMES // 4)
        if _actLeft < blend:
            t = _actLeft / blend
            steerAngle = int(steerAngle * t + modelAngle * (1.0 - t))
        if _actLeft <= 0:
            _state = _ST_NORMAL
    else:
        steerAngle = modelAngle

    # ------------------------------------------------------------------
    # STEP 3 : 속도 결정 (회피 중에는 멈추지 않게 우선순위 조정)
    # ------------------------------------------------------------------
    if _state in (_ST_AVOID, _ST_STRAIGHT, _ST_RETURN):
        speedRatio = AVOID_SPEED_RATIO
        # 회피 도중 정말 충돌 직전(50mm)이 아니면 멈추지 않고 진행
        if 20 < distance < 50:
            speedRatio = 0.0
    else:
        speedRatio, newState = _distToSpeed(distance)
        _state = newState

    eff = int(_baseSpeed * speedRatio)
    mL = eff + steerAngle
    mR = eff - steerAngle

    return _clamp(mL), _clamp(mR), steerAngle


def getStateLabel() -> str:
    """현재 상태 문자열을 반환합니다 (화면 표시용)."""
    return _ST_LABEL.get(_state, '')


def drawStatus(frame, distance: int, mL: int, mR: int):
    """
    ar.py 의 viewWin 에 추종/회피 상태를 오버레이합니다.
    VIEW_FLAG == True 일 때만 호출하세요.

    Parameters
    ----------
    frame    : viewWin (ar.py 의 800×480 표시 버퍼)
    distance : ToF 측정 거리 (mm)
    mL, mR   : 최종 모터 PWM 값
    """
    WHITE   = (255, 255, 255)
    YELLOW  = (  0, 255, 255)
    RED     = (  0,   0, 255)
    GREEN   = (  0, 255,   0)
    CYAN    = (255, 255,   0)
    MAGENTA = (255,   0, 255)

    label = getStateLabel()

    # ── 거리 텍스트 ──────────────────────────────────────────────────────
    dist_str = f'DIST:{distance:4d}mm' if distance < 2000 else 'DIST:----mm'
    cv.putText(frame, dist_str, (5, 418), cv.FONT_HERSHEY_PLAIN, 1.1, CYAN)

    # ── 상태 텍스트 ──────────────────────────────────────────────────────
    if 'STOP' in label:
        col = RED
    elif 'SLOW' in label:
        col = YELLOW
    elif 'AVOID' in label or 'RETURN' in label:
        col = MAGENTA
    else:
        col = GREEN
    cv.putText(frame, label, (5, 438), cv.FONT_HERSHEY_PLAIN, 1.1, col)

    # ── 세로 거리 게이지 바 (왼쪽 패널) ──────────────────────────────────
    _drawGauge(frame, distance)


# ===========================================================================================
# 내부 헬퍼 함수
# ===========================================================================================

def _reset():
    global _state, _actLeft, _obsCnt
    _state  = _ST_NORMAL
    _actLeft = 0
    _obsCnt  = 0


def _clamp(v: int, lo: int = -100, hi: int = 100) -> int:
    return max(lo, min(hi, int(v)))


def _distToSpeed(distance: int):
    """
    거리 → (속도 비율, 새 상태) 반환
    """
    if distance >= FOLLOW_SAFE_DIST:
        return 1.0, _ST_NORMAL
    elif distance >= FOLLOW_SLOW_DIST:
        # SLOW_DIST ~ SAFE_DIST 구간: 선형 감속
        t = (distance - FOLLOW_SLOW_DIST) / (FOLLOW_SAFE_DIST - FOLLOW_SLOW_DIST)
        ratio = MIN_SPEED_RATIO + (1.0 - MIN_SPEED_RATIO) * t
        return ratio, _ST_SLOW
    elif distance >= FOLLOW_STOP_DIST:
        return MIN_SPEED_RATIO, _ST_SLOW
    else:
        return 0.0, _ST_STOP


def _drawGauge(frame, distance: int):
    """왼쪽 패널에 세로 게이지 바로 앞 차 거리를 시각화."""
    GREEN   = (  0, 255,   0)
    YELLOW  = (  0, 255, 255)
    RED     = (  0,   0, 255)
    DARK    = ( 40,  40,  40)
    GRAY    = (128, 128, 128)

    BX, BY1, BY2, BW = 62, 30, 390, 12   # 게이지 바 위치·크기

    ratio = min(1.0, distance / FOLLOW_SAFE_DIST) if distance < 2000 else 1.0
    fillH = int((BY2 - BY1) * ratio)
    fillY = BY2 - fillH

    col = GREEN if ratio > 0.55 else (YELLOW if ratio > 0.25 else RED)

    # 배경
    cv.rectangle(frame, (BX, BY1), (BX + BW, BY2), DARK, -1)
    # 채움
    cv.rectangle(frame, (BX, fillY), (BX + BW, BY2), col, -1)
    # 테두리
    cv.rectangle(frame, (BX, BY1), (BX + BW, BY2), GRAY, 1)
    # SAFE 거리 기준선
    safeY = BY2 - int((BY2 - BY1) * 1.0)
    slowY = BY2 - int((BY2 - BY1) * (FOLLOW_SLOW_DIST / FOLLOW_SAFE_DIST))
    cv.line(frame, (BX - 4, slowY), (BX + BW + 4, slowY), YELLOW, 1)
