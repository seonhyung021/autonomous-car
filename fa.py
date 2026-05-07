#############################################################################################
#
# Follow & Obstacle Avoidance Module  V1.1                       <fa.py>
#
# 앞 차량 추종 (Car Following) + 장애물 감지/회피 + 차선 복귀 모듈
#
# ● 변경 사항 V1.0 → V1.1
#   - AVOID_SPEED_RATIO 0.85 → 0.50 (PWM 최대 100 제한 환경에서 조향 여유 확보)
#   - AVOID_STEER 35 → 45 (더 강한 회피 꺾기)
#   - AVOID_FRAMES 8 → 14 (회피 지속 시간 ~0.9초)
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
OBSTACLE_DELTA    = 100   # 한 프레임 내 거리 감소(mm)가 이 이상이면 장애물 의심
OBSTACLE_CONFIRM  = 3     # 장애물 확정을 위한 연속 감지 프레임 수
AVOID_TRIGGER_DIST = 180  # 급감이 감지될 수 있는 최대 거리 (mm)

# ─── 회피 동작 설정 ──────────────────────────────────────────────────────────────────────────
AVOID_STEER       = 45    # 35 → 45 : 더 강한 회피 꺾기
AVOID_FRAMES      = 14    # 8  → 14 : 회피 지속 ~0.9초
STRAIGHT_MAX      = 45    # 직진 최대 프레임 수 (약 3초 타임아웃)
RETURN_FRAMES     = 12    # 차선 복귀 조향 유지 프레임 수 (약 0.8초)

# ─── 속도 설정 ───────────────────────────────────────────────────────────────────────────────
MIN_SPEED_RATIO   = 0.15  # 서행 시 최솟값 (기본 속도의 15%)
AVOID_SPEED_RATIO = 0.50  # 0.85 → 0.50 : PWM 100 제한 환경에서 조향 여유 확보
                          # eff=35, mL=35+45=80, mR=35-45=-10 → 급격한 꺾기 가능

# ===========================================================================================
# 내부 상태 상수
# ===========================================================================================
_ST_NORMAL   = 0
_ST_SLOW     = 1
_ST_STOP     = 2
_ST_AVOID    = 3
_ST_STRAIGHT = 5
_ST_RETURN   = 4

_ST_LABEL = {
    _ST_NORMAL   : 'FOLLOW: OK',
    _ST_SLOW     : 'FOLLOW: SLOW',
    _ST_STOP     : 'FOLLOW: STOP',
    _ST_AVOID    : 'AVOID >>>',
    _ST_STRAIGHT : '-- STRAIGHT --',
    _ST_RETURN   : '<<< RETURN',
}

# ===========================================================================================
# 모듈 내부 변수
# ===========================================================================================
_state      = _ST_NORMAL
_avoidDir   = +1
_actLeft    = 0
_prevDist   = 9999
_obsCnt     = 0
_baseSpeed  = 70
_latestDist = 9999
_stop_tick  = 0

# ===========================================================================================
# 공개 API
# ===========================================================================================

def init(baseSpeed: int = 70, avoidRight: bool = True):
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
    # STEP 1 : 장애물 감지
    # ------------------------------------------------------------------
    delta = _prevDist - distance
    _prevDist = distance

    # 조건 1: 급감 감지
    is_sudden = (20 < distance < 300) and (delta >= OBSTACLE_DELTA)

    # 조건 2: 정지 타임아웃 (정적 장애물)
    if 20 < distance < FOLLOW_STOP_DIST + 30:
        _stop_tick += 1
    else:
        _stop_tick = 0

    if _state not in (_ST_AVOID, _ST_STRAIGHT, _ST_RETURN):
        if is_sudden or _stop_tick > 10:
            _obsCnt += 1
        else:
            _obsCnt = max(0, _obsCnt - 1)

        if _obsCnt >= 2:
            _obsCnt = 0
            _stop_tick = 0
            _state = _ST_AVOID
            _actLeft = AVOID_FRAMES
            print(f'[fa] !!! 장애물 회피 기동 !!! dist={distance}mm')

    # ------------------------------------------------------------------
    # STEP 2 : 상태 머신
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
        if (distance > FOLLOW_SLOW_DIST) or (distance <= 20) or (_actLeft <= 0):
            _state = _ST_RETURN
            _actLeft = RETURN_FRAMES

    elif _state == _ST_RETURN:
        steerAngle = -AVOID_STEER * _avoidDir
        _actLeft -= 1
        blend = max(1, RETURN_FRAMES // 4)
        if _actLeft < blend:
            t = _actLeft / blend
            steerAngle = int(steerAngle * t + modelAngle * (1.0 - t))
        if _actLeft <= 0:
            _state = _ST_NORMAL
    else:
        steerAngle = modelAngle

    # ------------------------------------------------------------------
    # STEP 3 : 속도 결정
    # ------------------------------------------------------------------
    if _state in (_ST_AVOID, _ST_STRAIGHT, _ST_RETURN):
        speedRatio = AVOID_SPEED_RATIO
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
    return _ST_LABEL.get(_state, '')


def drawStatus(frame, distance: int, mL: int, mR: int):
    WHITE   = (255, 255, 255)
    YELLOW  = (  0, 255, 255)
    RED     = (  0,   0, 255)
    GREEN   = (  0, 255,   0)
    CYAN    = (255, 255,   0)
    MAGENTA = (255,   0, 255)

    label = getStateLabel()

    dist_str = f'DIST:{distance:4d}mm' if distance < 2000 else 'DIST:----mm'
    cv.putText(frame, dist_str, (5, 418), cv.FONT_HERSHEY_PLAIN, 1.1, CYAN)

    if 'STOP' in label:
        col = RED
    elif 'SLOW' in label:
        col = YELLOW
    elif 'AVOID' in label or 'RETURN' in label:
        col = MAGENTA
    else:
        col = GREEN
    cv.putText(frame, label, (5, 438), cv.FONT_HERSHEY_PLAIN, 1.1, col)

    _drawGauge(frame, distance)


# ===========================================================================================
# 내부 헬퍼 함수
# ===========================================================================================

def _reset():
    global _state, _actLeft, _obsCnt
    _state   = _ST_NORMAL
    _actLeft = 0
    _obsCnt  = 0


def _clamp(v: int, lo: int = -100, hi: int = 100) -> int:
    return max(lo, min(hi, int(v)))


def _distToSpeed(distance: int):
    if distance >= FOLLOW_SAFE_DIST:
        return 1.0, _ST_NORMAL
    elif distance >= FOLLOW_SLOW_DIST:
        t = (distance - FOLLOW_SLOW_DIST) / (FOLLOW_SAFE_DIST - FOLLOW_SLOW_DIST)
        ratio = MIN_SPEED_RATIO + (1.0 - MIN_SPEED_RATIO) * t
        return ratio, _ST_SLOW
    elif distance >= FOLLOW_STOP_DIST:
        return MIN_SPEED_RATIO, _ST_SLOW
    else:
        return 0.0, _ST_STOP


def _drawGauge(frame, distance: int):
    GREEN   = (  0, 255,   0)
    YELLOW  = (  0, 255, 255)
    RED     = (  0,   0, 255)
    DARK    = ( 40,  40,  40)
    GRAY    = (128, 128, 128)

    BX, BY1, BY2, BW = 62, 30, 390, 12

    ratio = min(1.0, distance / FOLLOW_SAFE_DIST) if distance < 2000 else 1.0
    fillH = int((BY2 - BY1) * ratio)
    fillY = BY2 - fillH

    col = GREEN if ratio > 0.55 else (YELLOW if ratio > 0.25 else RED)

    cv.rectangle(frame, (BX, BY1), (BX + BW, BY2), DARK, -1)
    cv.rectangle(frame, (BX, fillY), (BX + BW, BY2), col, -1)
    cv.rectangle(frame, (BX, BY1), (BX + BW, BY2), GRAY, 1)
    slowY = BY2 - int((BY2 - BY1) * (FOLLOW_SLOW_DIST / FOLLOW_SAFE_DIST))
    cv.line(frame, (BX - 4, slowY), (BX + BW + 4, slowY), YELLOW, 1)