#############################################################################################
#
# Follow & Obstacle Avoidance Module  V2.0                       <fa.py>
#
# 앞 차량 추종 (Car Following) + 장애물 감지/회피 + 차선 복귀 모듈
#
# ● V2.0 (소형 RC카 + 소형 장애물 환경 최적화)
#   - 회피 트리거 거리 확대 (조기 회피로 충돌 방지)
#   - 회피/직진/복귀 시간 단축 (작은 차에 맞춤)
#   - 쿨다운 강화 (재트리거 방지)
#   - 부드러운 진입/종료 블렌딩 유지
#
# ● 차량 스펙 가정
#   - 차폭: 약 15cm (손바닥 크기)
#   - 장애물: 차보다 작음 (10cm 이하)
#   - 트랙폭: 좁음 (S자 곡선 많음)
#   - FPS: 15
#
#############################################################################################

import cv2 as cv
import numpy as np

# ===========================================================================================
# ★ 파라메터
# ===========================================================================================

# ─── 앞 차 추종 거리 (mm) ────────────────────────────────────────────────────────────────────
FOLLOW_SAFE_DIST  = 500
FOLLOW_SLOW_DIST  = 280
FOLLOW_STOP_DIST  = 110

# ─── 장애물 판단 기준 (V2.0: 조기 감지) ──────────────────────────────────────────────────────
OBSTACLE_DELTA       = 80    # 100 → 80 (작은 장애물도 감지)
AVOID_TRIGGER_DIST   = 350   # 250 → 350 (더 멀리서 미리 회피)

# ─── 회피 동작 설정 (V2.0: 소형차 맞춤 - 짧고 빠르게) ─────────────────────────────────────────
AVOID_FRAMES      = 30
STRAIGHT_MAX      = 900 
RETURN_FRAMES     = 30    

# ─── motorRun 직접 지정 (avoidRight=True 기준) ──────────────────────────────────────────────
AVOID_ML    = 10   # 30 → 10 (더 강한 꺾기)
AVOID_MR    = 95   # 80 → 95 (오른쪽 풀파워)
STRAIGHT_ML = 55   # 직진 왼쪽 (보정: 회피로 차체 틀어짐)
STRAIGHT_MR = 55   # 직진 오른쪽
RETURN_ML   = 95   # 80 → 95 (복귀 풀파워)
RETURN_MR   = 10   # 30 → 10 (강한 복귀 꺾기)

# ─── 속도 설정 ───────────────────────────────────────────────────────────────────────────────
MIN_SPEED_RATIO   = 0.15

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
_cooldown   = 0

# ===========================================================================================
# 공개 API
# ===========================================================================================

def init(baseSpeed: int = 70, avoidRight: bool = True):
    global _state, _avoidDir, _actLeft, _prevDist, _obsCnt, _baseSpeed, _latestDist, _stop_tick, _cooldown
    _state      = _ST_NORMAL
    _avoidDir   = +1 if avoidRight else -1
    _actLeft    = 0
    _prevDist   = 9999
    _obsCnt     = 0
    _baseSpeed  = baseSpeed
    _latestDist = 9999
    _stop_tick  = 0
    _cooldown   = 0
    print(f'[fa] V2.0 초기화 완료  baseSpeed={baseSpeed}  avoidDir={"RIGHT" if avoidRight else "LEFT"}')


def measureDistance(distance: int):
    global _latestDist
    _latestDist = distance


def update(distance: int, modelAngle: int, autoRun: bool):
    global _state, _actLeft, _prevDist, _obsCnt, _stop_tick, _cooldown

    if not autoRun:
        _reset()
        _stop_tick = 0
        mL = _baseSpeed + modelAngle
        mR = _baseSpeed - modelAngle
        return _clamp(mL), _clamp(mR), modelAngle

    # ------------------------------------------------------------------
    # STEP 1 : 장애물 감지 (V2.0: 조기 감지)
    # ------------------------------------------------------------------
    delta = _prevDist - distance
    _prevDist = distance

    # 조건 1: 급감 감지 (더 넓은 거리 범위에서)
    is_sudden = (20 < distance < AVOID_TRIGGER_DIST) and (delta >= OBSTACLE_DELTA)

    # 조건 2: 정지 타임아웃 (정적 장애물 - 더 일찍 감지)
    if 20 < distance < 280:   # 200 → 280 (더 멀리서 정적 장애물 감지)
        _stop_tick += 1
    else:
        _stop_tick = 0

    if _state not in (_ST_AVOID, _ST_STRAIGHT, _ST_RETURN):
        if _cooldown > 0:
            _cooldown -= 1
        elif is_sudden or _stop_tick > 8:    # 10 → 8 (더 빨리 정적 장애물 판단)
            _obsCnt += 1
        else:
            _obsCnt = max(0, _obsCnt - 1)

        if _obsCnt >= 2:
            _obsCnt = 0
            _stop_tick = 0
            _cooldown = 30                   # 20 → 30 (재트리거 강화 방지)
            _state = _ST_AVOID
            _actLeft = AVOID_FRAMES
            print(f'[fa] !!! 장애물 회피 기동 !!! dist={distance}mm')

    # ------------------------------------------------------------------
    # STEP 2 : 상태 머신 (부드러운 진입/종료 블렌딩)
    # ------------------------------------------------------------------
    if _state == _ST_AVOID:
        target_mL = AVOID_ML if _avoidDir == +1 else AVOID_MR
        target_mR = AVOID_MR if _avoidDir == +1 else AVOID_ML
        blend = max(1, AVOID_FRAMES // 4)
        progress = AVOID_FRAMES - _actLeft
        if progress < blend:
            t = progress / blend
            mL = int(_baseSpeed * (1.0 - t) + target_mL * t)
            mR = int(_baseSpeed * (1.0 - t) + target_mR * t)
        elif _actLeft < blend:
            t = _actLeft / blend
            mL = int(target_mL * t + STRAIGHT_ML * (1.0 - t))
            mR = int(target_mR * t + STRAIGHT_MR * (1.0 - t))
        else:
            mL = target_mL
            mR = target_mR
        _actLeft -= 1
        if _actLeft <= 0:
            _state = _ST_STRAIGHT
            _actLeft = STRAIGHT_MAX
        return _clamp(mL), _clamp(mR), mL - mR

    elif _state == _ST_STRAIGHT:
        mL = STRAIGHT_ML
        mR = STRAIGHT_MR
        _actLeft -= 1
        # 최소 직진 보장
        elapsed = STRAIGHT_MAX - _actLeft
        if elapsed > 21:
            if (distance > FOLLOW_SAFE_DIST) or (distance <= 20) or (_actLeft <= 0):
                _state = _ST_RETURN
                _actLeft = RETURN_FRAMES
        elif _actLeft <= 0:
            _state = _ST_RETURN
            _actLeft = RETURN_FRAMES
        return _clamp(mL), _clamp(mR), 0

    elif _state == _ST_RETURN:
        target_mL = RETURN_ML if _avoidDir == +1 else RETURN_MR
        target_mR = RETURN_MR if _avoidDir == +1 else RETURN_ML
        blend = max(1, RETURN_FRAMES // 4)
        progress = RETURN_FRAMES - _actLeft
        if progress < blend:
            t = progress / blend
            mL = int(STRAIGHT_ML * (1.0 - t) + target_mL * t)
            mR = int(STRAIGHT_MR * (1.0 - t) + target_mR * t)
        elif _actLeft < blend:
            t = _actLeft / blend
            cnn_mL = _baseSpeed + modelAngle
            cnn_mR = _baseSpeed - modelAngle
            mL = int(target_mL * t + cnn_mL * (1.0 - t))
            mR = int(target_mR * t + cnn_mR * (1.0 - t))
        else:
            mL = target_mL
            mR = target_mR
        _actLeft -= 1
        if _actLeft <= 0:
            _state = _ST_NORMAL
        return _clamp(mL), _clamp(mR), mL - mR

    else:
        # 정상 자율주행
        steerAngle = modelAngle
        speedRatio, newState = _distToSpeed(distance)
        _state = newState
        eff = int(_baseSpeed * speedRatio)
        mL = eff + steerAngle
        mR = eff - steerAngle
        return _clamp(mL), _clamp(mR), steerAngle


def getStateLabel() -> str:
    return _ST_LABEL.get(_state, '')


def isAvoiding() -> bool:
    """현재 회피/직진/복귀 중인지 반환 (ar.py 정지 로직 건너뛰기용)"""
    return _state in (_ST_AVOID, _ST_STRAIGHT, _ST_RETURN)


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
    """V2.0: 회피 추진력 보장"""
    if distance >= FOLLOW_SAFE_DIST:
        return 1.0, _ST_NORMAL
    elif distance >= FOLLOW_SLOW_DIST:
        t = (distance - FOLLOW_SLOW_DIST) / (FOLLOW_SAFE_DIST - FOLLOW_SLOW_DIST)
        ratio = 0.5 + 0.5 * t   # 50~100% 선형
        return ratio, _ST_SLOW
    elif distance >= FOLLOW_STOP_DIST:
        return 0.5, _ST_SLOW
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