import math
import numpy as np
from typing import Dict

def orthodontic_occlusion_reward(
    overjet_mm: float,
    overbite_mm: float,
    midline_dev_mm: float,
    anterior_contact_ratio: float,
    posterior_contact_ratio: float,
    left_contact_force: float,
    right_contact_force: float,
    working_side_interference: float,
    nonworking_side_interference: float,
    anterior_openbite_fraction: float,
    posterior_crossbite_count: int,
    scissors_bite_count: int,
) -> float:
    """
    교정학적 기준을 이용해 현재 교합 상태를 점수화하는 reward 함수.
    입력 값들은 mesh/landmark에서 이미 계산되어 있다고 가정한다.

    Parameters
    ----------
    overjet_mm : float
        상·하악 전치부 수평 피개 (정상: 대략 2mm, 허용 범위 1~3mm).
    overbite_mm : float
        상·하악 전치부 수직 피개 (정상: 2~3mm, 대략 30~40% 피개).
    midline_dev_mm : float
        상·하악 치열 정중선의 차이 (정상: 0mm, 1mm 이내 허용).
    anterior_contact_ratio : float
        전치부 접촉의 “적절성”을 0~1로 normalize한 값
        (치축 방향 적절한 접촉이 많을수록 1에 가까움).
    posterior_contact_ratio : float
        구치부 접촉의 “적절성”을 0~1로 normalize한 값
        (양측 구치 결절-와동 접촉이 균형있게 있을수록 1에 가까움).
    left_contact_force : float
        좌측 구치부 접촉 힘(합산) – relative scale.
    right_contact_force : float
        우측 구치부 접촉 힘(합산) – relative scale.
    working_side_interference : float
        작업측(intercuspal, 견치/그룹 유도)에서의 과도한 간섭 정도 (0 이상, 클수록 나쁨).
    nonworking_side_interference : float
        비작업측 간섭 정도 (0 이상, 클수록 나쁨) – 교정학/보철학에서 매우 싫어하는 요소.
    anterior_openbite_fraction : float
        전치부 교합면 중 “떠 있는(open bite)” 비율 (0~1, 클수록 나쁨).
    posterior_crossbite_count : int
        교합 시 후방부 교차교합 치아 수 (0 이상, 많을수록 나쁨).
    scissors_bite_count : int
        스키소스 바이트(상악 협측 cusp가 하악 협측 cusp 바깥으로 완전히 빠진 경우) 치아 수.

    Returns
    -------
    float
        RL에서 사용할 scalar reward. 높을수록 좋은 교합.
    """

    # -------------------------------
    # 1. Overjet / Overbite (이상적인 범위에 가까울수록 가산점)
    # -------------------------------
    # 이상값 around 2mm, 허용 범위 1~3mm 근처에 Gaussian 형태로 보상
    def gauss_term(x: float, mu: float, sigma: float) -> float:
        # 0~1 근처 값 (정규분포 모양)
        return math.exp(- ((x - mu) ** 2) / (2 * sigma ** 2))

    # 교정학적으로: overjet 1~3mm가 이상적
    overjet_score = gauss_term(overjet_mm, mu=2.0, sigma=1.0)  # TODO: sigma 튜닝

    # overbite도 2~3mm(약 30~40% 피개)를 이상으로 가정
    overbite_score = gauss_term(overbite_mm, mu=2.5, sigma=1.0)

    # -------------------------------
    # 2. Midline 정중선 (0에 가까울수록 좋음)
    # -------------------------------
    # 0~1로 정규화된 penalty, 0~1mm는 거의 허용, 그 이상은 점점 감소
    midline_penalty = max(0.0, midline_dev_mm - 0.5)  # 0.5mm까지는 거의 허용
    # 보상으로 뒤집기
    midline_score = math.exp(- (midline_penalty ** 2) / (2 * 1.0 ** 2))

    # -------------------------------
    # 3. 전치/구치 접촉의 질 (정상 교합 패턴)
    #    - 구치부: 수직적/축방향 지지
    #    - 전치부: 유도 역할 (anterior guidance)
    # -------------------------------
    # 이미 0~1로 normalize 되어 있다고 가정
    anterior_contact_score = anterior_contact_ratio
    posterior_contact_score = posterior_contact_ratio

    # -------------------------------
    # 4. 좌우 균형 (교합력 좌우 대칭)
    # -------------------------------
    total_force = max(left_contact_force + right_contact_force, 1e-6)
    left_ratio = left_contact_force / total_force
    right_ratio = right_contact_force / total_force
    balance_diff = abs(left_ratio - right_ratio)  # 0이면 완전 대칭

    balance_score = math.exp(- (balance_diff ** 2) / (2 * 0.1 ** 2))  # 0.1 이상 차이나면 급감

    # -------------------------------
    # 5. 간섭 (interference) – 특히 비작업측 간섭은 크게 패널티
    # -------------------------------
    # 작은 값이 좋으므로 negative term
    interference_penalty = (
        1.5 * nonworking_side_interference  # 비작업측 간섭 더 강하게
        + 1.0 * working_side_interference
    )

    # -------------------------------
    # 6. Open bite / Crossbite / Scissors bite 패널티
    # -------------------------------
    openbite_penalty = anterior_openbite_fraction  # 0~1 그대로 사용 (전치부 떠 있으면 나쁨)
    crossbite_penalty = float(posterior_crossbite_count)
    scissors_penalty = float(scissors_bite_count)

    # -------------------------------
    # 7. 각 항목을 weight로 합산
    # -------------------------------
    # 가산 항목 (좋을수록 reward↑)
    positive_term = (
        1.2 * overjet_score +
        1.2 * overbite_score +
        1.0 * midline_score +
        1.0 * anterior_contact_score +
        1.5 * posterior_contact_score +
        1.0 * balance_score
    )

    # 감산 항목 (클수록 reward↓)
    negative_term = (
        2.0 * interference_penalty +   # 간섭은 강하게 깎기
        1.5 * openbite_penalty +
        1.0 * crossbite_penalty +
        1.0 * scissors_penalty
    )

    reward = positive_term - negative_term

    # 수치 폭 조절 (optional): 너무 큰 음수/양수 방지
    # 임시로 tanh로 [-1, 1] 정도로 눌러줄 수도 있음
    reward_clipped = math.tanh(reward)

    return float(reward_clipped)
