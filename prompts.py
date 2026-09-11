# -*- coding: utf-8 -*-
"""
자연어 장면 검색 아카이브 - 프롬프트 정의 + 개선 실험 기록
==========================================================

필수 요건:
  - 학습 없이 5개 이상의 커스텀 대상을 다룰 것        -> TARGETS_V* (각 6개)
  - 프롬프트를 2회 이상 개선하고 근거를 기록할 것      -> PROMPT_LOG

--prompt-version v1|v2|v3 으로 선택해서 인덱싱한다.
"""

# ---------------------------------------------------------------------------
# 커스텀 대상 (거리/일상 영상 아카이브를 검색한다고 가정)
# 각 항목은 (검색키, Grounding DINO 프롬프트 구절)
# ---------------------------------------------------------------------------

TARGETS_V1 = {
    "사람":     "person",
    "자전거":   "bicycle",
    "자동차":   "car",
    "우산":     "umbrella",
    "강아지":   "dog",
    "가방":     "bag",
}

TARGETS_V2 = {
    # v1 에서 애매했던 대상에 구체 명사 / 상황을 넣음
    "자전거_탄_사람": "a person riding a bicycle",
    "빨간_차":        "a red car",
    "우산_쓴_사람":   "a person holding an umbrella",
    "강아지_산책":    "a person walking a dog",
    "배달_오토바이":  "a delivery motorcycle",
    "유모차":         "a baby stroller",
}

TARGETS_V3 = {
    # v2 에서 오검출 잦던 항목에 색 / 재질 / 위치 수식어 추가, 헷갈리는 대상 분리
    "자전거_탄_사람":   "a person sitting on and riding a bicycle on the road",
    "빨간_승용차":      "a bright red passenger car",
    "빨간_버스":        "a red city bus",          # 빨간 차와 분리 표기
    "우산_쓴_보행자":   "a pedestrian holding an open umbrella overhead",
    "목줄_한_강아지":   "a small dog on a leash held by a person",
    "노란_배달가방_오토바이": "a motorcycle with a yellow delivery box on the back",
}

VERSIONS = {"v1": TARGETS_V1, "v2": TARGETS_V2, "v3": TARGETS_V3}

# ---------------------------------------------------------------------------
# 프롬프트 개선 실험 기록  (슬라이드 8: "어떤 표현을 왜 바꿨는지 실험표로 남길 것")
# ---------------------------------------------------------------------------

PROMPT_LOG = [
    # (버전, 대상, 프롬프트, 관찰된 문제, 다음 버전에서 바꾼 점)
    ("v1", "car", "car",
     "주차된 차/버스/트럭까지 전부 잡혀서 '움직이는 승용차' 장면만 찾기 어려움",
     "v2: 'a red car' 로 색을 지정해 후보를 좁힘"),

    ("v1", "dog", "dog",
     "강아지 인형·간판 그림도 검출됨. 산책 장면을 원했는데 구분 불가",
     "v2: 'a person walking a dog' 로 상황(사람+개)을 묶음"),

    ("v1", "bicycle", "bicycle",
     "거치대에 세워둔 자전거가 대부분. 타고 가는 장면이 안 걸러짐",
     "v2: 'a person riding a bicycle' 로 행위를 명시"),

    ("v2", "a red car", "a red car",
     "빨간 버스를 빨간 차로 오검출 (score 0.4대)",
     "v3: 'a bright red passenger car' + 별도로 'a red city bus' 를 분리 표기"),

    ("v2", "a person holding an umbrella", "a person holding an umbrella",
     "접힌 우산을 든 사람도 잡힘. 비 오는 장면 검색 목적과 안 맞음",
     "v3: 'holding an open umbrella overhead' 로 펼친 상태 + 위치를 못박음"),

    ("v2", "a delivery motorcycle", "a delivery motorcycle",
     "일반 오토바이와 구분이 약함 (score 차이 0.05 이내)",
     "v3: 'a motorcycle with a yellow delivery box on the back' 로 색+부착물 수식어 추가"),
]


def print_log():
    print(f"{'ver':>4} | {'target':<28} | 문제 -> 개선")
    print("-" * 90)
    for ver, tgt, _p, problem, fix in PROMPT_LOG:
        print(f"{ver:>4} | {tgt:<28} | {problem}\n{'':>37}-> {fix}")


if __name__ == "__main__":
    print_log()
