# PromptScene — 자연어로 영상·사진첩을 검색하는 Open-Vocabulary Detection 실습

고정된 클래스가 아니라 **텍스트 프롬프트로 "찾을 대상"을 그때그때 지정**하는 검출(OVD, Open-Vocabulary Detection)을 실습하고,
이를 응용해 **문장으로 영상/사진첩을 검색해 해당 프레임과 시각(mm:ss)을 찾아주는** 아카이브 도구를 만든다.

| 스크립트 | 역할 |
|---|---|
| `ovd_compare.py` | YOLO-World vs Grounding DINO 를 같은 이미지·같은 대상으로 비교, 임계값 스윕 |
| `scene_search.py` | 프롬프트로 영상/이미지를 인덱싱하고 자연어로 장면 검색 (컨택트시트 + 타임스탬프) |
| `prompts.py` | 커스텀 검색 대상 정의 + 프롬프트 개선 실험 로그 (v1 → v2 → v3) |

사용 모델: [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) (단어 리스트), [Grounding DINO](https://huggingface.co/IDEA-Research/grounding-dino-tiny) (자연어 문장). 둘 다 **추가 학습 없이** 동작한다.

---

## 설치

```bash
conda create -n vision_new python=3.10 -y
conda activate vision_new
pip install -r requirements.txt
pip install "https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"
```

- 모델 가중치(`yolov8x-worldv2.pt`, CLIP `ViT-B-32.pt`)는 **최초 실행 시 자동 다운로드**된다. repo에는 포함하지 않는다.
- Grounding DINO는 GPU 권장. CPU로도 돌지만 느리다. 로컬이 어려우면 Google Colab GPU에서 실행.

---

## 1. 두 OVD 모델 비교 — `ovd_compare.py`

```bash
# 공개 샘플 이미지 자동 다운로드 + 전체 파이프라인
python ovd_compare.py --download-samples

# 내 이미지 + 내 프롬프트
python ovd_compare.py --image images/mine.jpg \
    --prompts "person" "helmet" "safety vest" "gloves" "forklift"

# 임계값 스윕 구간 조정
python ovd_compare.py --download-samples --sweep 0.05 0.1 0.2 0.3 0.4 0.5
```

**출력** (`results/`)

| 파일 | 내용 |
|---|---|
| `<이름>_yoloworld.jpg` | YOLO-World 검출 시각화 |
| `<이름>_groundingdino.jpg` | Grounding DINO 검출 시각화 |
| `<이름>_compare.jpg` | 두 결과를 나란히 이어붙인 비교 이미지 |
| `threshold_sweep.csv` | 신뢰도 임계값별 검출 수 표 |

같은 "대상"을 두 모델의 입력 형식에 맞게 자동 변환한다:
YOLO-World `["person", "car", ...]` ↔ Grounding DINO `"a person. a car. ..."`

---

## 2. 자연어 장면 검색 — `scene_search.py`

```bash
# (선택) 무료 샘플 거리 영상 다운로드
python scene_search.py getvideo

# 1) 영상(또는 images/ 폴더)을 인덱싱 — 1초 간격, v3 프롬프트, Grounding DINO
python scene_search.py index --video sample.mp4 --prompt-version v3 --every 1.0
python scene_search.py index --images images --prompt-version v3

# 2) 인덱싱된 대상 키워드로 검색
python scene_search.py search --query "빨간 버스" --topk 6
python scene_search.py search --query "우산 쓴 보행자"

# 3) 인덱스에 없는 새 문장으로 즉석 검색 (전체 프레임에 프롬프트 재실행)
python scene_search.py search --query "a person carrying a large box" --live

# 프롬프트 개선 실험표 출력
python scene_search.py log
```

**출력**: `results/scene_index.json` (프레임별 검출 인덱스), `results/search_*.jpg`
(검색결과 컨택트시트 — 각 프레임에 `mm:ss` 시각 + score 표시)

---

## 3. 프롬프트 개선 (v1 → v3)

`python scene_search.py log` 로 전체 실험표를 볼 수 있다. 요약:

- **v1** 단순 단어(`car`, `dog`, `bicycle`) → 주차된 차·인형·거치된 자전거까지 잡혀 "장면"이 안 걸러짐
- **v2** 행위/상황 추가(`a person riding a bicycle`, `a person walking a dog`) → 빨간 버스를 빨간 차로 오검출
- **v3** 색·재질·위치 수식어 + 헷갈리는 대상 분리
  (`a bright red passenger car` vs `a red city bus`, `holding an open umbrella overhead`)

| 팁 | 예시 |
|---|---|
| 추상어보다 구체적 명사 | `helmet` → `yellow safety helmet on head` |
| 색·재질·위치 수식어 추가 | `red helmet on a person` |
| 헷갈리는 대상은 분리 표기 | `red passenger car` / `red city bus` |
| 영어 프롬프트가 안정적 | 한국어는 성능 저하 |
| 임계값 튜닝 필수 | `--box-thr`, `--sweep` |

---

## 확장 아이디어

- 검출 결과를 LLM에 넘겨 장면 설명 자동 생성
- Gradio로 검색창 웹 데모
- 여러 프롬프트 결과를 합쳐 "비 오는 출근길" 같은 복합 장면 규칙 추가

---

## 라이선스 / 출처

- 샘플 이미지: [Ultralytics assets](https://github.com/ultralytics/ultralytics)
- 샘플 영상: [Pexels](https://www.pexels.com/) (무료, 저작권 표시 불필요)
- 교육용 실습 프로젝트
