# -*- coding: utf-8 -*-
"""
실습 2 - Open Vocabulary Detection (기본 소스코드)
================================================

슬라이드 7 "기본 소스코드가 하는 일" 4단계를 그대로 구현한다.

  1) YOLO-World 실행     : 단어 리스트를 클래스로 지정해 추론
  2) Grounding DINO 실행 : 자연어 문장을 프롬프트로 넣어 추론
  3) 동일 조건 비교       : 같은 이미지 · 같은 대상으로 두 모델 결과를 나란히 저장
  4) 임계값 조정          : 신뢰도(conf) 기준을 바꿔가며 검출 수 변화를 관찰 / CSV 기록

사용 예)
  # 샘플 이미지 자동 다운로드 후 전체 파이프라인 실행
  python ovd_compare.py --download-samples

  # 내가 넣은 이미지로, 원하는 프롬프트로 실행
  python ovd_compare.py --image images/street.jpg \
      --prompts "person" "car" "traffic light" "bicycle" "dog"

  # 임계값 스윕 구간 조정
  python ovd_compare.py --download-samples --sweep 0.05 0.1 0.2 0.3 0.4 0.5

출력)
  results/<이미지이름>_yoloworld.jpg       - YOLO-World 검출 시각화
  results/<이미지이름>_groundingdino.jpg   - Grounding DINO 검출 시각화
  results/<이미지이름>_compare.jpg         - 두 결과 가로로 이어붙인 비교 이미지
  results/threshold_sweep.csv              - 임계값별 검출 수 표 (4단계)
"""

import argparse
import csv
import os
import sys
import urllib.request

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# 0. 공통 유틸
# ----------------------------------------------------------------------------

def imread_kr(path):
    """한글/유니코드 경로에서도 동작하는 cv2.imread"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_kr(path, image):
    """한글/유니코드 경로에서도 동작하는 cv2.imwrite"""
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(path)
    return ok


HERE = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(HERE, "images")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# (색상, 라벨) 을 그려주는 헬퍼 -- 두 모델 시각화에서 공통으로 사용
def draw_detections(image_bgr, detections, title):
    """detections: list of dict(box=[x1,y1,x2,y2], score=float, label=str)"""
    out = image_bgr.copy()
    rng = np.random.default_rng(42)
    color_map = {}
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = det["label"]
        if label not in color_map:
            color_map[label] = tuple(int(c) for c in rng.integers(60, 255, size=3))
        color = color_map[label]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cap = f"{label} {det['score']:.2f}"
        (tw, th), _ = cv2.getTextSize(cap, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(out, cap, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # 상단 제목 바
    bar_h = 34
    banner = np.full((bar_h, out.shape[1], 3), 40, dtype=np.uint8)
    cv2.putText(banner, title, (10, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([banner, out])


SAMPLE_URLS = {
    # 사람/자동차/신호등/자전거 등 일상 객체가 섞인 거리 장면
    "street.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/bus.jpg",
    # 인물 장면
    "zidane.jpg": "https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/assets/zidane.jpg",
}


class _Redirect308(urllib.request.HTTPRedirectHandler):
    # urllib(구버전)은 308 Permanent Redirect 를 안 따라감 -> 307 처럼 처리
    def http_error_308(self, req, fp, code, msg, headers):
        return self.http_error_307(req, fp, code, msg, headers)


def _fetch(url, dst):
    opener = urllib.request.build_opener(_Redirect308)
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]
    with opener.open(url, timeout=60) as r, open(dst, "wb") as f:
        f.write(r.read())


def download_samples():
    for name, url in SAMPLE_URLS.items():
        dst = os.path.join(IMAGES_DIR, name)
        if os.path.exists(dst):
            print(f"[sample] 이미 있음: {dst}")
            continue
        print(f"[sample] 다운로드 {url} -> {dst}")
        _fetch(url, dst)
    return os.path.join(IMAGES_DIR, "street.jpg")


# ----------------------------------------------------------------------------
# 1. YOLO-World  (단어 리스트를 클래스로)
# ----------------------------------------------------------------------------

def run_yolo_world(image_path, class_words, conf=0.05):
    """
    ultralytics 의 YOLO-World 모델.
    class_words 를 그대로 open-vocabulary 클래스로 설정한다.
    conf 는 낮게 두고 (0.05) 검출을 최대한 받아온 뒤,
    임계값 스윕 단계에서 score 로 필터링한다.
    """
    from ultralytics import YOLOWorld

    model = YOLOWorld("yolov8x-worldv2.pt")   # 최초 실행 시 자동 다운로드
    model.set_classes(class_words)

    results = model.predict(imread_kr(image_path), conf=conf, verbose=False)
    r = results[0]
    dets = []
    for b in r.boxes:
        cls_idx = int(b.cls[0])
        dets.append({
            "box": b.xyxy[0].tolist(),
            "score": float(b.conf[0]),
            "label": class_words[cls_idx] if cls_idx < len(class_words) else str(cls_idx),
        })
    return dets


# ----------------------------------------------------------------------------
# 2. Grounding DINO  (자연어 문장을 프롬프트로)
# ----------------------------------------------------------------------------

def _post_process(processor, outputs, input_ids, box_thr, text_thr, hw):
    """transformers 버전에 따라 인자명이 다름 (box_threshold -> threshold)."""
    fn = processor.post_process_grounded_object_detection
    for kwargs in (
        dict(box_threshold=box_thr, text_threshold=text_thr),  # 구버전
        dict(threshold=box_thr, text_threshold=text_thr),       # 신버전
        dict(threshold=box_thr),                                # 더 신버전
    ):
        try:
            return fn(outputs, input_ids, target_sizes=[hw], **kwargs)[0]
        except TypeError:
            continue
    return fn(outputs, input_ids, target_sizes=[hw])[0]


def run_grounding_dino(image_path, text_prompt, box_threshold=0.05, text_threshold=0.05):
    """
    HuggingFace transformers 의 Grounding DINO (IDEA-Research/grounding-dino-tiny).
    text_prompt 는 자연어 문장 형태. 여러 대상은 '.' 으로 구분한다.
      예) "a person. a car. a traffic light. a bicycle. a dog."
    box_threshold 를 낮게 두고 스윕 단계에서 다시 필터링한다.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image_bgr = imread_kr(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    inputs = processor(images=image_rgb, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    h, w = image_bgr.shape[:2]
    results = _post_process(processor, outputs, inputs.input_ids,
                            box_threshold, text_threshold, (h, w))

    labels = results.get("text_labels", results.get("labels"))
    dets = []
    for box, score, label in zip(results["boxes"], results["scores"], labels):
        dets.append({
            "box": [float(v) for v in box.tolist()],
            "score": float(score),
            "label": str(label),
        })
    return dets


# ----------------------------------------------------------------------------
# 3. 동일 조건 비교
# ----------------------------------------------------------------------------

def filter_by_score(dets, thr):
    return [d for d in dets if d["score"] >= thr]


def compare_models(image_path, prompts, view_thr=0.25):
    """같은 이미지 · 같은 대상으로 두 모델을 실행하고 비교 이미지를 저장한다."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    image_bgr = imread_kr(image_path)
    if image_bgr is None:
        sys.exit(f"[에러] 이미지를 읽을 수 없음: {image_path}")

    # 같은 '대상'을 각 모델의 입력 형식에 맞게 변환
    yolo_classes = prompts                                  # 단어 리스트
    dino_prompt = ". ".join(f"a {p}" for p in prompts) + "."  # 자연어 문장

    print("\n[1] YOLO-World 추론 ...")
    yolo_raw = run_yolo_world(image_path, yolo_classes)
    print(f"    raw 검출 {len(yolo_raw)}개")

    print("[2] Grounding DINO 추론 ...")
    try:
        dino_raw = run_grounding_dino(image_path, dino_prompt)
        print(f"    raw 검출 {len(dino_raw)}개")
    except Exception as e:                                  # 모델 미설치/메모리 부족 등
        print(f"    [경고] Grounding DINO 실행 실패 -> 건너뜀: {e}")
        dino_raw = None

    # 시각화 (보기용 임계값 적용)
    yolo_vis = draw_detections(image_bgr, filter_by_score(yolo_raw, view_thr),
                               f"YOLO-World  (conf>={view_thr})")
    y_path = os.path.join(RESULTS_DIR, f"{stem}_yoloworld.jpg")
    imwrite_kr(y_path, yolo_vis)
    print(f"    저장: {y_path}")

    if dino_raw is not None:
        dino_vis = draw_detections(image_bgr, filter_by_score(dino_raw, view_thr),
                                   f"Grounding DINO  (conf>={view_thr})")
        d_path = os.path.join(RESULTS_DIR, f"{stem}_groundingdino.jpg")
        imwrite_kr(d_path, dino_vis)
        print(f"    저장: {d_path}")

        # 두 결과를 같은 높이로 맞춰 가로로 이어붙임
        h = min(yolo_vis.shape[0], dino_vis.shape[0])
        left = cv2.resize(yolo_vis, (int(yolo_vis.shape[1] * h / yolo_vis.shape[0]), h))
        right = cv2.resize(dino_vis, (int(dino_vis.shape[1] * h / dino_vis.shape[0]), h))
        compare = np.hstack([left, np.full((h, 4, 3), 255, np.uint8), right])
        c_path = os.path.join(RESULTS_DIR, f"{stem}_compare.jpg")
        imwrite_kr(c_path, compare)
        print(f"    저장: {c_path}")

    return yolo_raw, dino_raw


# ----------------------------------------------------------------------------
# 4. 임계값 조정 (신뢰도 기준별 검출 수 변화 관찰)
# ----------------------------------------------------------------------------

def threshold_sweep(yolo_raw, dino_raw, thresholds):
    rows = []
    print("\n[4] 임계값 스윕")
    print(f"    {'thr':>6} | {'YOLO-World':>11} | {'GroundingDINO':>13}")
    print("    " + "-" * 36)
    for thr in thresholds:
        n_yolo = len(filter_by_score(yolo_raw, thr))
        n_dino = len(filter_by_score(dino_raw, thr)) if dino_raw is not None else ""
        print(f"    {thr:6.2f} | {n_yolo:>11} | {str(n_dino):>13}")
        rows.append({"threshold": thr, "yolo_world_count": n_yolo,
                     "grounding_dino_count": n_dino})

    csv_path = os.path.join(RESULTS_DIR, "threshold_sweep.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "yolo_world_count",
                                          "grounding_dino_count"])
        w.writeheader()
        w.writerows(rows)
    print(f"    저장: {csv_path}")


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="실습 2 - Open Vocabulary Detection 기본 소스코드")
    ap.add_argument("--image", help="입력 이미지 경로 (미지정 시 --download-samples 필요)")
    ap.add_argument("--download-samples", action="store_true",
                    help="공개 샘플 이미지를 images/ 에 내려받아 사용")
    ap.add_argument("--prompts", nargs="+",
                    default=["person", "bus", "car", "backpack", "traffic light"],
                    help="찾을 대상(open-vocabulary). 공백으로 구분")
    ap.add_argument("--view-thr", type=float, default=0.25,
                    help="시각화에 표시할 최소 신뢰도")
    ap.add_argument("--sweep", nargs="+", type=float,
                    default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50],
                    help="임계값 스윕 목록")
    args = ap.parse_args()

    if args.download_samples:
        sample = download_samples()
        image_path = args.image or sample
    else:
        if not args.image:
            ap.error("--image 를 주거나 --download-samples 를 사용하세요.")
        image_path = args.image

    print("=" * 60)
    print(f"이미지 : {image_path}")
    print(f"대상   : {args.prompts}")
    print("=" * 60)

    yolo_raw, dino_raw = compare_models(image_path, args.prompts, args.view_thr)
    threshold_sweep(yolo_raw, dino_raw, sorted(args.sweep))

    print("\n완료. results/ 폴더를 확인하세요.")


if __name__ == "__main__":
    main()
