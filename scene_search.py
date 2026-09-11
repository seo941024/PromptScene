# -*- coding: utf-8 -*-
"""
개인 프로젝트 - 자연어 장면 검색 아카이브
=========================================

슬라이드 9 추천 주제 1번.
  "문장으로 사진첩·영상을 검색해 해당 프레임과 시각을 찾아줌"
  핵심 : 프롬프트 -> 프레임 인덱싱

필수 요건 충족 방식
  1) 검출 대상을 텍스트 프롬프트로 지정      -> Grounding DINO / YOLO-World (OVD)
  2) 학습 없이 커스텀 대상 5개 이상          -> prompts.py 의 TARGETS_V* (6개)
  3) 프롬프트 2회 이상 개선 + 근거 기록      -> prompts.py 의 PROMPT_LOG (v1->v2->v3)
  4) YOLO-World 또는 Grounding DINO 포함    -> --engine 으로 선택 (기본 dino)
  결과가 눈에 보이는 데모                    -> results/search_*.jpg 컨택트시트 + 시각(타임스탬프)

--------------------------------------------------------------------------
사용법
--------------------------------------------------------------------------
# 1) 영상(또는 images/ 폴더)을 인덱싱  (1초 간격 샘플링, v3 프롬프트)
python scene_search.py index --video sample.mp4 --prompt-version v3 --every 1.0
python scene_search.py index --images images --prompt-version v3

# 2) 인덱싱된 대상 키워드로 검색
python scene_search.py search --query "빨간 버스" --topk 6
python scene_search.py search --query "우산 쓴 보행자"

# 3) 인덱스에 없는 새 문장으로 즉석 검색 (전체 프레임에 프롬프트 재실행)
python scene_search.py search --query "a person carrying a large box" --live

# 프롬프트 개선 실험표 출력
python scene_search.py log
"""

import argparse
import glob
import json
import os

import cv2
import numpy as np

from prompts import VERSIONS, print_log


def imread_kr(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_kr(path, image):
    ext = os.path.splitext(path)[1] or ".jpg"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(path)
    return ok

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
INDEX_PATH = os.path.join(RESULTS_DIR, "scene_index.json")


# ===========================================================================
# 검출 엔진
# ===========================================================================

class DinoEngine:
    """Grounding DINO - 자연어 문장 프롬프트. 모델을 1회만 로드해 재사용."""

    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        mid = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(mid).to(self.device)
        print(f"[engine] Grounding DINO 로드 완료 (device={self.device})")

    def detect(self, image_bgr, phrases, box_threshold=0.30, text_threshold=0.25):
        """phrases: 프롬프트 구절 리스트. 반환: [(phrase, score, [x1,y1,x2,y2]), ...]"""
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        text = ". ".join(p.strip(". ") for p in phrases) + "."
        inputs = self.processor(images=rgb, text=text, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        h, w = image_bgr.shape[:2]
        fn = self.processor.post_process_grounded_object_detection
        res = None
        for kw in (dict(box_threshold=box_threshold, text_threshold=text_threshold),
                   dict(threshold=box_threshold, text_threshold=text_threshold),
                   dict(threshold=box_threshold)):
            try:
                res = fn(outputs, inputs.input_ids, target_sizes=[(h, w)], **kw)[0]
                break
            except TypeError:
                continue
        if res is None:
            res = fn(outputs, inputs.input_ids, target_sizes=[(h, w)])[0]
        labels = res.get("text_labels", res.get("labels"))
        out = []
        for box, score, label in zip(res["boxes"], res["scores"], labels):
            out.append((str(label), float(score), [float(v) for v in box.tolist()]))
        return out


class YoloWorldEngine:
    """YOLO-World - 단어/짧은 명사구 리스트."""

    def __init__(self):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld("yolov8x-worldv2.pt")
        print("[engine] YOLO-World 로드 완료")

    def detect(self, image_bgr, phrases, box_threshold=0.05, text_threshold=None):
        self.model.set_classes(list(phrases))
        r = self.model.predict(image_bgr, conf=box_threshold, verbose=False)[0]
        out = []
        for b in r.boxes:
            ci = int(b.cls[0])
            name = phrases[ci] if ci < len(phrases) else str(ci)
            out.append((name, float(b.conf[0]), [float(v) for v in b.xyxy[0].tolist()]))
        return out


def build_engine(name):
    return YoloWorldEngine() if name == "yoloworld" else DinoEngine()


# ===========================================================================
# 프레임 소스
# ===========================================================================

def iter_frames(video=None, images=None, every=1.0):
    """(frame_id, time_sec, image_bgr) 를 순서대로 yield."""
    if video:
        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            raise SystemExit(f"[에러] 영상을 열 수 없음: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * every)))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, idx / fps, frame
            idx += 1
        cap.release()
    elif images:
        files = sorted(glob.glob(os.path.join(images, "*.jpg")) +
                       glob.glob(os.path.join(images, "*.png")) +
                       glob.glob(os.path.join(images, "*.jpeg")))
        for i, f in enumerate(files):
            img = imread_kr(f)
            if img is not None:
                yield i, float(i), img   # 사진첩은 '시각' 대신 순번을 초로 기록
    else:
        raise SystemExit("--video 또는 --images 중 하나가 필요합니다.")


# ===========================================================================
# index
# ===========================================================================

def cmd_index(args):
    targets = VERSIONS[args.prompt_version]          # {검색키: 프롬프트구절}
    keys = list(targets.keys())
    phrases = list(targets.values())
    engine = build_engine(args.engine)

    src = args.video or args.images
    entries = []
    n_frames = 0
    for fid, tsec, frame in iter_frames(args.video, args.images, args.every):
        n_frames += 1
        dets = engine.detect(frame, phrases, box_threshold=args.box_thr)
        # 검출된 프롬프트 구절 -> 검색키로 되돌림 (부분일치)
        for phrase, score, box in dets:
            key = _phrase_to_key(phrase, targets) or phrase
            entries.append({
                "frame_id": fid, "time_sec": round(tsec, 2),
                "key": key, "phrase": phrase,
                "score": round(score, 3), "box": [round(v, 1) for v in box],
            })
        if n_frames % 10 == 0:
            print(f"  ... {n_frames} 프레임 처리, 누적 검출 {len(entries)}")

    index = {
        "source": src, "engine": args.engine,
        "prompt_version": args.prompt_version, "every": args.every,
        "targets": targets, "keys": keys,
        "n_frames": n_frames, "entries": entries,
    }
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    print(f"\n[index] {n_frames} 프레임 / 검출 {len(entries)}건 -> {INDEX_PATH}")
    _print_key_stats(index)


def _phrase_to_key(phrase, targets):
    p = phrase.lower()
    for key, prompt in targets.items():
        if p in prompt.lower() or prompt.lower() in p:
            return key
    return None


def _print_key_stats(index):
    from collections import Counter
    c = Counter(e["key"] for e in index["entries"])
    print("[index] 대상별 검출 프레임 수")
    for k in index["keys"]:
        print(f"    {k:<24} {c.get(k, 0)}")


# ===========================================================================
# search
# ===========================================================================

def cmd_search(args):
    if not os.path.exists(INDEX_PATH):
        raise SystemExit("먼저 'index' 를 실행하세요.")
    index = json.load(open(INDEX_PATH, encoding="utf-8"))

    if args.live:
        hits = _search_live(index, args.query, args.box_thr)
    else:
        hits = _search_index(index, args.query)
    hits = [h for h in hits if h["score"] >= args.min_score]

    if not hits:
        print(f"'{args.query}' 에 해당하는 장면을 찾지 못했습니다. (--live 로 재검색 가능)")
        return

    hits.sort(key=lambda h: h["score"], reverse=True)
    hits = hits[:args.topk]

    print(f"\n[검색결과] '{args.query}'  (상위 {len(hits)}개)")
    for h in hits:
        mm, ss = divmod(int(h["time_sec"]), 60)
        print(f"    {mm:02d}:{ss:02d}  frame {h['frame_id']:>6}  score {h['score']:.2f}  ({h['phrase']})")

    out = _contact_sheet(index, hits, args.query)
    print(f"[검색결과] 컨택트시트 저장: {out}")


def _search_index(index, query):
    q = query.lower().strip()
    hits = []
    for e in index["entries"]:
        hay = f"{e['key']} {e['phrase']}".lower()
        if q in hay or any(tok in hay for tok in q.split()):
            hits.append(e)
    return hits


def _search_live(index, query, box_thr):
    """인덱스에 없는 새 문장 -> 원본 프레임에 프롬프트를 직접 재실행."""
    src = index["source"]
    engine = build_engine(index["engine"])
    video = src if src and os.path.splitext(src)[1].lower() in (".mp4", ".avi", ".mov", ".mkv") else None
    images = src if video is None else None
    hits = []
    for fid, tsec, frame in iter_frames(video, images, index["every"]):
        for phrase, score, box in engine.detect(frame, [query], box_threshold=box_thr):
            hits.append({"frame_id": fid, "time_sec": round(tsec, 2),
                         "key": query, "phrase": phrase,
                         "score": round(score, 3), "box": [round(v, 1) for v in box]})
    return hits


def _grab_frame(index, frame_id):
    src = index["source"]
    if src and os.path.splitext(src)[1].lower() in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(src)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        cap.release()
        return frame if ok else None
    files = sorted(glob.glob(os.path.join(src, "*.jpg")) +
                   glob.glob(os.path.join(src, "*.png")) +
                   glob.glob(os.path.join(src, "*.jpeg")))
    return imread_kr(files[frame_id]) if frame_id < len(files) else None


def _contact_sheet(index, hits, query, cols=3, cell=380):
    tiles = []
    for h in hits:
        frame = _grab_frame(index, h["frame_id"])
        if frame is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in h["box"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 210, 0), 3)
        scale = cell / max(frame.shape[:2])
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        canvas = np.full((cell, cell, 3), 30, np.uint8)
        canvas[:frame.shape[0], :frame.shape[1]] = frame
        mm, ss = divmod(int(h["time_sec"]), 60)
        cv2.putText(canvas, f"{mm:02d}:{ss:02d}  s={h['score']:.2f}", (8, cell - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        tiles.append(canvas)

    if not tiles:
        return "(프레임 없음)"
    rows = []
    for i in range(0, len(tiles), cols):
        row = tiles[i:i + cols]
        while len(row) < cols:
            row.append(np.full((cell, cell, 3), 30, np.uint8))
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)

    safe = "".join(c if c.isalnum() else "_" for c in query)[:40]
    path = os.path.join(RESULTS_DIR, f"search_{safe}.jpg")
    imwrite_kr(path, sheet)
    return path


# ===========================================================================
# 샘플 영상 다운로드 (Pexels, 무료/저작권 표시 불필요)
# ===========================================================================

SAMPLE_VIDEOS = {
    # 인도를 걷는 사람들 (우산·자전거·행인 다양) 1080p, 약 8MB
    "sample.mp4": "https://www.pexels.com/download/video/854100/",
    # 도심 거리 (차량·버스·행인)
    "street.mp4": "https://www.pexels.com/download/video/1721294/",
}


def cmd_getvideo(args):
    import urllib.request

    class _R(urllib.request.HTTPRedirectHandler):
        def http_error_308(self, req, fp, c, m, h):
            return self.http_error_307(req, fp, c, m, h)

    o = urllib.request.build_opener(_R)
    o.addheaders = [("User-Agent", "Mozilla/5.0")]
    for name, url in SAMPLE_VIDEOS.items():
        dst = os.path.join(HERE, name)
        if os.path.exists(dst):
            print(f"[video] 이미 있음: {dst}")
            continue
        print(f"[video] 다운로드 {url} -> {dst}")
        with o.open(url, timeout=120) as r, open(dst, "wb") as f:
            f.write(r.read())
        print(f"        {os.path.getsize(dst) // 1024} KB")
    print("\n다음: python scene_search.py index --video sample.mp4 --prompt-version v3")


# ===========================================================================
# main
# ===========================================================================

def main():
    ap = argparse.ArgumentParser(description="자연어 장면 검색 아카이브")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="영상/사진첩을 인덱싱")
    pi.add_argument("--video")
    pi.add_argument("--images")
    pi.add_argument("--prompt-version", choices=list(VERSIONS), default="v3")
    pi.add_argument("--engine", choices=["dino", "yoloworld"], default="dino")
    pi.add_argument("--every", type=float, default=1.0, help="샘플링 간격(초)")
    pi.add_argument("--box-thr", type=float, default=0.30)
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", help="자연어로 장면 검색")
    ps.add_argument("--query", required=True)
    ps.add_argument("--topk", type=int, default=6)
    ps.add_argument("--live", action="store_true", help="인덱스에 없는 새 문장을 재추론")
    ps.add_argument("--box-thr", type=float, default=0.30)
    ps.add_argument("--min-score", type=float, default=0.0,
                    help="이 점수 미만 검출은 결과에서 제외 (오검출 억제)")
    ps.set_defaults(func=cmd_search)

    pl = sub.add_parser("log", help="프롬프트 개선 실험표 출력")
    pl.set_defaults(func=lambda a: print_log())

    pv = sub.add_parser("getvideo", help="무료 샘플 거리 영상 다운로드")
    pv.set_defaults(func=cmd_getvideo)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
