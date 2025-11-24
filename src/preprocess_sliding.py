#!/usr/bin/env python3
"""Sliding Window 방식 전처리 - Causal Training용"""
import json
import csv
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import deque, defaultdict
import numpy as np
import pywt

# 정규화 기준값
BASE_MAG = (-33.0, -15.0, -42.0)
COORD_CENTER = (-41.0, 0.0)
COORD_SCALE = 50.0

def normalize_mag(val: float, base: float) -> float:
    return (val - base) / 10.0

def normalize_coord(x: float, y: float) -> Tuple[float, float]:
    x_norm = (x - COORD_CENTER[0]) / COORD_SCALE
    y_norm = (y - COORD_CENTER[1]) / COORD_SCALE
    return (x_norm, y_norm)

def wavelet_denoise(signal: List[float], wavelet='db4', level=3) -> List[float]:
    """Wavelet denoising"""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(denoised_coeffs, wavelet).tolist()

def read_nodes(path: Path) -> Tuple[Dict[int, Tuple[float, float]], set]:
    """노드 위치 및 회전 노드 읽기"""
    positions = {}
    turn_nodes = set()
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = int(row["id"])
            x = float(row["x_m"])
            y = float(row["y_m"])
            node_type = row.get("type", "marker")
            positions[node_id] = (x, y)
            if node_type == "turn":
                turn_nodes.add(node_id)
    return positions, turn_nodes

def build_graph(positions: Dict[int, Tuple[float, float]]) -> Dict[int, List[Tuple[int, float]]]:
    """그래프 구축 - 복도 구조만 연결 (같은 행/열)"""
    graph = {node: [] for node in positions}
    nodes = sorted(positions.keys())

    # 물리적으로 연결 불가능한 쌍 (벽 등)
    blocked_connections = {(10, 28), (28, 10), (24, 25), (25, 24)}

    for i, a in enumerate(nodes):
        for b in nodes[i+1:]:
            # 차단된 연결 확인
            if (a, b) in blocked_connections or (b, a) in blocked_connections:
                continue

            xa, ya = positions[a]
            xb, yb = positions[b]
            # Manhattan distance (복도 구조: 대각선 이동 불가)
            dist = abs(xb - xa) + abs(yb - ya)

            # 5m 이하이고 같은 행/열만 연결
            if dist <= 5.0:
                same_row = abs(ya - yb) < 0.5  # y 차이 < 0.5m
                same_col = abs(xa - xb) < 0.5  # x 차이 < 0.5m

                if same_row or same_col:
                    graph[a].append((b, dist))
                    graph[b].append((a, dist))

    return graph

def find_shortest_path(graph: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> List[int]:
    """BFS로 최단 경로 찾기"""
    if start == end:
        return [start]

    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        node, path = queue.popleft()

        for neighbor, _ in graph[node]:
            if neighbor in visited:
                continue

            new_path = path + [neighbor]

            if neighbor == end:
                return new_path

            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return None  # 경로 없음

def get_turn_waypoints(
    path_nodes: List[int],
    turn_nodes: set
) -> List[int]:
    """경로에서 회전 노드만 추출 (시작/끝 포함)"""
    waypoints = [path_nodes[0]]  # 시작 노드

    # 중간의 회전 노드만 추가
    for node in path_nodes[1:-1]:
        if node in turn_nodes:
            waypoints.append(node)

    waypoints.append(path_nodes[-1])  # 끝 노드
    return waypoints

def interpolate_along_waypoints(
    step_idx: int,
    total_steps: int,
    waypoints: List[int],
    positions: Dict[int, Tuple[float, float]]
) -> Tuple[float, float]:
    """회전 노드를 기준으로 세그먼트별 선형 보간"""

    # 각 세그먼트(회전 노드 간) 거리 계산 (Manhattan distance)
    segment_lengths = []
    for i in range(len(waypoints) - 1):
        p1 = positions[waypoints[i]]
        p2 = positions[waypoints[i+1]]
        dist = abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])
        segment_lengths.append(dist)

    total_length = sum(segment_lengths)

    # 현재 스텝의 진행률
    progress = step_idx / (total_steps - 1) if total_steps > 1 else 0.5
    target_dist = progress * total_length

    # 어느 세그먼트에 있는지 찾기
    cumulative = 0
    for i, seg_len in enumerate(segment_lengths):
        if cumulative + seg_len >= target_dist or i == len(segment_lengths) - 1:
            # i번째 세그먼트에 있음 (선형 보간)
            seg_progress = (target_dist - cumulative) / seg_len if seg_len > 0 else 0.5

            p1 = positions[waypoints[i]]
            p2 = positions[waypoints[i+1]]

            x = p1[0] + seg_progress * (p2[0] - p1[0])
            y = p1[1] + seg_progress * (p2[1] - p1[1])

            return (x, y)

        cumulative += seg_len

    # 끝점 반환
    return positions[waypoints[-1]]

def process_csv_sliding(
    file_path: Path,
    positions: Dict[int, Tuple[float, float]],
    graph: Dict[int, List[Tuple[int, float]]],
    turn_nodes: set,
    feature_mode: str = "mag3",
    window_size: int = 250,
    stride: int = 50,
    debug_count: List[int] = None,
) -> List[Dict]:
    """CSV를 sliding window로 처리

    Returns:
        List of samples, each: {"features": [250, n_features], "target": [x, y]}
    """
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < window_size:
        return []

    # 센서 데이터 추출
    try:
        magx = [float(row["MagX"]) for row in rows]
        magy = [float(row["MagY"]) for row in rows]
        magz = [float(row["MagZ"]) for row in rows]
    except (KeyError, ValueError):
        return []

    # Wavelet denoising
    clean_magx = wavelet_denoise(magx)
    clean_magy = wavelet_denoise(magy)
    clean_magz = wavelet_denoise(magz)

    # 경로 정보로 위치 얻기
    parts = file_path.stem.split("_")
    if len(parts) < 2:
        return []

    try:
        start_node = int(parts[0])
        end_node = int(parts[1])
    except ValueError:
        return []

    if start_node not in positions or end_node not in positions:
        return []

    # 최단 경로 찾기 (BFS)
    path = find_shortest_path(graph, start_node, end_node)

    if path is None:
        # 경로를 찾을 수 없으면 스킵
        print(f"  ⚠️  {file_path.name}: 경로 없음 ({start_node}→{end_node})")
        return []

    # 회전 노드만 추출 (waypoints)
    waypoints = get_turn_waypoints(path, turn_nodes)

    # 경로 정보 출력 (디버그용, 처음 10개만)
    if debug_count is not None and len(waypoints) > 2 and debug_count[0] < 10:
        print(f"  🛤️  {file_path.name}: {start_node}→{end_node}")
        print(f"      전체 경로: {path}")
        print(f"      회전 포인트: {waypoints} ({len(waypoints)-2}개 회전)")
        debug_count[0] += 1

    # 회전 노드 기준 세그먼트별 선형 보간
    num_steps = len(rows)
    positions_list = []
    for i in range(num_steps):
        pos = interpolate_along_waypoints(i, num_steps, waypoints, positions)
        positions_list.append(pos)

    # Adaptive normalization: 파일별 평균/std 계산 (캘리브레이션 drift 처리)
    magx_mean = np.mean(clean_magx)
    magy_mean = np.mean(clean_magy)
    magz_mean = np.mean(clean_magz)
    magx_std = np.std(clean_magx)
    magy_std = np.std(clean_magy)
    magz_std = np.std(clean_magz)

    # 안전 장치: std가 너무 작으면 (정지 상태) 최소값 사용
    magx_std = max(magx_std, 1e-6)
    magy_std = max(magy_std, 1e-6)
    magz_std = max(magz_std, 1e-6)

    # Feature 모드에 따라 특징 생성
    features_list = []
    for i in range(num_steps):
        # Z-score normalization (adaptive)
        magx_norm = (clean_magx[i] - magx_mean) / magx_std
        magy_norm = (clean_magy[i] - magy_mean) / magy_std
        magz_norm = (clean_magz[i] - magz_mean) / magz_std

        if feature_mode == "mag3":
            feat = [magx_norm, magy_norm, magz_norm]
        elif feature_mode == "mag4":
            # Magnitude도 adaptive하게
            mag_magnitude = math.sqrt(clean_magx[i]**2 + clean_magy[i]**2 + clean_magz[i]**2)
            mag_array = [math.sqrt(clean_magx[j]**2 + clean_magy[j]**2 + clean_magz[j]**2)
                        for j in range(len(clean_magx))]
            mag_mean = np.mean(mag_array)
            mag_std = max(np.std(mag_array), 1e-6)
            mag_magnitude_norm = (mag_magnitude - mag_mean) / mag_std
            feat = [magx_norm, magy_norm, magz_norm, mag_magnitude_norm]
        elif feature_mode == "full":
            pitch = float(rows[i]["Pitch"])
            roll = float(rows[i]["Roll"])
            yaw = float(rows[i]["Yaw"])
            pitch_norm = pitch / 180.0
            roll_norm = roll / 180.0
            yaw_norm = yaw / 180.0
            feat = [magx_norm, magy_norm, magz_norm, pitch_norm, roll_norm, yaw_norm]
        else:
            raise ValueError(f"Unknown feature_mode: {feature_mode}")

        features_list.append(feat)

    # Sliding window 생성
    samples = []
    for i in range(0, num_steps - window_size + 1, stride):
        window_features = features_list[i:i + window_size]  # [250, n_features]

        # 마지막 타임스텝 위치가 label
        last_idx = i + window_size - 1
        last_pos = positions_list[last_idx]
        target = normalize_coord(last_pos[0], last_pos[1])  # (x_norm, y_norm)

        sample = {
            "features": window_features,
            "target": list(target)
        }
        samples.append(sample)

    return samples

def preprocess_sliding(
    raw_dir: Path,
    nodes_path: Path,
    output_dir: Path,
    feature_mode: str = "mag3",
    window_size: int = 250,
    stride: int = 50,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
):
    """Sliding window 방식 전처리"""
    print("=" * 80)
    print("🔄 Sliding Window 전처리 시작")
    print("=" * 80)
    print(f"  Feature mode: {feature_mode}")
    print(f"  Window size: {window_size}")
    print(f"  Stride: {stride}")
    print()

    # 노드 및 그래프
    positions, turn_nodes = read_nodes(nodes_path)
    graph = build_graph(positions)

    print(f"🔄 회전 가능 노드: {sorted(turn_nodes)}")
    print()

    # 모든 CSV 파일 처리 (경로별로 그룹화)
    path_to_samples = defaultdict(list)
    csv_files = list(raw_dir.glob("*.csv"))

    print(f"📂 총 {len(csv_files)}개 파일 처리 중...")
    print(f"   (경로 찾기 활성화: 회전 경로 자동 감지)\n")

    debug_count = [0]  # mutable counter
    for csv_file in csv_files:
        samples = process_csv_sliding(
            csv_file, positions, graph, turn_nodes, feature_mode, window_size, stride, debug_count
        )
        if samples:
            # 경로 ID 추출 (파일명: "1_23_0.csv" → path_id: "1_23")
            parts = csv_file.stem.split("_")
            path_id = f"{parts[0]}_{parts[1]}"
            path_to_samples[path_id].extend(samples)

    # 전체 샘플 수 계산
    total_samples = sum(len(samples) for samples in path_to_samples.values())
    print(f"✅ 총 {total_samples}개 샘플 생성 ({len(path_to_samples)}개 경로)")
    print()

    # Train/Val/Test 분할 (경로 기반 - 층화 추출)
    print("📊 경로 기반 층화 분할 수행 중...")
    paths = list(path_to_samples.keys())
    random.shuffle(paths)

    n_paths = len(paths)
    n_train_paths = int(n_paths * train_ratio)
    n_val_paths = int(n_paths * val_ratio)

    train_paths = paths[:n_train_paths]
    val_paths = paths[n_train_paths:n_train_paths + n_val_paths]
    test_paths = paths[n_train_paths + n_val_paths:]

    # 각 split의 샘플 수집
    train_samples = []
    val_samples = []
    test_samples = []

    for path in train_paths:
        train_samples.extend(path_to_samples[path])
    for path in val_paths:
        val_samples.extend(path_to_samples[path])
    for path in test_paths:
        test_samples.extend(path_to_samples[path])

    print(f"  Train: {len(train_samples)}개 샘플 ({len(train_paths)}개 경로)")
    print(f"  Val:   {len(val_samples)}개 샘플 ({len(val_paths)}개 경로)")
    print(f"  Test:  {len(test_samples)}개 샘플 ({len(test_paths)}개 경로)")
    print()

    # 저장
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, samples in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        output_path = output_dir / f"{split_name}.jsonl"
        with output_path.open("w") as f:
            for sample in samples:
                f.write(json.dumps(sample) + "\n")
        print(f"💾 {output_path} 저장 완료")

    # 메타데이터 저장
    n_features = len(train_samples[0]["features"][0])
    meta = {
        "feature_mode": feature_mode,
        "n_features": n_features,
        "window_size": window_size,
        "stride": stride,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
    }

    meta_path = output_dir / "meta.json"
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"💾 {meta_path} 저장 완료")

    print()
    print("=" * 80)
    print("✅ 전처리 완료!")
    print("=" * 80)

    return meta

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="원본 CSV 디렉토리")
    parser.add_argument("--nodes", default="data/nodes_final.csv")
    parser.add_argument("--output", default="data/sliding", help="출력 디렉토리")
    parser.add_argument("--feature-mode", default="mag3", choices=["mag3", "mag4", "full"])
    parser.add_argument("--window-size", type=int, default=250)
    parser.add_argument("--stride", type=int, default=50)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    args = parser.parse_args()

    preprocess_sliding(
        raw_dir=Path(args.raw_dir),
        nodes_path=Path(args.nodes),
        output_dir=Path(args.output),
        feature_mode=args.feature_mode,
        window_size=args.window_size,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
