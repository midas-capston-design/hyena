#!/usr/bin/env python3
"""additional_data 품질 검사"""
import csv
from pathlib import Path
import numpy as np
import math

# 정규화 기준값
BASE_MAG = (-33.0, -15.0, -42.0)

def analyze_quality(file_path):
    """파일 품질 점수 계산"""
    with file_path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 100:
        return None

    try:
        magx = np.array([float(row["MagX"]) for row in rows])
        magy = np.array([float(row["MagY"]) for row in rows])
        magz = np.array([float(row["MagZ"]) for row in rows])

        quality = {
            "filename": file_path.name,
            "length": len(rows),
            "magx_mean": np.mean(magx),
            "magy_mean": np.mean(magy),
            "magz_mean": np.mean(magz),
            "magx_std": np.std(magx),
            "magy_std": np.std(magy),
            "magz_std": np.std(magz),
        }

        # 경로 정보
        parts = file_path.stem.split("_")
        if len(parts) >= 2:
            quality["path"] = f"{parts[0]}->{parts[1]}"
            quality["start"] = int(parts[0])
            quality["end"] = int(parts[1])

        # 품질 점수 계산
        score = 0

        # 1. 길이 점수 (500 이상이면 좋음)
        if quality["length"] >= 1000:
            score += 3
        elif quality["length"] >= 500:
            score += 2
        elif quality["length"] >= 250:
            score += 1

        # 2. 센서 안정성 (std가 너무 작거나 크면 나쁨)
        if 5 < quality["magx_std"] < 20:
            score += 2
        elif 3 < quality["magx_std"] < 30:
            score += 1

        # 3. 노이즈 체크 (급격한 점프)
        jumps = np.sum(np.abs(np.diff(magx)) > 30)
        if jumps < len(magx) * 0.01:  # 1% 미만
            score += 2
        elif jumps < len(magx) * 0.05:  # 5% 미만
            score += 1

        quality["score"] = score
        quality["jumps"] = jumps

        # 자기장 이상치 (BASE_MAG 기준)
        outlier_x = abs(quality["magx_mean"] - BASE_MAG[0])
        outlier_y = abs(quality["magy_mean"] - BASE_MAG[1])
        outlier_z = abs(quality["magz_mean"] - BASE_MAG[2])
        quality["outlier_score"] = outlier_x + outlier_y + outlier_z

        # 움직임 (센서 변화량)
        quality["movement"] = quality["magx_std"] + quality["magy_std"] + quality["magz_std"]

        return quality

    except Exception as e:
        print(f"❌ {file_path.name}: 분석 실패 - {e}")
        return None

def main():
    data_dir = Path("additional_data")
    csv_files = sorted(data_dir.glob("*.csv"))

    print("=" * 100)
    print("📊 additional_data 품질 검사")
    print("=" * 100)
    print(f"\n총 {len(csv_files)}개 파일 분석 중...\n")

    all_files = []
    for f in csv_files:
        q = analyze_quality(f)
        if q:
            all_files.append(q)

    if not all_files:
        print("❌ 분석할 파일이 없습니다.")
        return

    print(f"✅ {len(all_files)}개 파일 분석 완료\n")

    # ============================================================================
    # 1. 전체 요약
    # ============================================================================
    print("=" * 100)
    print("1. 전체 품질 요약")
    print("=" * 100)

    scores = [f["score"] for f in all_files]
    lengths = [f["length"] for f in all_files]
    outlier_scores = [f["outlier_score"] for f in all_files]
    movements = [f["movement"] for f in all_files]

    print(f"\n품질 점수: 평균 {np.mean(scores):.2f} (범위: {min(scores)} ~ {max(scores)})")
    print(f"길이: 평균 {np.mean(lengths):.0f} (범위: {min(lengths)} ~ {max(lengths)})")
    print(f"Outlier 점수: 평균 {np.mean(outlier_scores):.1f}")
    print(f"움직임: 평균 {np.mean(movements):.2f}")

    # ============================================================================
    # 2. 파일별 상세 정보
    # ============================================================================
    print("\n" + "=" * 100)
    print("2. 파일별 상세 정보")
    print("=" * 100)
    print()
    print(f"{'파일명':<20} {'경로':<12} {'점수':<5} {'길이':<6} {'MagX 평균':<10} {'Std':<8} {'점프':<6} {'Outlier':<8} {'움직임':<8}")
    print("-" * 100)

    for f in sorted(all_files, key=lambda x: x["score"], reverse=True):
        print(f"{f['filename']:<20} {f.get('path', 'N/A'):<12} {f['score']:<5} "
              f"{f['length']:<6} {f['magx_mean']:<10.2f} {f['magx_std']:<8.2f} "
              f"{f['jumps']:<6} {f['outlier_score']:<8.1f} {f['movement']:<8.2f}")

    # ============================================================================
    # 3. 경로별 분석
    # ============================================================================
    print("\n" + "=" * 100)
    print("3. 경로별 샘플 수")
    print("=" * 100)

    from collections import defaultdict
    path_counts = defaultdict(list)
    for f in all_files:
        if "path" in f:
            path_counts[f["path"]].append(f)

    print()
    for path, files in sorted(path_counts.items()):
        avg_score = np.mean([f["score"] for f in files])
        print(f"  {path:<12}: {len(files)}개 샘플, 평균 점수 {avg_score:.2f}")

    # ============================================================================
    # 4. 품질 판정
    # ============================================================================
    print("\n" + "=" * 100)
    print("4. 품질 판정")
    print("=" * 100)

    good_files = [f for f in all_files if f["score"] >= 5]
    ok_files = [f for f in all_files if 3 <= f["score"] < 5]
    bad_files = [f for f in all_files if f["score"] < 3]

    print(f"\n✅ 좋음 (점수 ≥ 5): {len(good_files)}개")
    for f in good_files:
        print(f"   {f['filename']}: 점수={f['score']}, 길이={f['length']}")

    print(f"\n⚠️  보통 (점수 3-4): {len(ok_files)}개")
    for f in ok_files:
        print(f"   {f['filename']}: 점수={f['score']}, 길이={f['length']}")

    print(f"\n❌ 나쁨 (점수 < 3): {len(bad_files)}개")
    for f in bad_files:
        issues = []
        if f["length"] < 250:
            issues.append(f"짧음({f['length']})")
        if f["magx_std"] < 3 or f["magx_std"] > 30:
            issues.append(f"불안정(std={f['magx_std']:.1f})")
        if f["jumps"] > f["length"] * 0.05:
            issues.append(f"노이즈({f['jumps']}점프)")
        if f["outlier_score"] > 20:
            issues.append(f"outlier({f['outlier_score']:.1f})")
        if f["movement"] < 5:
            issues.append(f"움직임없음({f['movement']:.1f})")

        print(f"   {f['filename']}: 점수={f['score']}, 문제={', '.join(issues)}")

    # ============================================================================
    # 5. 최종 권장사항
    # ============================================================================
    print("\n" + "=" * 100)
    print("🎯 최종 권장사항")
    print("=" * 100)

    raw_style = [f for f in all_files if f["magx_mean"] < 0 and abs(f["outlier_score"]) < 20]

    print(f"""
📊 품질 분포:
   좋음 (≥5점): {len(good_files)}개
   보통 (3-4점): {len(ok_files)}개
   나쁨 (<3점): {len(bad_files)}개

🎯 사용 권장:
   Raw 스타일 (MagX < 0, outlier < 20): {len(raw_style)}개
   → 현재 BASE_MAG으로 바로 사용 가능!

⚠️  주의 필요:
   Outlier 높음 (> 20): {len([f for f in all_files if f['outlier_score'] > 20])}개
   움직임 적음 (< 5): {len([f for f in all_files if f['movement'] < 5])}개
   점프 많음 (> 5%): {len([f for f in all_files if f['jumps'] > f['length'] * 0.05])}개

💡 제안:
   1. 좋음 + 보통 ({len(good_files) + len(ok_files)}개) → data/raw/에 추가
   2. 나쁨 ({len(bad_files)}개) → 제외 또는 재수집
""")

    # Raw 스타일 파일 리스트
    if raw_style:
        print("✅ Raw 스타일 파일 (바로 사용 가능):")
        for f in raw_style:
            print(f"   {f['filename']}")

    print("\n" + "=" * 100)

if __name__ == "__main__":
    main()
