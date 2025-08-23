#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV(true,pred) から混同行列ヒートマップ・クラス別リコール・誤分類トップを可視化/保存
"""

import argparse
import os
from pathlib import Path

import numpy as np
import polars as pl
import matplotlib.pyplot as plt


def build_confusion_matrix(df: pl.DataFrame, labels: list[str], normalize: str) -> tuple[np.ndarray, np.ndarray]:
    """returns (raw_mat, display_mat)"""
    # クロス集計
    ct = (
        df.group_by(["true", "pred"])
        .len()
        .pivot(values="len", index="true", columns="pred")
        .fill_null(0)
    )
    # 行（真値）を全ラベルで埋める
    ct = pl.DataFrame({"true": labels}).join(ct, on="true", how="left").fill_null(0)
    # 列（予測）に足りないラベル列を追加
    for l in labels:
        if l not in ct.columns[1:]:
            ct = ct.with_columns(pl.lit(0).alias(l))
    ct = ct.select(["true"] + labels)

    raw = ct.select(pl.all().exclude("true")).to_numpy().astype(float)

    if normalize == "row":
        denom = raw.sum(axis=1, keepdims=True)
        disp = np.divide(raw, denom, out=np.zeros_like(raw), where=denom != 0)
    elif normalize == "col":
        denom = raw.sum(axis=0, keepdims=True)
        disp = np.divide(raw, denom, out=np.zeros_like(raw), where=denom != 0)
    else:
        disp = raw.copy()

    return raw, disp


def per_class_recall(df: pl.DataFrame) -> pl.DataFrame:
    support = df.group_by("true").len().rename({"len": "support"})
    correct = (
        df.filter(pl.col("true") == pl.col("pred"))
        .group_by("true").len().rename({"len": "correct"})
    )
    out = (
        support.join(correct, on="true", how="left")
        .fill_null(0)
        .with_columns((pl.col("correct") / pl.col("support")).alias("recall"))
        .sort("true")
    )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Visualize confusion matrix etc. from a CSV with columns true,pred."
    )
    ap.add_argument("--csv", required=True, help="入力CSVのパス（true,pred の2列を想定）")
    ap.add_argument("--true-col", default="true", help="真値列名（既定: true）")
    ap.add_argument("--pred-col", default="pred", help="予測列名（既定: pred）")
    ap.add_argument("--sep", default=",", help="区切り文字（既定: ,）")
    ap.add_argument("--labels", default=None, help="ラベル順序ファイル（1行1ラベル）。未指定なら自動推定&ソート")
    ap.add_argument("--normalize", choices=["none", "row", "col"], default="row",
                    help="ヒートマップの正規化（既定: row）")
    ap.add_argument("--topk", type=int, default=20, help="誤分類トップKを表示/保存（既定: 20）")
    ap.add_argument("--outdir", default="viz_out", help="画像/CSVの出力先ディレクトリ（既定: viz_out）")
    ap.add_argument("--show", action="store_true", help="描画も表示（既定は保存のみ）")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- データ読み込み
    df = pl.read_csv(args.csv, separator=args.sep)
    # 列名合わせ
    if args.true_col != "true":
        df = df.rename({args.true_col: "true"})
    if args.pred_col != "pred":
        df = df.rename({args.pred_col: "pred"})
    df = df.select(["true", "pred"]).with_columns(
        pl.col("true").cast(pl.String), pl.col("pred").cast(pl.String)
    )

    # --- ラベル順
    if args.labels:
        labels = [ln.strip() for ln in Path(args.labels).read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        labels = pl.concat([df["true"], df["pred"]]).unique().sort().to_list()

    # --- 混同行列
    raw_mat, disp_mat = build_confusion_matrix(df, labels, args.normalize)

    # プロット
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5), max(8, len(labels) * 0.5)))
    im = ax.imshow(disp_mat, aspect="auto")
    ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, rotation=90)
    ax.set_yticks(np.arange(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("pred"); ax.set_ylabel("true")
    ttl = "Confusion Matrix"
    if args.normalize != "none":
        ttl += f" (normalized by {args.normalize})"
    ax.set_title(ttl)
    # セル内注記：件数 + 率（正規化時）
    for i in range(raw_mat.shape[0]):
        for j in range(raw_mat.shape[1]):
            n = int(raw_mat[i, j])
            if args.normalize == "none":
                text = f"{n}" if n > 0 else ""
            else:
                pct = disp_mat[i, j] * 100.0
                text = f"{n}\n({pct:.0f}%)" if n > 0 else ""
            if text:
                ax.text(j, i, text, ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    cm_path = outdir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig)

    # --- クラス別リコール
    pc = per_class_recall(df)
    pc_path = outdir / "per_class_recall.csv"
    pc.write_csv(pc_path)

    # 棒グラフ
    fig2, ax2 = plt.subplots(figsize=(max(8, len(labels) * 0.4), 4))
    # 左結合で順序を labels に合わせる
    pc_ordered = pl.DataFrame({"true": labels}).join(pc, on="true", how="left").fill_null(0)
    x = np.arange(pc_ordered.height)
    ax2.bar(x, pc_ordered["recall"].to_numpy())
    ax2.set_xticks(x); ax2.set_xticklabels(pc_ordered["true"].to_list(), rotation=90)
    ax2.set_ylim(0, 1); ax2.set_ylabel("recall"); ax2.set_title("Per-class recall")
    plt.tight_layout()
    recall_png = outdir / "per_class_recall.png"
    fig2.savefig(recall_png, dpi=200)
    if args.show:
        plt.show()
    plt.close(fig2)

    # --- 誤分類トップ
    mistakes = (
        df.filter(pl.col("true") != pl.col("pred"))
        .group_by(["true", "pred"]).len()
        .sort("len", descending=True)
    )
    top_path = outdir / "top_mistakes.csv"
    mistakes.write_csv(top_path)
    print(mistakes.head(args.topk))

    # --- 全体精度
    acc = (df["true"] == df["pred"]).mean()
    print(f"Accuracy: {acc:.4f}")

    print(f"\nSaved:\n- {cm_path}\n- {pc_path}\n- {recall_png}\n- {top_path}")


if __name__ == "__main__":
    main()
