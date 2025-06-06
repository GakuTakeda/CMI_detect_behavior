# main.py
import os
import pathlib
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CompetitionMetric は hold-out での評価用
from cmi_2025_metric_copy_for_import import CompetitionMetric
from utils import preprocess_sequence, build_two_branch_model, MixupGenerator


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Hydra を使って設定を読み込み、5-fold 交差検証を行うスクリプト
    """
    # ---------------------------------------------------------------------------- #
    # 1) 基本パラメータの展開
    # ---------------------------------------------------------------------------- #
    raw_dir     = pathlib.Path(cfg.param.raw_dir)
    export_dir  = HydraConfig.get().runtime.output_dir / pathlib.Path(cfg.param.export_dir)
    train_csv   = cfg.param.train_csv

    PAD_PERCENTILE = cfg.param.pad_percentile
    BATCH_SIZE     = cfg.param.batch_size
    LR_INIT        = cfg.param.lr_init
    WD             = cfg.param.wd
    MIXUP_ALPHA    = cfg.param.mixup_alpha
    EPOCHS         = cfg.param.epochs
    PATIENCE       = cfg.param.patience
    N_SPLITS       = cfg.param.n_splits
    RANDOM_SEED    = cfg.param.random_seed

    # 出力ディレクトリを作っておく
    export_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------- #
    # 2) データ読み込み & 前処理（ラベルエンコード・特徴量選定）
    # ---------------------------------------------------------------------------- #
    print("▶ Loading dataset …")
    df = pd.read_csv(raw_dir / train_csv)

    # ラベルエンコード
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    # 全クラスの順序を numpy 配列で保存しておく
    np.save(export_dir / "gesture_classes.npy", le.classes_)

    # メタ情報を除いた特徴量リストを作る
    meta_cols = {
        'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
        'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'
    }
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # IMU系 と TOF/THM 系の列を分ける
    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    # ---------------------------------------------------------------------------- #
    # 3) シーケンスごとの行列化 & パディング長の決定
    # ---------------------------------------------------------------------------- #
    # 各 sequence_id ごとに前処理を行い、特徴行列を作る
    seq_gp = df.groupby('sequence_id')
    X_list, y_list, lens = [], [], []
    for seq_id, seq in seq_gp:
        # preprocess_sequence は (pandas.DataFrame, feature_cols, scaler など) を受け取って
        # (N_steps, N_features) の numpy 配列を返す既存関数とする
        # ここでは暫定的にスケーラーなしで行列を作り、あとで scaler を適用する
        mat = seq[feature_cols].ffill().bfill().fillna(0).values  # 一時的にスケーリング前の行列
        X_list.append(mat)
        # シーケンスごとのラベルは先頭の gesture_int と仮定
        y_list.append(seq['gesture_int'].iloc[0])
        lens.append(len(mat))

    # pad_len を percentile で決定
    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(export_dir / "sequence_maxlen.npy", pad_len)
    # feature_cols を保存しておく
    np.save(export_dir / "feature_cols.npy", np.array(feature_cols))

    # ---------------------------------------------------------------------------- #
    # 4) グローバルスケーラーの作成
    #    ※ 交差検証では fold ごとにスケーラーを作り直す方法もあるが、
    #      ここでは全データで fit しておき、各 fold 内で transform のみを行う方針とする
    # ---------------------------------------------------------------------------- #
    print("▶ Fitting global scaler on all data …")
    # X_list は各シーケンスごとに (N_steps, N_features) のリストなので、
    # 一度すべて縦に連結してから StandardScaler にかける
    all_data = np.vstack(X_list)
    scaler = StandardScaler().fit(all_data)
    joblib.dump(scaler, export_dir / "scaler.pkl")

    # ---------------------------------------------------------------------------- #
    # 5) すべてのシーケンスに対してスケーラーとパディングを適用し、最終的なテンソルを作成
    # ---------------------------------------------------------------------------- #
    print("▶ Applying scaler and padding to all sequences …")
    X_padded = []
    for mat in X_list:
        mat_scaled = scaler.transform(mat)  # (N_steps, N_features)
        # pad_sequences は 2D → 3D にするイメージ (N_steps → pad_len, N_features)
        # Keras の pad_sequences は list of 2D をうまく扱ってくれるので、
        # dtype='float32' 指定しておけば、最後に (N_seqs, pad_len, N_features) の配列が取れる
        X_padded.append(mat_scaled)

    # ここで pad_sequences を使って 3D テンソルにする
    X = pad_sequences(
        X_padded,
        maxlen=pad_len,
        padding='post',
        truncating='post',
        dtype='float32'
    )  # 形状: (N_sequences, pad_len, N_features)

    # one-hot にする前に y_list を numpy array にしておく
    y_array = np.array(y_list)

    # ---------------------------------------------------------------------------- #
    # 6) 5-Fold Stratified Split の準備
    # ---------------------------------------------------------------------------- #
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    # 各 fold の結果を保存するための辞書（任意で用途に合わせて拡張可）
    fold_metrics = {}

    # ---------------------------------------------------------------------------- #
    # 7) Fold ごとの学習ループ
    # ---------------------------------------------------------------------------- #
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_array)):
        print(f"\n=== Fold {fold+1}/{N_SPLITS} ===")


        # train/val のデータを取り出す
        X_tr = X[train_idx]
        X_val = X[val_idx]
        y_tr = y_array[train_idx]
        y_val = y_array[val_idx]

        # one-hot エンコーディング
        num_classes = len(le.classes_)
        y_tr_cat = to_categorical(y_tr, num_classes=num_classes)
        y_val_cat = to_categorical(y_val, num_classes=num_classes)

        # クラス重みを計算（train データに対してのみ）
        cw_vals = compute_class_weight(
            'balanced',
            classes=np.arange(num_classes),
            y=y_tr
        )
        class_weight = dict(enumerate(cw_vals))

        # モデル構築：引数は (pad_len, len(imu_cols), len(tof_cols), num_classes, wd=WD) と仮定
        model = build_two_branch_model(
            pad_len=pad_len,
            imu_dim=len(imu_cols),
            tof_dim=len(tof_cols),
            n_classes=num_classes,
            wd=WD
        )

        # 学習率スケジューラ
        steps_per_epoch = len(X_tr) // BATCH_SIZE
        lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(
            initial_learning_rate=LR_INIT,
            first_decay_steps=5 * max(1, steps_per_epoch)
        )

        optimizer = Adam(learning_rate=lr_sched)
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy']
        )

        # Mixup ジェネレータ（既存のクラスを流用）
        train_gen = MixupGenerator(X_tr, y_tr_cat, batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)

        # コールバック（EarlyStopping）
        es_cb = tf.keras.callbacks.EarlyStopping(
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        ckpt_cb = ModelCheckpoint(
            filepath=str(export_dir / f"best_of_fold_{fold+1}.h5"),
            monitor="val_loss",       # or "val_accuracy"
            mode="min",               # "max" にするなら monitor を val_accuracy に
            save_best_only=True,      # 最良モデルだけ残す
            save_weights_only=False,  # True にすると .h5 が重みだけになる
            verbose=1
        )

        # ---------------------------------------------------------------------------- #
        # 7-1) モデルの学習
        # ---------------------------------------------------------------------------- #
        print(f"▶ Fold {fold+1}: Training start …")
        history = model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=(X_val, y_val_cat),
            class_weight=class_weight,
            callbacks=[es_cb, ckpt_cb],   # ★ ここを更新 ★
            verbose=1
        )


        # ---------------------------------------------------------------------------- #
        # 7-3) Hold-out （ここでは validation）の評価指標計算
        # ---------------------------------------------------------------------------- #
        print(f"▶ Fold {fold+1}: Evaluating on validation set …")
        preds = model.predict(X_val).argmax(axis=1)
        true_labels = y_val  # 整数ラベル
        # CompetitionMetric を使って Hierarchical F1 を計算（クラス名の DataFrame を作成）
        df_true = pd.DataFrame({'gesture': le.classes_[true_labels]})
        df_pred = pd.DataFrame({'gesture': le.classes_[preds]})
        h_f1 = CompetitionMetric().calculate_hierarchical_f1(df_true, df_pred)

        print(f"▶ Fold {fold+1}: Hold-out H-F1 = {h_f1:.4f}")
        fold_metrics[f"fold_{fold+1}"] = float(h_f1)

        # ────────────────────────────────────────────────────────────────────────────── #

    # ---------------------------------------------------------------------------- #
    # 8) 全 Fold の結果をまとめて保存・表示
    # ---------------------------------------------------------------------------- #
    print("\n=== 5-Fold CV Summary ===")
    for k, v in fold_metrics.items():
        print(f"  {k}: H-F1 = {v:.4f}")
    avg_score = np.mean(list(fold_metrics.values()))
    print(f"\n  Average H-F1 over {N_SPLITS} folds: {avg_score:.4f}")

    # 必要に応じて、fold_metrics を JSON や CSV で書き出してもよい
    metrics_df = pd.DataFrame.from_dict(fold_metrics, orient='index', columns=['H_F1'])
    metrics_df.loc['average'] = avg_score
    metrics_df.to_csv(export_dir / "5fold_results.csv", index=True)

    print("✔ All folds done. Artefacts saved in", export_dir)


if __name__ == "__main__":
    main()

