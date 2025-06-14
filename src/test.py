import joblib, os, numpy as np, pandas as pd, pathlib
import warnings 
warnings.filterwarnings("ignore")


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl

def time_sum(x):
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])


# Residual CNN Block with SE
def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

# MixUp the data argumentation in order to regularize the neural network. 

class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix
    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU deep branch
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.3, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.3, wd=wd)

    # TOF/Thermal lighter branch
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.3)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.3)(x2)

    merged = Concatenate()([x1, x2])

    x = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)

# ──────────────────────────────────────────
# 0) 出力ディレクトリを指定
#    Hydra の run.dir をそのままコピーしたパス例
PRETRAINED_DIR = pathlib.Path(
    "../outputs/2025-06-07/15-58-24/output"   # ← 実際の run フォルダに置き換える
)

# ──────────────────────────────────────────
# 1) メタ情報をロード
pad_len          = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
feature_cols     = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
gesture_classes  = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)
scaler           = joblib.load(PRETRAINED_DIR / "scaler.pkl")
MODEL_PATHS      = [PRETRAINED_DIR / f"best_of_fold_{i}.h5" for i in range(1, 6)]
GPU_IDS          = 0                       # 使う GPU 一覧
ensemble_models  = None 

CUSTOM_OBJECTS = {
    # util 関数
    "time_sum":        tf.keras.backend.sum,        # K.sum と等価
    "squeeze_last_axis": tf.squeeze,
    "expand_last_axis":  tf.expand_dims,
    # レイヤ
    "se_block":           se_block,
    "residual_se_cnn_block": residual_se_cnn_block,
    "attention_layer":    attention_layer,
}

def _init_gpus_and_load_models():
    """GPU 制御 + モデル 5 個を 2 GPU に常駐させる"""
    global ensemble_models

    # 1) 使う GPU だけ可視化し、メモリを必要分ずつ確保
    phys_gpus = tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices([phys_gpus[i] for i in GPU_IDS], "GPU")
    for i in GPU_IDS:
        tf.config.experimental.set_memory_growth(phys_gpus[i], True)

    # 2) モデルをラウンドロビンで GPU にロード
    ensemble_models = []
    for idx, path in enumerate(MODEL_PATHS):
        gpu_id = GPU_IDS[idx % len(GPU_IDS)]     # 例: 0,1,0,1,0
        with tf.device(f"/GPU:{gpu_id}"):
            m = load_model(path,
                           custom_objects=CUSTOM_OBJECTS,
                           compile=False)
        ensemble_models.append(m)

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Kaggle の inference API 互換:
        sequence   : pl.DataFrame  (可変長 IMU / ToF 等の時系列)
        demographics: pl.DataFrame (今回は無視)
        戻り値       : 予測ジェスチャクラス (str)
    """
    global gesture_classes, ensemble_models

    # ---------- ① 必要なら初期化 ----------
    if gesture_classes is None:
        gesture_classes = np.load(
            PRETRAINED_DIR / "gesture_classes.npy",
            allow_pickle=True
        )
    if ensemble_models is None:
        _init_gpus_and_load_models()

    # ---------- ② 前処理 ----------
    df = sequence.to_pandas()
    mat = preprocess_sequence(df, feature_cols, scaler)          # shape (T, D)
    pad = pad_sequences([mat], maxlen=pad_len, padding='post',
                        truncating='post', dtype='float32')       # shape (1, pad_len, D)

    # ---------- ③ 2 GPU 上の 5 モデルで同時推論 ----------
    # モデルはあらかじめ別 GPU に載っているので for ループでも並列実行される
    probs = np.zeros((gesture_classes.size,), dtype=np.float32)
    for m in ensemble_models:
        probs += m.predict(pad, verbose=0)[0]                    # ソフトマックス和

    probs /= len(ensemble_models)                                # 平均
    idx   = int(probs.argmax())                                  # 予測クラス index
    return str(gesture_classes[idx])

import kaggle_evaluation.cmi_inference_server
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('../data/KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '../data/test.csv',
            '../data/test_demographics.csv',
        )
    )

