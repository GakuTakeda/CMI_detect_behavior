import optuna
from optuna.trial import TrialState
import pandas as pd
import os

STUDY_NAME = "imu_tuning"
STORAGE = "sqlite:///lstmgru_tuning.db"  # ここは絶対パスにしておくと安全

# 既存のスタディをロード
study = optuna.load_study(study_name=STUDY_NAME, storage=STORAGE)

# ベスト試行
best = study.best_trial
print("best number:", best.number)
print("best value :", best.value)     # direction="maximize"なので大きいほど良い
print("best params:", best.params)

# すべての試行をDataFrameで
df = study.trials_dataframe(attrs=(
    "number","value","state","params","user_attrs",
    "system_attrs","datetime_start","datetime_complete","duration"
))
# 完了試行だけに絞る
df_complete = df[df["state"] == "COMPLETE"].copy()

