import pandas as pd
import numpy as np

from catboost import CatBoostClassifier, CatBoostRegressor

# =========================================================
# 0. 파일 경로
TRAIN_PATH = "train_merged_featured_재현.csv"
TEST_PATH = "test_merged_featured_재현.csv"
SAMPLE_SUB_PATH = "sample_submission.csv"
OUTPUT_PATH = "submission_final.csv"

# =========================================================
# 1. 데이터 불러오기
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)

print("=" * 80)
print("train shape:", train_df.shape)
print("test shape :", test_df.shape)
print("sample shape:", sample_sub.shape)
print("=" * 80)

# =========================================================
# 2. 기본 컬럼 정의
ID_COL = "customer_id"
CHURN_TARGET = "target_churn"
LTV_TARGET = "target_ltv"

# =========================================================
# 3. Churn용 feature
churn_features = [
    "days_since_last_purchase",
    "active_last_7d",
    "active_last_30d",
    "active_last_60d",
    "active_last_90d",
    "trans_cnt_7d",
    "trans_cnt_30d",
    "trans_cnt_60d",
    "trans_amount_sum_7d",
    "trans_amount_sum_30d",
    "trans_amount_sum_60d",
    "purchase_interval_mean",
    "purchase_interval_std",
    "monthly_cnt_slope",
    "monthly_amt_slope",
    "online_ratio",
    "installment_ratio",
    "days_since_join",
    "credit_score",
    "fin_overdue_days",
    "total_deposit_balance",
    "total_loan_balance",
    "card_cash_service_amt",
    "card_loan_amt",
    "fin_asset_trend_score",
    "gender",
    "region_code",
    "prefer_category",
    "income_group",
    "age",
    "is_married",
]

# =========================================================
# 4. LTV용 feature
ltv_features = [
    "trans_cnt_total",
    "trans_amount_sum",
    "trans_amount_mean",
    "trans_amount_median",
    "trans_amount_max",
    "trans_amount_std",
    "category_nunique",
    "online_ratio",
    "installment_ratio",
    "high_amt_cnt",
    "high_amt_ratio",
    "amt_q75",
    "amt_q90",
    "active_months",
    "monthly_presence_ratio",
    "monthly_amt_slope",
    "prefer_match_ratio",
    "days_since_join",
    "credit_score",
    "num_active_cards",
    "total_deposit_balance",
    "total_loan_balance",
    "card_cash_service_amt",
    "card_loan_amt",
    "fin_overdue_days",
    "fin_asset_trend_score",
    "gender",
    "region_code",
    "prefer_category",
    "income_group",
    "age",
    "is_married",
]

# =========================================================
# 5. train / test 공통 컬럼만 사용
common_cols = set(train_df.columns).intersection(set(test_df.columns))

churn_features = [col for col in churn_features if col in common_cols]
ltv_features = [col for col in ltv_features if col in common_cols]

print("Churn feature 개수:", len(churn_features))
print("LTV feature 개수  :", len(ltv_features))

# =========================================================
# 6. 범주형 컬럼 정의
cat_cols = ["gender", "region_code", "prefer_category", "income_group"]

churn_cat_cols = [col for col in cat_cols if col in churn_features]
ltv_cat_cols = [col for col in cat_cols if col in ltv_features]

print("Churn 범주형:", churn_cat_cols)
print("LTV 범주형  :", ltv_cat_cols)

# =========================================================
# 7. 학습 / 예측 데이터 분리
X_train_churn = train_df[churn_features].copy()
y_train_churn = train_df[CHURN_TARGET].copy()
X_test_churn = test_df[churn_features].copy()

X_train_ltv = train_df[ltv_features].copy()
y_train_ltv = train_df[LTV_TARGET].copy()
X_test_ltv = test_df[ltv_features].copy()

# =========================================================
# 8. Churn 모델 학습
print("\n" + "=" * 80)
print("[1] Churn CatBoost 학습 시작")
print("=" * 80)

churn_model = CatBoostClassifier(
    loss_function="Logloss",
    eval_metric="AUC",
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=100,
    auto_class_weights="Balanced"
)

churn_model.fit(
    X_train_churn,
    y_train_churn,
    cat_features=churn_cat_cols
)

test_pred_churn = churn_model.predict_proba(X_test_churn)[:, 1]
test_pred_churn = np.clip(test_pred_churn, 0, 1)

print("Churn 예측 완료")

# =========================================================
# 9. LTV 모델 학습 (원본 target_ltv)
print("\n" + "=" * 80)
print("[2] LTV CatBoost 학습 시작 (원본 target_ltv)")
print("=" * 80)

ltv_model = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=400,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=100
)

ltv_model.fit(
    X_train_ltv,
    y_train_ltv,
    cat_features=ltv_cat_cols
)

test_pred_ltv = ltv_model.predict(X_test_ltv)
test_pred_ltv = np.clip(test_pred_ltv, 0, None)

print("LTV 예측 완료")

# =========================================================
# 10. 제출 파일 생성
submission = sample_sub[[ID_COL]].copy()
submission = submission.merge(
    pd.DataFrame({
        ID_COL: test_df[ID_COL],
        "target_churn": test_pred_churn,
        "target_ltv": test_pred_ltv
    }),
    on=ID_COL,
    how="left"
)

# =========================================================
# 11. 최종 점검
print("\n" + "=" * 80)
print("[최종 제출 파일 점검]")
print("=" * 80)
print("shape:", submission.shape)
print(submission.head())

print("\n결측치 개수:")
print(submission.isnull().sum())

print("\nChurn 예측 범위:")
print("min =", submission["target_churn"].min())
print("max =", submission["target_churn"].max())

print("\nLTV 예측 범위:")
print("min =", submission["target_ltv"].min())
print("max =", submission["target_ltv"].max())

print("\ncustomer_id 중복 개수:")
print(submission[ID_COL].duplicated().sum())

# =========================================================
# 12. 저장
submission.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\n저장 완료: {OUTPUT_PATH}")
