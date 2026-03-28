# 1. 드라이브 연결
from google.colab import drive
drive.mount('/content/drive')

# 2. 라이브러리
import pandas as pd

# 3. 데이터 불러오기
df = pd.read_csv("/content/drive/MyDrive/gym_members_exercise_tracking.csv")

# 4. 데이터 확인
print(df.head())

# 5. 결측값 제거
df = df.dropna()

# 6. Gender 처리 (에러 방지 포함)
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].astype(str)
    df = pd.get_dummies(df, columns=["Gender"], drop_first=True)

# 7. User_ID 제거 (있으면)
df = df.drop(["User_ID"], axis=1, errors='ignore')

# 8. 0 나누기 방지
df = df[df["Calories_Burned"] != 0]

# 9. 기간 컬럼 생성
df["Estimated_Days"] = (df["Fat_Percentage"] / df["Calories_Burned"]) * 100

# =========================
# 운동 추천 모델 (분류)
# =========================

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_cls = df.drop(["Workout_Type", "Estimated_Days"], axis=1)
y_cls = df["Workout_Type"]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_c, y_train_c)

pred_c = clf.predict(X_test_c)
print("운동 추천 Accuracy:", accuracy_score(y_test_c, pred_c))

# =========================
# 기간 예측 모델 (회귀)
# =========================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

X_reg = df.drop(["Estimated_Days", "Workout_Type"], axis=1)
y_reg = df["Estimated_Days"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg = RandomForestRegressor(random_state=42)
reg.fit(X_train_r, y_train_r)

pred_r = reg.predict(X_test_r)

print("MSE:", mean_squared_error(y_test_r, pred_r))
print("R2:", r2_score(y_test_r, pred_r))

# =========================
# 최종 테스트
# =========================

sample = X_cls.iloc[[0]]

print("추천 운동:", clf.predict(sample))
print("예상 기간:", reg.predict(sample))

# 디테일한 출력
sample = X_cls.iloc[[0]]

print("입력 데이터:")
print(sample)

print("\n추천 운동:", clf.predict(sample))
print("예상 기간:", reg.predict(sample))

# 그래프 활용 시각적 요소 더하기
import matplotlib.pyplot as plt

importances = clf.feature_importances_
features = X_cls.columns

plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()