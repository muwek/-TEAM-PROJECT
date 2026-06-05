"""
ML 모델 관리자
Colab 노트북의 학습 로직을 그대로 이식한 파일입니다.
- RandomForestClassifier: 운동 종류 예측
- RandomForestRegressor x3: 운동 시간 / 주당 빈도 / 칼로리 예측
"""
import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(DATA_DIR, "model.pkl")
USER_LOG_PATH = os.path.join(DATA_DIR, "user_logs.csv")
BASE_DATA_PATH = os.path.join(DATA_DIR, "base_dataset.csv")

FEATURE_COLS = [
    "Age", "Gender", "Weight (kg)", "Height (m)", "BMI",
    "Fat_Percentage", "Max_BPM", "Avg_BPM", "Resting_BPM"
]
ALL_COLS = FEATURE_COLS + [
    "Workout_Type", "Session_Duration (hours)",
    "Workout_Frequency (days/week)", "Calories_Burned"
]


class ModelManager:
    """
    싱글턴 패턴으로 모델 상태를 유지합니다.
    서버 시작 시 한 번 load()를 호출하면 이후 predict/retrain이 가능합니다.
    """
    clf = None
    reg_duration = None
    reg_freq = None
    reg_cal = None
    gender_encoder: LabelEncoder = None
    workout_encoder: LabelEncoder = None
    bpm_by_workout: pd.DataFrame = None
    df: pd.DataFrame = None
    _loaded = False

    @classmethod
    def load(cls):
        """저장된 model.pkl 로드, 없으면 base_dataset.csv로 초기 학습"""
        os.makedirs(DATA_DIR, exist_ok=True)

        if os.path.exists(MODEL_PATH):
            bundle = joblib.load(MODEL_PATH)
            cls.clf = bundle["clf"]
            cls.reg_duration = bundle["reg_duration"]
            cls.reg_freq = bundle["reg_freq"]
            cls.reg_cal = bundle["reg_cal"]
            cls.gender_encoder = bundle["gender_encoder"]
            cls.workout_encoder = bundle["workout_encoder"]
            cls.bpm_by_workout = bundle["bpm_by_workout"]
            cls.df = bundle["df"]
            print(f"✅ model.pkl 로드 완료 (데이터 {len(cls.df)}건)")
        elif os.path.exists(BASE_DATA_PATH):
            print("⚙️  base_dataset.csv로 초기 학습 시작...")
            cls._train_from_csv(BASE_DATA_PATH)
        else:
            print("⚠️  base_dataset.csv 없음 — 더미 데이터로 초기화")
            cls._init_with_dummy()

        cls._loaded = True

    @classmethod
    def _train_from_csv(cls, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df[ALL_COLS].dropna()

        cls.gender_encoder = LabelEncoder()
        df["Gender"] = cls.gender_encoder.fit_transform(df["Gender"])

        cls.workout_encoder = LabelEncoder()
        df["Workout_Type"] = cls.workout_encoder.fit_transform(df["Workout_Type"])

        cls.df = df
        cls.bpm_by_workout = df.groupby("Workout_Type")[
            ["Max_BPM", "Avg_BPM", "Resting_BPM"]
        ].mean()

        cls._fit_all()
        cls._save()
        print(f"✅ 초기 학습 완료 ({len(df)}건)")

    @classmethod
    def _init_with_dummy(cls):
        """base_dataset.csv가 없을 때 서버가 뜰 수 있도록 최소 더미 초기화"""
        cls.gender_encoder = LabelEncoder()
        cls.gender_encoder.fit(["Male", "Female"])
        cls.workout_encoder = LabelEncoder()
        cls.workout_encoder.fit(["Cardio", "HIIT", "Strength", "Yoga"])

        dummy = pd.DataFrame([
            [25, 0, 70, 1.75, 22.9, 18, 160, 130, 60, 0, 1.0, 3, 550],
            [30, 1, 60, 1.65, 22.0, 22, 150, 120, 65, 1, 1.2, 4, 480],
            [40, 0, 85, 1.80, 26.2, 28, 155, 125, 70, 2, 1.5, 5, 700],
            [22, 1, 55, 1.60, 21.5, 19, 170, 140, 58, 3, 0.8, 3, 400],
        ], columns=ALL_COLS)

        cls.df = dummy
        cls.bpm_by_workout = dummy.groupby("Workout_Type")[
            ["Max_BPM", "Avg_BPM", "Resting_BPM"]
        ].mean()
        cls._fit_all()

    @classmethod
    def _fit_all(cls):
        X = cls.df[FEATURE_COLS]
        cls.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        cls.clf.fit(X, cls.df["Workout_Type"])

        cls.reg_duration = RandomForestRegressor(n_estimators=100, random_state=42)
        cls.reg_duration.fit(X, cls.df["Session_Duration (hours)"])

        cls.reg_freq = RandomForestRegressor(n_estimators=100, random_state=42)
        cls.reg_freq.fit(X, cls.df["Workout_Frequency (days/week)"])

        cls.reg_cal = RandomForestRegressor(n_estimators=100, random_state=42)
        cls.reg_cal.fit(X, cls.df["Calories_Burned"])

    @classmethod
    def _save(cls):
        bundle = {
            "clf": cls.clf,
            "reg_duration": cls.reg_duration,
            "reg_freq": cls.reg_freq,
            "reg_cal": cls.reg_cal,
            "gender_encoder": cls.gender_encoder,
            "workout_encoder": cls.workout_encoder,
            "bpm_by_workout": cls.bpm_by_workout,
            "df": cls.df,
        }
        joblib.dump(bundle, MODEL_PATH)
        print(f"💾 model.pkl 저장 완료 ({len(cls.df)}건)")

    @classmethod
    def predict_v1(cls, age, gender, weight, height, bmi, fat) -> dict:
        """V1: 기본 추천 — Colab recommend_exercise_v1() 이식"""
        if not cls._loaded:
            raise RuntimeError("모델이 로드되지 않았습니다. ModelManager.load()를 먼저 호출하세요.")

        gender_encoded = cls.gender_encoder.transform([gender])[0]

        # 1단계: BPM 없이 운동 종류 선행 예측
        temp_input = pd.DataFrame([{
            "Age": age, "Gender": gender_encoded,
            "Weight (kg)": weight, "Height (m)": height,
            "BMI": bmi, "Fat_Percentage": fat,
            "Max_BPM": 0, "Avg_BPM": 0, "Resting_BPM": 0,
        }])
        pred_type_encoded = cls.clf.predict(temp_input)[0]
        workout_type = cls.workout_encoder.inverse_transform([pred_type_encoded])[0]

        # 2단계: 예측된 운동 종류의 통계적 평균 BPM 가져오기
        bpm_values = cls.bpm_by_workout.loc[pred_type_encoded]

        # 3단계: 평균 BPM을 결합한 최종 예측
        final_input = pd.DataFrame([{
            "Age": age, "Gender": gender_encoded,
            "Weight (kg)": weight, "Height (m)": height,
            "BMI": bmi, "Fat_Percentage": fat,
            "Max_BPM": float(bpm_values["Max_BPM"]),
            "Avg_BPM": float(bpm_values["Avg_BPM"]),
            "Resting_BPM": float(bpm_values["Resting_BPM"]),
        }])

        duration = float(cls.reg_duration.predict(final_input)[0])
        freq = float(cls.reg_freq.predict(final_input)[0])
        calories = float(cls.reg_cal.predict(final_input)[0])

        return {
            "workoutType": workout_type,
            "sessionDuration": round(duration, 2),
            "frequency": round(freq, 1),
            "calories": round(calories, 1),
        }

    @classmethod
    def predict_v2(cls, age, gender, weight, height, bmi, fat, target_calories) -> dict:
        """V2: 목표 칼로리 기반 시간 조정 — Colab recommend_exercise_v2() 이식"""
        base = cls.predict_v1(age, gender, weight, height, bmi, fat)

        base_cal = base["calories"]
        base_time = base["sessionDuration"]

        adjusted_time = (
            base_time * (target_calories / base_cal)
            if base_cal != 0 else base_time
        )

        return {
            "workoutType": base["workoutType"],
            "sessionDuration": round(adjusted_time, 2),
            "frequency": base["frequency"],
            "calories": target_calories,
        }

    @classmethod
    def add_user_data_and_retrain(cls, row: dict):
        """
        사용자 데이터를 누적하고 모델을 즉시 재학습합니다.
        Colab의 on_click() 단계 2~4를 이식한 함수입니다.
        """
        gender_encoded = cls.gender_encoder.transform([row["gender"]])[0]
        workout_encoded = cls.workout_encoder.transform([row["workoutType"]])[0]
        bpm_values = cls.bpm_by_workout.loc[workout_encoded]

        new_row = pd.DataFrame([{
            "Age": row["age"],
            "Gender": gender_encoded,
            "Weight (kg)": row["weight"],
            "Height (m)": row["height"],
            "BMI": row["bmi"],
            "Fat_Percentage": row["fatPercentage"],
            "Max_BPM": float(bpm_values["Max_BPM"]),
            "Avg_BPM": float(bpm_values["Avg_BPM"]),
            "Resting_BPM": float(bpm_values["Resting_BPM"]),
            "Workout_Type": workout_encoded,
            "Session_Duration (hours)": row["sessionDuration"],
            "Workout_Frequency (days/week)": row["frequency"],
            "Calories_Burned": row["calories"],
        }])

        # 누적 업데이트
        cls.df = pd.concat([cls.df, new_row], ignore_index=True)

        # 재학습
        cls._fit_all()

        # BPM 통계치 최신화
        cls.bpm_by_workout = cls.df.groupby("Workout_Type")[
            ["Max_BPM", "Avg_BPM", "Resting_BPM"]
        ].mean()

        # user_logs.csv에도 저장 (원시 데이터 보존)
        log_row = pd.DataFrame([{
            "age": row["age"], "gender": row["gender"],
            "weight": row["weight"], "height": row["height"],
            "bmi": row["bmi"], "fat_percentage": row["fatPercentage"],
            "workout_type": row["workoutType"],
            "session_duration": row["sessionDuration"],
            "frequency": row["frequency"],
            "calories": row["calories"],
        }])
        header = not os.path.exists(USER_LOG_PATH)
        log_row.to_csv(USER_LOG_PATH, mode="a", header=header, index=False)

        # 모델 파일 저장
        cls._save()

        return {"totalDataCount": len(cls.df)}
