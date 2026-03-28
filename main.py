# 1 ===================================
# Colab 파일 업로드
from google.colab import files
uploaded = files.upload()

import pandas as pd

# 파일 읽기
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

df.head()

# 2 ===================================
df = df[['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
         'Fat_Percentage',
         'Max_BPM', 'Avg_BPM', 'Resting_BPM',
         'Workout_Type', 'Session_Duration (hours)',
         'Workout_Frequency (days/week)', 'Calories_Burned']]

# 3 ===================================
from sklearn.preprocessing import LabelEncoder

# 성별 인코딩
gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

# 운동 종류 인코딩
workout_encoder = LabelEncoder()
df['Workout_Type'] = workout_encoder.fit_transform(df['Workout_Type'])

# 운동 종류별 평균 BPM 계산
bpm_by_workout = df.groupby('Workout_Type')[['Max_BPM', 'Avg_BPM', 'Resting_BPM']].mean()

# 4 ===================================
X = df[['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
        'Fat_Percentage',
        'Max_BPM', 'Avg_BPM', 'Resting_BPM']]

y_type = df['Workout_Type']
y_duration = df['Session_Duration (hours)']
y_freq = df['Workout_Frequency (days/week)']
y_cal = df['Calories_Burned']

# 5 ===================================
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 운동 종류 분류 모델
clf = RandomForestClassifier()
clf.fit(X, y_type)

# 운동 시간
reg_duration = RandomForestRegressor()
reg_duration.fit(X, y_duration)

# 운동 빈도
reg_freq = RandomForestRegressor()
reg_freq.fit(X, y_freq)

# 칼로리
reg_cal = RandomForestRegressor()
reg_cal.fit(X, y_cal)

# 6 ===================================
def recommend_exercise_v1(age, gender, weight, height, bmi, fat):

    gender_encoded = gender_encoder.transform([gender])[0]

    # 🔹 1단계: BPM 없이 운동 종류 예측
    temp_input = pd.DataFrame([{
        'Age': age,
        'Gender': gender_encoded,
        'Weight (kg)': weight,
        'Height (m)': height,
        'BMI': bmi,
        'Fat_Percentage': fat,
        'Max_BPM': 0,
        'Avg_BPM': 0,
        'Resting_BPM': 0
    }])

    pred_type_encoded = clf.predict(temp_input)[0]
    workout_type = workout_encoder.inverse_transform([pred_type_encoded])[0]

    # 🔹 2단계: 해당 운동의 평균 BPM 가져오기
    bpm_values = bpm_by_workout.loc[pred_type_encoded]

    # 🔹 3단계: 최종 입력
    final_input = pd.DataFrame([{
        'Age': age,
        'Gender': gender_encoded,
        'Weight (kg)': weight,
        'Height (m)': height,
        'BMI': bmi,
        'Fat_Percentage': fat,
        'Max_BPM': bpm_values['Max_BPM'],
        'Avg_BPM': bpm_values['Avg_BPM'],
        'Resting_BPM': bpm_values['Resting_BPM']
    }])

    # 🔹 최종 예측
    duration = float(reg_duration.predict(final_input)[0])
    freq = float(reg_freq.predict(final_input)[0])
    calories = float(reg_cal.predict(final_input)[0])

    return {
        "운동종류": workout_type,
        "운동시간": round(duration, 2),
        "주당횟수": round(freq, 1),
        "칼로리": round(calories, 1)
    }

# 7 ===================================
def recommend_exercise_v2(age, gender, weight, height, bmi, fat, target_calories):

    base = recommend_exercise_v1(age, gender, weight, height, bmi, fat)

    base_cal = base['칼로리']
    base_time = base['운동시간']

    if base_cal == 0:
        adjusted_time = base_time
    else:
        adjusted_time = base_time * (target_calories / base_cal)

    return {
        "운동종류": base['운동종류'],
        "추천운동시간": round(adjusted_time, 2),
        "주당횟수": base['주당횟수']
    }

# 8 ===================================
recommend_exercise_v1(25, 'Male', 70, 1.75, 22, 18)
recommend_exercise_v2(25, 'Male', 70, 1.75, 22, 18, 500)

# 9 ===================================
def recommend_exercise_v3():

    import ipywidgets as widgets
    from IPython.display import display, clear_output

    output = widgets.Output()

    # 👉 공통 레이아웃
    label_layout = widgets.Layout(width='120px')
    input_layout = widgets.Layout(width='200px')

    # 👉 입력 생성 함수 (핵심)
    def create_input(label, widget):
        return widgets.HBox([
            widgets.Label(label, layout=label_layout),
            widget
        ])

    # 입력 필드
    age = widgets.IntText(layout=input_layout)
    gender = widgets.Dropdown(options=['Male','Female'], layout=input_layout)
    weight = widgets.FloatText(layout=input_layout)
    height = widgets.FloatText(layout=input_layout)
    bmi = widgets.FloatText(layout=input_layout)
    fat = widgets.FloatText(layout=input_layout)
    cal = widgets.FloatText(layout=input_layout)

    # UI 묶기
    age_box = create_input("Age", age)
    gender_box = create_input("Gender", gender)
    weight_box = create_input("Weight", weight)
    height_box = create_input("Height", height)
    bmi_box = create_input("BMI", bmi)
    fat_box = create_input("Fat%", fat)
    cal_box = create_input("TargetCal", cal)

    # 처음엔 숨김
    cal_box.layout.display = 'none'

    # 버전 선택
    version = widgets.ToggleButtons(
        options=['v1 기본 추천', 'v2 목표 칼로리'],
        description='Mode'
    )

    # 버전 변경 이벤트
    def on_change(change):
        if change['new'] == 'v2 목표 칼로리':
            cal_box.layout.display = 'flex'
        else:
            cal_box.layout.display = 'none'

    version.observe(on_change, names='value')

    # 실행 버튼
    run_btn = widgets.Button(description="추천 실행", button_style='success')

    def on_click(b):
        with output:
            clear_output()
            try:
                if version.value == 'v1 기본 추천':
                    result = recommend_exercise_v1(
                        age.value, gender.value,
                        weight.value, height.value,
                        bmi.value, fat.value
                    )
                else:
                    result = recommend_exercise_v2(
                        age.value, gender.value,
                        weight.value, height.value,
                        bmi.value, fat.value,
                        cal.value
                    )

                print("===== 추천 결과 =====")
                for k, v in result.items():
                    print(f"{k} : {v}")

            except Exception as e:
                print("오류:", e)

    run_btn.on_click(on_click)

    # 전체 UI
    ui = widgets.VBox([
        version,
        age_box,
        gender_box,
        weight_box,
        height_box,
        bmi_box,
        fat_box,
        cal_box,
        run_btn,
        output
    ])

    display(ui)

# 10 ===================================
# v1(25, 'Male', 70, 1.75, 22, 18)
recommend_exercise_v3()

# v2(25, 'Male', 70, 1.75, 22, 18, 500)
recommend_exercise_v3()