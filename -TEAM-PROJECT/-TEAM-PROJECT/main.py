# ==============================================================================
# 1. Colab 파일 업로드 및 데이터 로드
# ==============================================================================
from google.colab import files
import pandas as pd

print("👉 CSV 파일을 업로드해주세요.")
uploaded = files.upload()

# 파일 읽기
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

print("\n데이터 로드 완료! 상위 5개 행:")
display(df.head())

# ==============================================================================
# 2. 사용할 컬럼 선택
# ==============================================================================
df = df[['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
         'Fat_Percentage',
         'Max_BPM', 'Avg_BPM', 'Resting_BPM',
         'Workout_Type', 'Session_Duration (hours)',
         'Workout_Frequency (days/week)', 'Calories_Burned']]

# ==============================================================================
# 3. 데이터 전처리 (인코딩 및 초기 그룹화)
# ==============================================================================
from sklearn.preprocessing import LabelEncoder

# 성별 인코딩
gender_encoder = LabelEncoder()
df['Gender'] = gender_encoder.fit_transform(df['Gender'])

# 운동 종류 인코딩
workout_encoder = LabelEncoder()
df['Workout_Type'] = workout_encoder.fit_transform(df['Workout_Type'])

# 운동 종류별 평균 BPM 계산 (초기값)
bpm_by_workout = df.groupby('Workout_Type')[['Max_BPM', 'Avg_BPM', 'Resting_BPM']].mean()

# ==============================================================================
# 4. 피처(X) 및 타겟(y) 분리
# ==============================================================================
X = df[['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
        'Fat_Percentage', 'Max_BPM', 'Avg_BPM', 'Resting_BPM']]

y_type = df['Workout_Type']
y_duration = df['Session_Duration (hours)']
y_freq = df['Workout_Frequency (days/week)']
y_cal = df['Calories_Burned']

# ==============================================================================
# 5. 머신러닝 모델 초기 생성 및 학습
# ==============================================================================
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 운동 종류 분류 모델
clf = RandomForestClassifier()
clf.fit(X, y_type)

# 운동 시간 예측 모델
reg_duration = RandomForestRegressor()
reg_duration.fit(X, y_duration)

# 운동 빈도 예측 모델
reg_freq = RandomForestRegressor()
reg_freq.fit(X, y_freq)

# 칼로리 예측 모델
reg_cal = RandomForestRegressor()
reg_cal.fit(X, y_cal)

# ==============================================================================
# 6. 추천 알고리즘 V1 (기본 추천)
# ==============================================================================
def recommend_exercise_v1(age, gender, weight, height, bmi, fat):
    # 입력받은 성별 문자열을 인코딩된 숫자로 변환
    gender_encoded = gender_encoder.transform([gender])[0]

    # 🔹 1단계: BPM 정보 없이 운동 종류 선행 예측
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

    # 🔹 2단계: 예측된 운동 종류의 통계적 평균 BPM 가져오기
    bpm_values = bpm_by_workout.loc[pred_type_encoded]

    # 🔹 3단계: 평균 BPM을 결합한 최종 입력 데이터 생성
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

    # 🔹 최종 예측 수행
    duration = float(reg_duration.predict(final_input)[0])
    freq = float(reg_freq.predict(final_input)[0])
    calories = float(reg_cal.predict(final_input)[0])

    return {
        "운동종류": workout_type,
        "운동시간": round(duration, 2),
        "주당횟수": round(freq, 1),
        "칼로리": round(calories, 1)
    }

# ==============================================================================
# 7. 추천 알고리즘 V2 (목표 칼로리 기반 시간 조정)
# ==============================================================================
def recommend_exercise_v2(age, gender, weight, height, bmi, fat, target_calories):
    # V1 기본 추천 결과 획득
    base = recommend_exercise_v1(age, gender, weight, height, bmi, fat)

    base_cal = base['칼로리']
    base_time = base['운동시간']

    # 예외 처리 및 목표 칼로리에 비례한 운동시간 재계산
    if base_cal == 0:
        adjusted_time = base_time
    else:
        adjusted_time = base_time * (target_calories / base_cal)

    return {
        "운동종류": base['운동종류'],
        "추천운동시간": round(adjusted_time, 2),
        "주당횟수": base['주당횟수']
    }

# ==============================================================================
# 8. 백엔드 로직 테스트 (정상 작동 확인용)
# ==============================================================================
print("\n--- [테스트] 함수 작동 여부 확인 ---")
print("V1 결과:", recommend_exercise_v1(25, 'Male', 70, 1.75, 22, 18))
print("V2 결과:", recommend_exercise_v2(25, 'Male', 70, 1.75, 22, 18, 500))
print("----------------------------------\n")

# ==============================================================================
# 9. 🔥 [교수님 피드백 반영] 사용자 데이터 누적 및 실시간 재학습 UI 시스템
# ==============================================================================
def recommend_exercise_v3_updated():
    import ipywidgets as widgets
    from IPython.display import display, clear_output

    # 전역 변수(데이터프레임 및 모델들)를 함수 내부에서 업데이트하기 위해 global 선언
    global df, clf, reg_duration, reg_freq, reg_cal, gender_encoder, workout_encoder, bpm_by_workout

    output = widgets.Output()

    # UI 레이아웃 설정
    label_layout = widgets.Layout(width='120px')
    input_layout = widgets.Layout(width='200px')

    def create_input(label, widget):
        return widgets.HBox([
            widgets.Label(label, layout=label_layout),
            widget
        ])

    # 위젯 세팅 및 초기값 설정
    age = widgets.IntText(value=25, layout=input_layout)
    gender = widgets.Dropdown(options=['Male','Female'], value='Male', layout=input_layout)
    weight = widgets.FloatText(value=70.0, layout=input_layout)
    height = widgets.FloatText(value=1.75, layout=input_layout)
    bmi = widgets.FloatText(value=22.0, layout=input_layout)
    fat = widgets.FloatText(value=18.0, layout=input_layout)
    cal = widgets.FloatText(value=500.0, layout=input_layout)

    # UI 구조 생성
    age_box = create_input("Age", age)
    gender_box = create_input("Gender", gender)
    weight_box = create_input("Weight (kg)", weight)
    height_box = create_input("Height (m)", height)
    bmi_box = create_input("BMI", bmi)
    fat_box = create_input("Fat%", fat)
    cal_box = create_input("TargetCal", cal)

    # 기본 모드에서는 목표 칼로리 입력창 숨김
    cal_box.layout.display = 'none'

    # 모드 선택 버튼
    version = widgets.ToggleButtons(
        options=['v1 기본 추천', 'v2 목표 칼로리'],
        description='Mode'
    )

    # 데이터 실시간 누적 확인용 라벨
    status_label = widgets.Label(value=f"현재 학습된 데이터 개수: {len(df)}건")

    # 인터랙티브 토글 이벤트 정의
    def on_change(change):
        if change['new'] == 'v2 목표 칼로리':
            cal_box.layout.display = 'flex'
        else:
            cal_box.layout.display = 'none'

    version.observe(on_change, names='value')

    # 데이터 업그레이드 전용 실행 버튼
    run_btn = widgets.Button(description="추천 및 데이터 반영", button_style='success', layout=widgets.Layout(width='324px'))

    # 버튼 클릭 시 실행될 피드백 루프 핵심 로직
    def on_click(b):
        global df, clf, reg_duration, reg_freq, reg_cal, bpm_by_workout
        with output:
            clear_output()
            try:
                # [단계 1] 선택된 모드에 따라 추천 실행 및 결과 추출
                if version.value == 'v1 기본 추천':
                    result = recommend_exercise_v1(
                        age.value, gender.value, weight.value, 
                        height.value, bmi.value, fat.value
                    )
                    final_calories = result['칼로리']
                    final_duration = result['운동시간']
                else:
                    result = recommend_exercise_v2(
                        age.value, gender.value, weight.value, 
                        height.value, bmi.value, fat.value, cal.value
                    )
                    final_calories = cal.value
                    final_duration = result['추천운동시간']

                # 추천 결과 시각화
                print("===== ✨ 추천 결과 =====")
                for k, v in result.items():
                    print(f"{k} : {v}")
                print("=======================\n")

                # [단계 2] 인코딩 역산 및 새 데이터 Row 생성
                g_encoded = gender_encoder.transform([gender.value])[0]
                w_encoded = workout_encoder.transform([result['운동종류']])[0]
                bpm_values = bpm_by_workout.loc[w_encoded] # 예측된 운동의 평균 BPM 매핑

                new_data = pd.DataFrame([{
                    'Age': age.value,
                    'Gender': g_encoded,
                    'Weight (kg)': weight.value,
                    'Height (m)': height.value,
                    'BMI': bmi.value,
                    'Fat_Percentage': fat.value,
                    'Max_BPM': bpm_values['Max_BPM'],
                    'Avg_BPM': bpm_values['Avg_BPM'],
                    'Resting_BPM': bpm_values['Resting_BPM'],
                    'Workout_Type': w_encoded,
                    'Session_Duration (hours)': final_duration,
                    'Workout_Frequency (days/week)': result['주당횟수'],
                    'Calories_Burned': final_calories
                }])

                # [단계 3] 기존 데이터프레임(df)에 유저 정보 누적 업데이트
                df = pd.concat([df, new_data], ignore_index=True)
                
                # [단계 4] 새 데이터가 업데이트된 데이터셋으로 전체 모델 실시간 재학습
                X_new = df[['Age', 'Gender', 'Weight (kg)', 'Height (m)', 'BMI',
                            'Fat_Percentage', 'Max_BPM', 'Avg_BPM', 'Resting_BPM']]
                
                clf.fit(X_new, df['Workout_Type'])
                reg_duration.fit(X_new, df['Session_Duration (hours)'])
                reg_freq.fit(X_new, df['Workout_Frequency (days/week)'])
                reg_cal.fit(X_new, df['Calories_Burned'])
                
                # 운동 종류별 평균 BPM 통계치 최신화
                bpm_by_workout = df.groupby('Workout_Type')[['Max_BPM', 'Avg_BPM', 'Resting_BPM']].mean()

                # 피드백 반영 완료 시각화
                status_label.value = f"✅ 데이터 반영 및 재학습 완료! (총 데이터: {len(df)}건)"
                print(f"📢 [시스템 통림] 새로운 사용자의 입력과 결과가 데이터셋에 저장되었으며, 모델이 실시간으로 재학습되었습니다.")

            except Exception as e:
                print("오류 발생:", e)

    run_btn.on_click(on_click)

    # UI 컴포넌트 결합 및 화면 표시
    ui = widgets.VBox([
        version,
        age_box, gender_box, weight_box, height_box, bmi_box, fat_box, cal_box,
        status_label,
        run_btn,
        output
    ])

    display(ui)

# ==============================================================================
# 10. 인터페이스 실행
# ==============================================================================
recommend_exercise_v3_updated()