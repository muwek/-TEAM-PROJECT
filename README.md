# THE PHYSIQ — Full Stack Project

AI 기반 개인 맞춤 운동 추천 웹 서비스입니다.

```
physiq_full/
├── frontend/    # React + Vite + TypeScript + MUI
└── backend/     # Python FastAPI + scikit-learn (RandomForest)
```

---

## 빠른 시작 (두 터미널 필요)

### 터미널 1 — 백엔드

```bash
cd backend

# 가상환경 생성 (최초 1회)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 의존성 설치 (최초 1회)
pip install -r requirements.txt

# (선택) 초기 데이터셋 배치
# Kaggle gym 데이터셋을 data/base_dataset.csv 로 복사

# 서버 실행
uvicorn main:app --reload --port 8000
```

### 터미널 2 — 프론트엔드

```bash
cd frontend

# 의존성 설치 (최초 1회)
npm install

# 개발 서버 실행
npm run dev
```

브라우저에서 **http://localhost:5173** 접속

---

## 데이터 흐름

```
사용자 입력
    ↓
프론트엔드 (React)
    ↓  POST /api/recommend/v1 or v2
백엔드 (FastAPI)
    ↓  ModelManager.predict_v1/v2()
RandomForest 모델 추론
    ↓  결과 반환
프론트엔드 — 결과 표시
    ↓  POST /api/retrain (동의한 경우)
백엔드 — 사용자 데이터 누적
    ↓
model.pkl + user_logs.csv 업데이트
    ↓
전체 모델 즉시 재학습
```

---

## 초기 데이터셋 안내

`backend/data/base_dataset.csv` 파일이 있으면 서버 시작 시 자동으로 학습합니다.

컬럼 형식:
```
Age, Gender, Weight (kg), Height (m), BMI, Fat_Percentage,
Max_BPM, Avg_BPM, Resting_BPM,
Workout_Type, Session_Duration (hours), Workout_Frequency (days/week), Calories_Burned
```

없으면 더미 4건으로 자동 초기화됩니다.
실제 서비스를 위해서는 Kaggle의 **gym_members_exercise_tracking** 데이터셋 사용을 권장합니다.

---

## API 문서

백엔드 실행 후 http://localhost:8000/docs 에서 Swagger UI 확인 가능
