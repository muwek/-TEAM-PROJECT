# THE PHYSIQ — Backend

FastAPI 기반 운동 추천 백엔드입니다.

## 시작하기

### 1. Python 가상환경 생성 및 활성화

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 초기 데이터셋 준비 (선택사항)

기존 Kaggle 데이터셋(gym_members_exercise_tracking.csv)이 있다면
아래 경로에 복사하고 파일명을 변경하세요:

```
backend/data/base_dataset.csv
```

컬럼 구성:
- Age, Gender, Weight (kg), Height (m), BMI, Fat_Percentage
- Max_BPM, Avg_BPM, Resting_BPM
- Workout_Type, Session_Duration (hours), Workout_Frequency (days/week), Calories_Burned

없으면 서버가 더미 데이터로 자동 초기화됩니다.

### 4. 서버 실행

```bash
uvicorn main:app --reload --port 8000
```

서버가 뜨면 http://localhost:8000/docs 에서 API 문서를 확인할 수 있습니다.

## API 엔드포인트

| Method | URL | 설명 |
|--------|-----|------|
| POST | /api/recommend/v1 | 기본 운동 추천 |
| POST | /api/recommend/v2 | 목표 칼로리 기반 추천 |
| POST | /api/retrain | 사용자 데이터 누적 + 재학습 |
| GET  | /api/retrain/status | 현재 데이터 수 조회 |

## 생성되는 파일

- `data/model.pkl` — 학습된 모델 저장 파일
- `data/user_logs.csv` — 누적된 사용자 데이터 로그
