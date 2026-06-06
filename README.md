<div align="center">

# 🏋️ THE PHYSIQ

### AI-Powered Personalized Exercise Recommendation System

[![Frontend](https://img.shields.io/badge/Frontend-Vercel-black?style=for-the-badge&logo=vercel)](https://thephysiq.vercel.app)
[![Backend](https://img.shields.io/badge/Backend-Railway-purple?style=for-the-badge&logo=railway)](https://the-physiq.up.railway.app)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python)](https://python.org)

<br/>

> 신체 데이터를 입력하면 RandomForest 머신러닝 알고리즘이  
> 최적의 운동 종류, 시간, 빈도, 소모 칼로리를 예측합니다.  
> 사용자 데이터가 누적될수록 모델이 스스로 개선됩니다.

<br/>

**🌐 [thephysiq.vercel.app](https://thephysiq.vercel.app)**

</div>

---

## 📸 주요 기능

| 기능 | 설명 |
|------|------|
| 🤖 **AI 운동 추천** | 나이, 성별, 체중, 신장, BMI, 체지방률 기반 맞춤 추천 |
| 🎯 **목표 칼로리 모드** | 목표 칼로리를 입력하면 운동 시간 자동 조정 |
| 📊 **실시간 재학습** | 사용자 데이터 동의 시 모델에 즉시 반영 |
| 🌙 **다크 / 라이트 모드** | 시스템 설정 연동 + 수동 전환 |
| 📱 **반응형 디자인** | 모바일 / 태블릿 / 데스크톱 모두 지원 |

---

## 🏗️ 기술 스택

### Frontend
- **React 18** + **TypeScript** + **Vite**
- **MUI (Material UI)** — UI 컴포넌트
- **Motion** — 애니메이션
- **Vercel** — 배포

### Backend
- **FastAPI** — REST API 서버
- **scikit-learn** — RandomForest 모델
- **pandas** — 데이터 처리
- **joblib** — 모델 직렬화
- **Railway** — 배포

---

## 🔄 데이터 흐름

```
사용자 신체 데이터 입력
        ↓
   개인정보 동의 확인
        ↓
  POST /api/recommend/v1 or v2
        ↓
  RandomForest 모델 추론
        ↓
  운동 종류 / 시간 / 빈도 / 칼로리 반환
        ↓
  결과 화면 표시
        ↓
  POST /api/retrain (백그라운드)
        ↓
  user_logs.csv 누적 → 모델 재학습 → model.pkl 저장
```

---

## 📁 프로젝트 구조

```
-TEAM-PROJECT/
├── frontend/                        # React 프론트엔드
│   ├── src/
│   │   ├── App.tsx                  # 라우팅 진입점
│   │   ├── pages/
│   │   │   └── HomePage.tsx         # 메인 페이지
│   │   ├── components/
│   │   │   ├── Hero/                # 히어로 섹션
│   │   │   ├── RecommendForm/       # 입력 폼 (V1 / V2)
│   │   │   ├── ResultCard/          # 추천 결과 카드
│   │   │   ├── ConsentDialog/       # 개인정보 동의
│   │   │   ├── ThemeProvider/       # 테마 컨텍스트
│   │   │   └── ThemeToggle/         # 테마 토글 버튼
│   │   ├── hooks/
│   │   │   └── useRecommend.ts      # 추천 로직 훅
│   │   ├── services/
│   │   │   └── api.ts               # 백엔드 API 통신
│   │   └── types/
│   │       └── index.ts             # TypeScript 타입 정의
│   └── package.json
│
└── backend/                         # FastAPI 백엔드
    ├── main.py                      # 서버 진입점
    ├── app/
    │   ├── routes/
    │   │   ├── recommend.py         # 추천 API 엔드포인트
    │   │   └── retrain.py           # 재학습 API 엔드포인트
    │   ├── ml/
    │   │   └── model.py             # RandomForest 모델 관리
    │   └── models/
    │       └── schemas.py           # Pydantic 스키마
    ├── data/
    │   ├── base_dataset.csv         # 초기 학습 데이터
    │   ├── model.pkl                # 저장된 모델
    │   └── user_logs.csv            # 누적 사용자 데이터
    └── requirements.txt
```

---

## 🚀 로컬 실행 방법

### 사전 요구사항
- Python 3.10 이상
- Node.js 18 이상

### 백엔드 실행 (터미널 1)

```bash
cd backend

# 가상환경 생성 및 활성화 (최초 1회)
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 의존성 설치 (최초 1회)
pip install -r requirements.txt

# 서버 실행
uvicorn main:app --reload --port 8000
```

### 프론트엔드 실행 (터미널 2)

```bash
cd frontend

# 의존성 설치 (최초 1회)
npm install

# 개발 서버 실행
npm run dev
```

브라우저에서 **http://localhost:5173** 접속

---

## 🌐 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/api/recommend/v1` | 기본 운동 추천 |
| `POST` | `/api/recommend/v2` | 목표 칼로리 기반 추천 |
| `POST` | `/api/retrain` | 사용자 데이터 누적 + 재학습 |
| `GET` | `/api/retrain/status` | 현재 학습 데이터 수 조회 |

API 문서: **[the-physiq.up.railway.app/docs](https://the-physiq.up.railway.app/docs)**

---

## 📊 ML 모델 정보

- **알고리즘**: RandomForestClassifier + RandomForestRegressor × 3
- **입력 변수**: Age, Gender, Weight, Height, BMI, Fat%, Max/Avg/Resting BPM
- **출력 변수**: Workout Type, Session Duration, Frequency, Calories Burned
- **학습 데이터**: [Kaggle — Gym Members Exercise Tracking](https://www.kaggle.com/datasets/valakhorasani/gym-members-exercise-dataset)

---

## 👨‍💻 개발팀

<div align="center">

**TEAM A**

김도현 · 김민성 · 김준영 · 노채영 · 이민용 · 이형균

[![GitHub](https://img.shields.io/badge/GitHub-muwek%2F--TEAM--PROJECT-181717?style=flat-square&logo=github)](https://github.com/muwek/-TEAM-PROJECT)
[![Colab](https://img.shields.io/badge/Colab-ML%20Notebook-F9AB00?style=flat-square&logo=googlecolab)](https://colab.research.google.com/drive/1qafBSjUzeVtGbZGO56H0PqJ07wgGzMPy?usp=sharing)

<br/>

THE PHYSIQ © 2026

</div>
