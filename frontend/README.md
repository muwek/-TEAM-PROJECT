# THE PHYSIQ — Frontend

React + Vite + TypeScript + MUI 기반 프론트엔드입니다.

## 시작하기

```bash
cd frontend
npm install
npm run dev
```

브라우저에서 http://localhost:5173 접속

## 구조

```
src/
├── App.tsx                   # 라우팅만 담당 (최상위 진입점)
├── main.tsx
├── pages/
│   └── HomePage.tsx          # 메인 페이지 (About / Recommender / Developer Info)
├── components/
│   ├── Hero/                 # 히어로 섹션 (THE PHYSIQ 타이틀)
│   ├── RecommendForm/        # 입력 폼 (V1 / V2 모드)
│   ├── ResultCard/           # 추천 결과 카드
│   ├── ConsentDialog/        # 개인정보 동의 다이얼로그
│   ├── ThemeProvider/        # 라이트/다크 테마 컨텍스트
│   ├── ThemeToggle/          # 테마 토글 버튼
│   └── Layout/               # AnimatedSection 등 레이아웃 유틸
├── hooks/
│   └── useRecommend.ts       # 추천 로직 + API 통신 훅
├── services/
│   └── api.ts                # 백엔드 API 호출 함수 모음
├── types/
│   └── index.ts              # TypeScript 타입 정의
└── styles/
    └── index.css
```

## 백엔드 연결

`vite.config.ts`에서 `/api` 경로를 `http://localhost:8000`으로 프록시합니다.
백엔드가 실행 중이어야 추천 기능이 동작합니다.
