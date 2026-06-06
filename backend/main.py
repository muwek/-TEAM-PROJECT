from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import recommend, retrain
from app.ml.model import ModelManager

app = FastAPI(
    title="THE PHYSIQ API",
    description="AI-powered exercise recommendation backend",
    version="1.0.0",
)

# CORS - 프론트엔드(localhost:5173)에서의 요청 허용
app.add_middleware(
    app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://team-project-ybdo.vercel.app",
        "https://thephysiq-ljciujmhn-muwek-s-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 시작 시 모델 초기 로드
@app.on_event("startup")
async def startup_event():
    ModelManager.load()
    print("✅ ML 모델 로드 완료")

app.include_router(recommend.router, prefix="/api")
app.include_router(retrain.router, prefix="/api")


@app.get("/")
def root():
    return {"status": "THE PHYSIQ API running"}
