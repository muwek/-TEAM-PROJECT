from fastapi import APIRouter, HTTPException
from app.ml.model import ModelManager
from app.models.schemas import UserDataPayload, RetrainResponse

router = APIRouter(tags=["retrain"])


@router.post("/retrain", response_model=RetrainResponse)
def retrain(body: UserDataPayload):
    """
    사용자 데이터 누적 및 실시간 재학습
    - 사용자의 입력값 + 추천 결과를 데이터셋에 추가
    - 전체 모델 즉시 재학습
    - model.pkl 및 user_logs.csv 업데이트
    """
    try:
        result = ModelManager.add_user_data_and_retrain(body.model_dump())
        return RetrainResponse(
            message="데이터 반영 및 재학습 완료",
            totalDataCount=result["totalDataCount"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrain/status")
def retrain_status():
    """현재 학습된 데이터 수 조회"""
    count = len(ModelManager.df) if ModelManager.df is not None else 0
    return {"totalDataCount": count, "modelLoaded": ModelManager._loaded}
