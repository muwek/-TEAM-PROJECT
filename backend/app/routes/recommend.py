from fastapi import APIRouter, HTTPException
from app.ml.model import ModelManager
from app.models.schemas import (
    RecommendV1Request,
    RecommendV2Request,
    RecommendResponse,
)

router = APIRouter(tags=["recommend"])


@router.post("/recommend/v1", response_model=RecommendResponse)
def recommend_v1(body: RecommendV1Request):
    """
    기본 추천 (V1)
    BMI, 나이, 성별, 체지방률 기반 운동 종류 / 시간 / 빈도 / 칼로리 예측
    """
    try:
        result = ModelManager.predict_v1(
            age=body.age,
            gender=body.gender,
            weight=body.weight,
            height=body.height,
            bmi=body.bmi,
            fat=body.fatPercentage,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend/v2", response_model=RecommendResponse)
def recommend_v2(body: RecommendV2Request):
    """
    목표 칼로리 모드 (V2)
    V1 결과를 기반으로 목표 칼로리에 맞게 운동 시간을 비례 조정
    """
    try:
        result = ModelManager.predict_v2(
            age=body.age,
            gender=body.gender,
            weight=body.weight,
            height=body.height,
            bmi=body.bmi,
            fat=body.fatPercentage,
            target_calories=body.targetCalories,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
