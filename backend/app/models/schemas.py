from pydantic import BaseModel, Field
from typing import Literal


class RecommendV1Request(BaseModel):
    age: int = Field(..., ge=10, le=100, description="나이")
    gender: Literal["Male", "Female"]
    weight: float = Field(..., gt=0, description="체중 (kg)")
    height: float = Field(..., gt=0, description="신장 (m)")
    bmi: float = Field(..., gt=0, description="BMI")
    fatPercentage: float = Field(..., ge=0, le=100, description="체지방률 (%)")


class RecommendV2Request(RecommendV1Request):
    targetCalories: float = Field(..., gt=0, description="목표 칼로리 (kcal)")


class RecommendResponse(BaseModel):
    workoutType: str
    sessionDuration: float
    frequency: float
    calories: float


class UserDataPayload(BaseModel):
    """사용자 데이터 누적 및 재학습 요청 스키마"""
    age: int
    gender: Literal["Male", "Female"]
    weight: float
    height: float
    bmi: float
    fatPercentage: float
    workoutType: str
    sessionDuration: float
    frequency: float
    calories: float


class RetrainResponse(BaseModel):
    message: str
    totalDataCount: int
