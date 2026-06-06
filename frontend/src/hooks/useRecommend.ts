import { useState, useCallback } from "react";
import { recommendV1, recommendV2, submitUserData } from "../services/api";
import type {
  RecommendMode,
  RecommendResult,
  RecommendV1Params,
} from "../types";

interface FormState extends RecommendV1Params {
  targetCalories: number;
}

const DEFAULT_FORM: FormState = {
  age: 25,
  gender: "Male",
  weight: 70,
  height: 1.75,
  bmi: 22.9,
  fatPercentage: 18,
  targetCalories: 500,
};

export function useRecommend() {
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [mode, setMode] = useState<RecommendMode>("v1");
  const [result, setResult] = useState<RecommendResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [totalDataCount, setTotalDataCount] = useState<number | null>(null);

  /** BMI 자동 계산 포함 폼 업데이트 */
const updateField = useCallback(
  (key: string, value: number | string) => {
    setForm((prev) => {
      const next = { ...prev, [key]: value };
        if (key === "weight" || key === "height") {
          const w = key === "weight" ? (value as number) : prev.weight;
          const h = key === "height" ? (value as number) : prev.height;
          if (h > 0) next.bmi = parseFloat((w / (h * h)).toFixed(2));
        }
        return next;
      });
    },
    []
  );

  /** 추천 실행 + 사용자 데이터 백엔드 전송 */
  const recommend = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      let res: RecommendResult;
      if (mode === "v1") {
        res = await recommendV1(form);
      } else {
        res = await recommendV2({ ...form, targetCalories: form.targetCalories });
      }
      setResult(res);

      // 추천 결과를 사용자 데이터로 백엔드에 전송 (재학습용)
      const retrainRes = await submitUserData({
        age: form.age,
        gender: form.gender,
        weight: form.weight,
        height: form.height,
        bmi: form.bmi,
        fatPercentage: form.fatPercentage,
        workoutType: res.workoutType,
        sessionDuration: res.sessionDuration,
        frequency: res.frequency,
        calories: res.calories,
      });
      setTotalDataCount(retrainRes.totalDataCount);
    } catch (e) {
      setError(e instanceof Error ? e.message : "오류가 발생했습니다.");
    } finally {
      setLoading(false);
    }
  }, [form, mode]);

  return {
    form,
    mode,
    result,
    loading,
    error,
    totalDataCount,
    updateField,
    setMode,
    recommend,
  };
}
