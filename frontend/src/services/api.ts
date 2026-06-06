/// <reference types="vite/client" />
import type {
  RecommendV1Params,
  RecommendV2Params,
  RecommendResult,
  UserDataPayload,
  RetrainResponse,
  RetrainStatus,
} from "../types";

const BASE = import.meta.env.VITE_API_URL ?? "/api";

async function post<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail ?? "Request failed");
  }
  return res.json();
}

async function get<T>(url: string): Promise<T> {
  const res = await fetch(`${BASE}${url}`);
  if (!res.ok) throw new Error("Request failed");
  return res.json();
}

/** V1: 기본 추천 */
export const recommendV1 = (params: RecommendV1Params): Promise<RecommendResult> =>
  post("/recommend/v1", params);

/** V2: 목표 칼로리 모드 */
export const recommendV2 = (params: RecommendV2Params): Promise<RecommendResult> =>
  post("/recommend/v2", params);

/** 사용자 데이터 누적 + 재학습 요청 */
export const submitUserData = (payload: UserDataPayload): Promise<RetrainResponse> =>
  post("/retrain", payload);

/** 현재 학습 데이터 수 조회 */
export const getRetrainStatus = (): Promise<RetrainStatus> =>
  get("/retrain/status");
