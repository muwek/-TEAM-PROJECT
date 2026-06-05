export interface RecommendV1Params {
  age: number;
  gender: "Male" | "Female";
  weight: number;
  height: number;
  bmi: number;
  fatPercentage: number;
}

export interface RecommendV2Params extends RecommendV1Params {
  targetCalories: number;
}

export interface RecommendResult {
  workoutType: string;
  sessionDuration: number;
  frequency: number;
  calories: number;
}

export interface UserDataPayload extends RecommendV1Params {
  workoutType: string;
  sessionDuration: number;
  frequency: number;
  calories: number;
}

export interface RetrainResponse {
  message: string;
  totalDataCount: number;
}

export interface RetrainStatus {
  totalDataCount: number;
  modelLoaded: boolean;
}

export type RecommendMode = "v1" | "v2";
export type Theme = "light" | "dark";
