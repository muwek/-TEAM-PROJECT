import {
  Box, Typography, TextField, Button, Select, MenuItem,
  FormControl, InputLabel, ToggleButtonGroup, ToggleButton,
  CircularProgress, Alert,
} from "@mui/material";
import { useTheme } from "../ThemeProvider";
import type { RecommendMode } from "../../types";

interface Props {
  form: {
    age: number; gender: "Male" | "Female"; weight: number;
    height: number; bmi: number; fatPercentage: number; targetCalories: number;
  };
  mode: RecommendMode;
  loading: boolean;
  error: string | null;
  consentGiven: boolean;
  totalDataCount: number | null;
  onFieldChange: <K extends string>(key: K, value: number | string) => void;
  onModeChange: (mode: RecommendMode) => void;
  onSubmit: () => void;
}

export function RecommendForm({
  form, mode, loading, error, consentGiven,
  totalDataCount, onFieldChange, onModeChange, onSubmit,
}: Props) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const inputSx = {
    "& .MuiOutlinedInput-root": {
      color: isDark ? "#fff" : "#2C3E50",
      "& fieldset": { borderColor: isDark ? "rgba(255,255,255,0.2)" : "rgba(44, 62, 80, 0.2)" },
      "&:hover fieldset": { borderColor: isDark ? "rgba(255,255,255,0.4)" : "rgba(44, 62, 80, 0.4)" },
      "&.Mui-focused fieldset": { borderColor: isDark ? "#42a5f5" : "#E67E22" },
    },
    "& .MuiInputLabel-root": { color: isDark ? "rgba(255,255,255,0.7)" : "#5A6C7D" },
  };

  const fieldProps = (label: string, key: string, value: number, step?: number) => ({
    fullWidth: true, label, type: "number" as const, value,
    inputProps: step ? { step } : undefined,
    onChange: (e: React.ChangeEvent<HTMLInputElement>) =>
      onFieldChange(key, Number(e.target.value)),
    sx: inputSx,
  });

  return (
    <Box>
      {/* 동의 완료 배지 */}
      {consentGiven && (
        <Box sx={{ textAlign: "center", mb: 2 }}>
          <Typography
            variant="caption"
            sx={{
              display: "inline-block", px: 2, py: 0.5,
              bgcolor: "rgba(76, 175, 80, 0.2)",
              color: isDark ? "#66bb6a" : "#27AE60",
              borderRadius: 2, border: "1px solid rgba(76, 175, 80, 0.3)",
            }}
          >
            ✓ 데이터 수집 동의 완료
            {totalDataCount !== null && ` (총 학습 데이터: ${totalDataCount}건)`}
          </Typography>
        </Box>
      )}

      {/* 모드 선택 */}
      <Box sx={{ mb: 4 }}>
        <ToggleButtonGroup
          value={mode}
          exclusive
          onChange={(_, v) => v && onModeChange(v)}
          fullWidth
          sx={{
            "& .MuiToggleButton-root": {
              color: isDark ? "rgba(255,255,255,0.7)" : "#5A6C7D",
              borderColor: isDark ? "rgba(255,255,255,0.2)" : "rgba(44, 62, 80, 0.2)",
              "&.Mui-selected": {
                bgcolor: isDark ? "rgba(25, 118, 210, 0.3)" : "rgba(230, 126, 34, 0.15)",
                color: isDark ? "#42a5f5" : "#E67E22",
                borderColor: isDark ? "#42a5f5" : "#E67E22",
              },
            },
          }}
        >
          <ToggleButton value="v1">Basic Recommendation</ToggleButton>
          <ToggleButton value="v2">Target Calorie Mode</ToggleButton>
        </ToggleButtonGroup>
      </Box>

      {/* 입력 필드 그리드 */}
      <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3, mb: 4 }}>
        <Box sx={{ flex: { xs: "1 1 100%", sm: "1 1 calc(50% - 12px)", md: "1 1 calc(33.333% - 16px)" } }}>
          <TextField {...fieldProps("Age", "age", form.age)} />
        </Box>

        <Box sx={{ flex: { xs: "1 1 100%", sm: "1 1 calc(50% - 12px)", md: "1 1 calc(33.333% - 16px)" } }}>
          <FormControl fullWidth>
            <InputLabel sx={{ color: isDark ? "rgba(255,255,255,0.7)" : "#5A6C7D" }}>Gender</InputLabel>
            <Select
              value={form.gender}
              label="Gender"
              onChange={(e) => onFieldChange("gender", e.target.value)}
              sx={{
                color: isDark ? "#fff" : "#2C3E50",
                "& .MuiOutlinedInput-notchedOutline": { borderColor: isDark ? "rgba(255,255,255,0.2)" : "rgba(44, 62, 80, 0.2)" },
                "&:hover .MuiOutlinedInput-notchedOutline": { borderColor: isDark ? "rgba(255,255,255,0.4)" : "rgba(44, 62, 80, 0.4)" },
                "&.Mui-focused .MuiOutlinedInput-notchedOutline": { borderColor: isDark ? "#42a5f5" : "#E67E22" },
                "& .MuiSvgIcon-root": { color: isDark ? "rgba(255,255,255,0.7)" : "#5A6C7D" },
              }}
            >
              <MenuItem value="Male">Male</MenuItem>
              <MenuItem value="Female">Female</MenuItem>
            </Select>
          </FormControl>
        </Box>

        <Box sx={{ flex: { xs: "1 1 100%", sm: "1 1 calc(50% - 12px)", md: "1 1 calc(33.333% - 16px)" } }}>
          <TextField {...fieldProps("Weight (kg)", "weight", form.weight)} />
        </Box>

        <Box sx={{ flex: { xs: "1 1 100%", sm: "1 1 calc(50% - 12px)", md: "1 1 calc(33.333% - 16px)" } }}>
          <TextField {...fieldProps("Height (m)", "height", form.height, 0.01)} />
        </Box>

        <Box sx={{ flex: { xs: "1 1 100%", sm: "1 1 calc(50% - 12px)", md: "1 1 calc(33.333% - 16px)" } }}>
          <TextField {...fieldProps("BMI (자동계산)", "bmi", form.bmi)} />
        </Box>

        <Box sx={{ flex: { xs: "1 1 100%", sm: "1 1 calc(50% - 12px)", md: "1 1 calc(33.333% - 16px)" } }}>
          <TextField {...fieldProps("Fat Percentage (%)", "fatPercentage", form.fatPercentage)} />
        </Box>

        {mode === "v2" && (
          <Box sx={{ flex: "1 1 100%" }}>
            <TextField {...fieldProps("Target Calories (kcal)", "targetCalories", form.targetCalories)} />
          </Box>
        )}
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Button
        fullWidth
        variant="contained"
        size="large"
        onClick={onSubmit}
        disabled={loading}
        sx={{
          bgcolor: isDark ? "#1976d2" : "#E67E22",
          "&:hover": { bgcolor: isDark ? "#1565c0" : "#D35400" },
          py: 1.5, fontSize: "1.1rem", fontWeight: 700,
        }}
      >
        {loading ? <CircularProgress size={24} color="inherit" /> : "Get Recommendation"}
      </Button>
    </Box>
  );
}
