import { Box, Card, Typography } from "@mui/material";
import { motion } from "motion/react";
import { useTheme } from "../ThemeProvider";
import type { RecommendResult } from "../../types";

interface Props {
  result: RecommendResult;
}

export function ResultCard({ result }: Props) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const cards = [
    {
      label: "Workout Type",
      value: result.workoutType,
      bgLight: "rgba(25, 118, 210, 0.15)",
      bgDark: "rgba(25, 118, 210, 0.15)",
      borderLight: "rgba(25, 118, 210, 0.4)",
      borderDark: "rgba(66, 165, 245, 0.3)",
      colorLight: "#1565C0",
      colorDark: "#42a5f5",
    },
    {
      label: "Session Duration",
      value: `${result.sessionDuration} hrs`,
      bgLight: "rgba(142, 68, 173, 0.15)",
      bgDark: "rgba(142, 68, 173, 0.15)",
      borderLight: "rgba(142, 68, 173, 0.4)",
      borderDark: "rgba(186, 104, 200, 0.3)",
      colorLight: "#6A1B9A",
      colorDark: "#ba68c8",
    },
    {
      label: "Frequency",
      value: `${result.frequency} days/week`,
      bgLight: "rgba(39, 174, 96, 0.15)",
      bgDark: "rgba(56, 142, 60, 0.15)",
      borderLight: "rgba(39, 174, 96, 0.4)",
      borderDark: "rgba(102, 187, 106, 0.3)",
      colorLight: "#27AE60",
      colorDark: "#66bb6a",
    },
    {
      label: "Calories",
      value: `${result.calories} kcal`,
      bgLight: "rgba(243, 156, 18, 0.15)",
      bgDark: "rgba(245, 124, 0, 0.15)",
      borderLight: "rgba(243, 156, 18, 0.4)",
      borderDark: "rgba(255, 167, 38, 0.3)",
      colorLight: "#F39C12",
      colorDark: "#ffa726",
    },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Box sx={{ mt: 4 }}>
        <Typography
          variant="h5"
          sx={{
            mb: 3,
            fontWeight: 700,
            color: isDark ? "#fff" : "#2C3E50",
            textAlign: "center",
          }}
        >
          Your Personalized Recommendation
        </Typography>

        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
          {cards.map((card) => (
            <Box
              key={card.label}
              sx={{ flex: { xs: "1 1 100%", sm: "1 1 calc(50% - 8px)", md: "1 1 calc(25% - 12px)" } }}
            >
              <Card
                sx={{
                  bgcolor: isDark ? card.bgDark : card.bgLight,
                  textAlign: "center",
                  p: 3,
                  border: `1px solid ${isDark ? card.borderDark : card.borderLight}`,
                  minHeight: "120px",
                  display: "flex",
                  flexDirection: "column",
                  justifyContent: "center",
                }}
              >
                <Typography
                  variant="body2"
                  sx={{ color: isDark ? "rgba(255,255,255,0.6)" : "#5A6C7D", mb: 1 }}
                >
                  {card.label}
                </Typography>
                <Typography
                  variant="h4"
                  sx={{
                    fontWeight: 700,
                    color: isDark ? card.colorDark : card.colorLight,
                    fontSize: card.value.length > 10 ? "1.75rem" : "2.125rem",
                    whiteSpace: "nowrap",
                  }}
                >
                  {card.value}
                </Typography>
              </Card>
            </Box>
          ))}
        </Box>
      </Box>
    </motion.div>
  );
}
