import { IconButton, Tooltip } from "@mui/material";
import { LightMode, DarkMode } from "@mui/icons-material";
import { useTheme } from "../ThemeProvider";

export function ThemeToggle() {
  const { resolvedTheme, toggleTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  return (
    <Tooltip title={isDark ? "라이트 모드" : "다크 모드"}>
      <IconButton
        onClick={toggleTheme}
        sx={{
          color: isDark ? "#fff" : "#2C3E50",
          bgcolor: isDark ? "rgba(255,255,255,0.1)" : "rgba(44,62,80,0.08)",
          "&:hover": {
            bgcolor: isDark ? "rgba(255,255,255,0.2)" : "rgba(44,62,80,0.15)",
          },
        }}
      >
        {isDark ? <LightMode /> : <DarkMode />}
      </IconButton>
    </Tooltip>
  );
}
