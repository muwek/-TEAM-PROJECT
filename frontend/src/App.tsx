import { Box } from "@mui/material";
import { ThemeProvider } from "./components/ThemeProvider";
import { ThemeToggle } from "./components/ThemeToggle";
import { HomePage } from "./pages/HomePage";

export default function App() {
  return (
    <ThemeProvider>
      {/* 테마 토글 버튼 — 모든 페이지에 고정 */}
      <Box sx={{ position: "fixed", top: 20, right: 20, zIndex: 1000 }}>
        <ThemeToggle />
      </Box>
      <HomePage />
    </ThemeProvider>
  );
}
