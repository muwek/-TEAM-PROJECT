import {
  Dialog, DialogTitle, DialogContent, DialogActions,
  Button, Typography, Box,
} from "@mui/material";
import { useTheme } from "../ThemeProvider";

interface ConsentDialogProps {
  open: boolean;
  onAccept: () => void;
  onReject: () => void;
}

export function ConsentDialog({ open, onAccept, onReject }: ConsentDialogProps) {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const sectionBg = isDark ? "rgba(66, 165, 245, 0.1)" : "rgba(230, 126, 34, 0.1)";
  const sectionColor = isDark ? "#42a5f5" : "#E67E22";
  const bodyColor = isDark ? "rgba(255,255,255,0.8)" : "#5A6C7D";

  return (
    <Dialog
      open={open}
      onClose={onReject}
      maxWidth="md"
      fullWidth
      PaperProps={{
        sx: {
          bgcolor: isDark ? "#1a1a2e" : "#FFFFFF",
          color: isDark ? "#fff" : "#2C3E50",
          border: isDark
            ? "1px solid rgba(66, 165, 245, 0.3)"
            : "1px solid rgba(230, 126, 34, 0.3)",
        },
      }}
    >
      <DialogTitle sx={{ color: sectionColor, fontWeight: 700, fontSize: "1.5rem" }}>
        개인정보 수집 및 이용 동의
      </DialogTitle>

      <DialogContent>
        <Typography variant="body1" sx={{ mb: 2, color: isDark ? "rgba(255,255,255,0.9)" : "#2C3E50" }}>
          THE PHYSIQ는 「개인정보 보호법」 제15조 및 제22조에 따라 귀하의 개인정보를 수집 및 이용하고자 합니다.
        </Typography>

        {[
          {
            title: "1. 개인정보의 수집 및 이용 목적",
            content: "• 운동 추천 서비스 제공\n• AI 모델 학습 및 서비스 개선\n• 맞춤형 운동 프로그램 제공",
          },
          {
            title: "2. 수집하는 개인정보 항목",
            content: "• 나이, 성별, 체중, 신장, BMI, 체지방률\n• 목표 칼로리 (선택적)",
          },
          {
            title: "3. 개인정보의 보유 및 이용 기간",
            content: "• 수집된 정보는 AI 모델 학습을 위해 익명화되어 보관됩니다.\n• 서비스 이용 기간 동안 보유되며, 서비스 종료 시 파기됩니다.",
          },
        ].map((section) => (
          <Box key={section.title} sx={{ bgcolor: sectionBg, p: 2, borderRadius: 1, mb: 2 }}>
            <Typography variant="h6" sx={{ color: sectionColor, mb: 1, fontWeight: 600 }}>
              {section.title}
            </Typography>
            <Typography variant="body2" sx={{ color: bodyColor, whiteSpace: "pre-line" }}>
              {section.content}
            </Typography>
          </Box>
        ))}

        <Box
          sx={{
            bgcolor: isDark ? "rgba(255, 152, 0, 0.1)" : "rgba(211, 84, 0, 0.1)",
            p: 2,
            borderRadius: 1,
            mb: 2,
            border: isDark
              ? "1px solid rgba(255, 152, 0, 0.3)"
              : "1px solid rgba(211, 84, 0, 0.3)",
          }}
        >
          <Typography variant="h6" sx={{ color: isDark ? "#ffa726" : "#D35400", mb: 1, fontWeight: 600 }}>
            4. 동의 거부권 및 불이익
          </Typography>
          <Typography variant="body2" sx={{ color: bodyColor }}>
            귀하는 개인정보 수집 및 이용에 대한 동의를 거부할 권리가 있습니다.<br />
            다만, 동의를 거부할 경우 운동 추천 서비스 이용이 제한될 수 있습니다.
          </Typography>
        </Box>

        <Typography variant="body2" sx={{ mt: 3, color: bodyColor, fontStyle: "italic" }}>
          본 동의는 서비스 이용 시 1회 동의하시면 브라우저 세션이 유지되는 동안 유효합니다.
        </Typography>
      </DialogContent>

      <DialogActions sx={{ p: 3, gap: 2 }}>
        <Button
          onClick={onReject}
          variant="outlined"
          sx={{
            color: isDark ? "rgba(255,255,255,0.7)" : "#5A6C7D",
            borderColor: isDark ? "rgba(255,255,255,0.3)" : "rgba(90, 108, 125, 0.3)",
            "&:hover": {
              borderColor: isDark ? "rgba(255,255,255,0.5)" : "rgba(90, 108, 125, 0.5)",
              bgcolor: isDark ? "rgba(255,255,255,0.05)" : "rgba(0,0,0,0.05)",
            },
          }}
        >
          동의하지 않음
        </Button>
        <Button
          onClick={onAccept}
          variant="contained"
          sx={{
            bgcolor: isDark ? "#1976d2" : "#E67E22",
            "&:hover": { bgcolor: isDark ? "#1565c0" : "#D35400" },
          }}
        >
          동의하고 계속하기
        </Button>
      </DialogActions>
    </Dialog>
  );
}
