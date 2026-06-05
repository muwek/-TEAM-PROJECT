import { Box, Typography } from "@mui/material";
import { FitnessCenter } from "@mui/icons-material";
import { motion, useScroll, useTransform } from "motion/react";
import { useTheme } from "../ThemeProvider";

export function Hero() {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const { scrollYProgress } = useScroll();
  const heroScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.85]);
  const heroOpacity = useTransform(scrollYProgress, [0, 0.4], [1, 0]);

  return (
    <Box
      sx={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        position: "relative",
        background: isDark ? "#0a1929" : "#FFF8F0",
      }}
    >
      <Box
        component={motion.div}
        sx={{ textAlign: "center", px: 2 }}
        style={{ scale: heroScale, opacity: heroOpacity }}
      >
        {/* Dumbbell Icon */}
        <motion.div
          initial={{ scale: 0, rotate: -180 }}
          animate={{ scale: 1, rotate: 0 }}
          transition={{ duration: 1.2, delay: 0.1, type: "spring", stiffness: 200, damping: 15 }}
        >
          <FitnessCenter
            sx={{
              fontSize: { xs: 80, md: 120 },
              color: isDark ? "#1976d2" : "#E67E22",
              mb: 3,
            }}
          />
        </motion.div>

        {/* THE PHYSIQ 타이틀 */}
        <Typography
          variant="h1"
          component="h1"
          sx={{
            fontWeight: 800,
            fontSize: { xs: "3rem", sm: "5rem", md: "7rem" },
            mb: 3,
            letterSpacing: "-0.02em",
            textAlign: "center",
          }}
        >
          {"THE PHYSIQ".split("").map((char, index) => (
            <motion.span
              key={index}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.5 + index * 0.1, ease: [0.4, 0, 0.2, 1] }}
              style={{
                display: "inline",
                backgroundImage: isDark
                  ? "linear-gradient(90deg, #1976d2 0%, #42a5f5 100%)"
                  : "linear-gradient(90deg, #E67E22 0%, #F39C12 100%)",
                backgroundClip: "text",
                WebkitBackgroundClip: "text",
                WebkitTextFillColor: "transparent",
                marginRight: char === " " ? "0.3em" : "0",
              }}
            >
              {char === " " ? "\u00A0" : char}
            </motion.span>
          ))}
        </Typography>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 1.8 }}
        >
          <Typography
            variant="h4"
            sx={{
              color: isDark ? "rgba(255,255,255,0.8)" : "#2C3E50",
              fontWeight: 300,
              mb: 2,
              fontSize: { xs: "1.5rem", md: "2.5rem" },
            }}
          >
            AI-Powered Exercise Recommendation
          </Typography>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 2.2 }}
        >
          <Typography
            variant="h6"
            sx={{
              color: isDark ? "rgba(255,255,255,0.6)" : "#5A6C7D",
              fontWeight: 300,
              maxWidth: 600,
              mx: "auto",
              fontSize: { xs: "1rem", md: "1.25rem" },
            }}
          >
            Advanced machine learning algorithm for personalized workout routines
          </Typography>
        </motion.div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 2.6, repeat: Infinity, repeatType: "reverse" }}
        >
          <Typography
            sx={{
              color: isDark ? "rgba(255,255,255,0.4)" : "rgba(90, 108, 125, 0.6)",
              mt: 8,
              fontSize: "0.875rem",
            }}
          >
            ↓ Scroll to explore
          </Typography>
        </motion.div>
      </Box>
    </Box>
  );
}
