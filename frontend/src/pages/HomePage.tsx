import { useState } from "react";
import {
  Box, Container, Paper, Typography, Card, CardContent,
  Divider, Link,
} from "@mui/material";
import { GitHub, Code } from "@mui/icons-material";
import { motion, useScroll, useTransform } from "motion/react";
import { useTheme } from "../components/ThemeProvider";
import { Hero } from "../components/Hero";
import { ConsentDialog } from "../components/ConsentDialog";
import { RecommendForm } from "../components/RecommendForm";
import { ResultCard } from "../components/ResultCard";
import { AnimatedSection } from "../components/Layout/AnimatedSection";
import { useRecommend } from "../hooks/useRecommend";

export function HomePage() {
  const { resolvedTheme } = useTheme();
  const isDark = resolvedTheme === "dark";

  const [consentGiven, setConsentGiven] = useState(false);
  const [showConsent, setShowConsent] = useState(false);

  const { form, mode, result, loading, error, totalDataCount,
    updateField, setMode, recommend } = useRecommend();

  const { scrollYProgress } = useScroll();
  const contentY = useTransform(scrollYProgress, [0, 0.3], [0, -50]);

  const handleSubmit = () => {
    if (!consentGiven) {
      setShowConsent(true);
    } else {
      recommend();
    }
  };

  const handleConsentAccept = () => {
    setConsentGiven(true);
    setShowConsent(false);
    recommend();
  };

  const bg = isDark ? "#0a1929" : "#FFF8F0";

  return (
    <Box sx={{ minHeight: "100vh", bgcolor: bg, overflow: "hidden", position: "relative" }}>
      <ConsentDialog
        open={showConsent}
        onAccept={handleConsentAccept}
        onReject={() => setShowConsent(false)}
      />

      {/* Hero */}
      <Hero />

      {/* Content */}
      <Box component={motion.div} style={{ y: contentY }} sx={{ bgcolor: bg }}>
        <Container maxWidth="lg" sx={{ py: 8 }}>

          {/* About Section */}
          <AnimatedSection>
            <Paper
              elevation={3}
              sx={{ p: 4, mb: 6, bgcolor: isDark ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.8)", backdropFilter: "blur(10px)" }}
            >
              <Typography variant="h4" sx={{ mb: 4, fontWeight: 700, color: isDark ? "#fff" : "#2C3E50", textAlign: "center" }}>
                About THE PHYSIQ
              </Typography>
              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3 }}>
                {[
                  {
                    title: "Data Collection",
                    body: "We analyze 13 key parameters: Age, Gender, Weight, Height, BMI, Body Fat %, Max/Avg/Resting BPM, Workout Type, Session Duration, Frequency, and Calories Burned.",
                  },
                  {
                    title: "How It Works",
                    body: "Using RandomForest algorithm trained on gym member data, our system predicts optimal workout type, duration, frequency, and expected calorie burn based on your profile.",
                  },
                  {
                    title: "Accuracy",
                    body: "Trained on extensive gym tracking data with multiple regression and classification models to ensure personalized and effective workout recommendations.",
                  },
                ].map((card) => (
                  <Box key={card.title} sx={{ flex: { xs: "1 1 100%", md: "1 1 calc(33.333% - 16px)" } }}>
                    <Card
                      sx={{
                        bgcolor: isDark ? "rgba(25, 118, 210, 0.1)" : "rgba(230, 126, 34, 0.1)",
                        border: isDark ? "1px solid rgba(25, 118, 210, 0.3)" : "1px solid rgba(230, 126, 34, 0.3)",
                        height: "100%",
                      }}
                    >
                      <CardContent>
                        <Typography variant="h6" sx={{ color: isDark ? "#42a5f5" : "#E67E22", mb: 2, fontWeight: 600 }}>
                          {card.title}
                        </Typography>
                        <Typography variant="body2" sx={{ color: isDark ? "rgba(255,255,255,0.7)" : "#5A6C7D" }}>
                          {card.body}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Box>
                ))}
              </Box>
            </Paper>
          </AnimatedSection>

          {/* Recommender */}
          <AnimatedSection delay={0.2}>
            <Paper
              elevation={3}
              sx={{ p: 4, mb: 6, bgcolor: isDark ? "rgba(255,255,255,0.05)" : "rgba(255,255,255,0.8)", backdropFilter: "blur(10px)" }}
            >
              <Typography variant="h4" sx={{ mb: 4, fontWeight: 700, color: isDark ? "#fff" : "#2C3E50", textAlign: "center" }}>
                Get Your Recommendation
              </Typography>

              <RecommendForm
                form={form}
                mode={mode}
                loading={loading}
                error={error}
                consentGiven={consentGiven}
                totalDataCount={totalDataCount}
                onFieldChange={updateField}
                onModeChange={setMode}
                onSubmit={handleSubmit}
              />

              {result && <ResultCard result={result} />}
            </Paper>
          </AnimatedSection>

          {/* Developer Info */}
          <AnimatedSection delay={0.4}>
            <Paper
              elevation={3}
              sx={{
                p: 4,
                bgcolor: isDark ? "rgba(0,0,0,0.5)" : "rgba(255,255,255,0.8)",
                backdropFilter: "blur(10px)",
                border: isDark ? "1px solid rgba(255,255,255,0.1)" : "1px solid rgba(44, 62, 80, 0.2)",
              }}
            >
              <Typography variant="h4" sx={{ mb: 4, fontWeight: 700, color: isDark ? "#fff" : "#2C3E50", textAlign: "center" }}>
                Developer Information
              </Typography>

              <Box sx={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
                {[
                  {
                    icon: <GitHub sx={{ fontSize: 32, color: isDark ? "#42a5f5" : "#E67E22" }} />,
                    title: "GitHub Repository",
                    href: "https://github.com/muwek/-TEAM-PROJECT",
                    label: "github.com/muwek/-TEAM-PROJECT",
                  },
                  {
                    icon: <Code sx={{ fontSize: 32, color: isDark ? "#42a5f5" : "#E67E22" }} />,
                    title: "Google Colab Notebook",
                    href: "https://colab.research.google.com/drive/1qafBSjUzeVtGbZGO56H0PqJ07wgGzMPy?usp=sharing",
                    label: "View Training Notebook",
                  },
                ].map((item) => (
                  <Box key={item.title} sx={{ flex: { xs: "1 1 100%", md: "1 1 calc(50% - 16px)" } }}>
                    <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2 }}>
                      {item.icon}
                      <Typography variant="h6" sx={{ color: isDark ? "#fff" : "#2C3E50" }}>
                        {item.title}
                      </Typography>
                    </Box>
                    <Link
                      href={item.href}
                      target="_blank"
                      sx={{
                        color: isDark ? "#64b5f6" : "#E67E22",
                        textDecoration: "none",
                        fontSize: "1.1rem",
                        "&:hover": { textDecoration: "underline", color: isDark ? "#90caf9" : "#D35400" },
                      }}
                    >
                      {item.label}
                    </Link>
                  </Box>
                ))}
              </Box>

              <Divider sx={{ my: 4, bgcolor: isDark ? "rgba(255,255,255,0.1)" : "rgba(44, 62, 80, 0.15)" }} />

              <Typography variant="body1" sx={{ textAlign: "center", color: isDark ? "rgba(255,255,255,0.6)" : "#5A6C7D" }}>
                Developed by{" "}
                <span style={{ color: isDark ? "#42a5f5" : "#E67E22", fontWeight: 600 }}>
                  TEAM A : 김도현 김민성 김준영 노채영 이민용 이형균
                </span>
              </Typography>
              <Typography
                variant="body2"
                sx={{ textAlign: "center", color: isDark ? "rgba(255,255,255,0.4)" : "rgba(90, 108, 125, 0.5)", mt: 1 }}
              >
                THE PHYSIQ © 2026
              </Typography>
            </Paper>
          </AnimatedSection>

        </Container>
      </Box>
    </Box>
  );
}
