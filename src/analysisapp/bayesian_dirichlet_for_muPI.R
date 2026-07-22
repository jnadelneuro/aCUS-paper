# ===========================================================
# RI60 Microstructure: Bayesian Dirichlet Regression
# Requires: brms (which wraps Stan), tidyverse
# Run from the behaviormodels folder
# ===========================================================

library(brms)
library(tidyverse)
source("_config.R")

base_dir <- file.path(RI60_DIR, "behaviormodels")
setwd(base_dir)


# --- Load and prep ---
sess <- read_csv("session_weights.csv")

# Factor with naive as reference
sess$group <- factor(sess$group, levels = c("naive", "stress"))

# brms Dirichlet needs strictly positive values; clip any near-zero
sess <- sess %>%
  mutate(across(starts_with("pi_"), ~ pmax(.x, 1e-6)))

sess <- sess %>%
  mutate(across(starts_with("pi_"), ~ pmax(.x, 1e-6))) %>%
  mutate(pi_sum = pi_0 + pi_1 + pi_2 + pi_3 + pi_4,
         across(starts_with("pi_"), ~ .x / pi_sum)) %>%
  select(-pi_sum)

# Response matrix — first column (pi_0) becomes the reference category
# brms names the other categories mupi_1, mupi_2, mupi_3, mupi_4
sess$Y <- with(sess, cbind(pi_0, pi_1, pi_2, pi_3, pi_4))

cat("Sessions:", nrow(sess), " Mice:", n_distinct(sess$mouse),
    " Groups:", table(sess$group), "\n")

# ===========================================================
# Model 1: Does composition differ by group (controlling for day)?
#   Each non-reference category gets its own group + day coefficients
#   on the log-ratio scale (relative to pi_0)
# ===========================================================
fit_group <- brm(
  Y ~ group + day + (1 | mouse),
  data = sess,
  family = dirichlet(),
  chains = 4,
  cores = 4,
  iter = 4000,
  warmup = 1000,
  seed = 42,
  control = list(adapt_delta = 0.95),
  file = "brm_dirichlet_group"
)

# Check convergence — all Rhat should be < 1.01
summary(fit_group)
# Quick visual check:
# plot(fit_group, ask = FALSE)

# ===========================================================
# Model 2: Does the group effect change across training days?
#   Adds group × day interaction per category
# ===========================================================
fit_interact <- brm(
  Y ~ group * day + (1 | mouse),
  data = sess,
  family = dirichlet(),
  chains = 4,
  cores = 4,
  iter = 4000,
  warmup = 1000,
  seed = 42,
  control = list(adapt_delta = 0.95),
  file = "brm_dirichlet_interact"
)

summary(fit_interact)

# ===========================================================
# Model comparison: does adding the interaction improve fit?
# ===========================================================
fit_group   <- add_criterion(fit_group, "loo")
fit_interact <- add_criterion(fit_interact, "loo")
print(loo_compare(fit_group, fit_interact))

# ===========================================================
# Post-hoc: group effect per component
#
# brms Dirichlet coefficients are log-ratios relative to pi_0.
# Coefficient names: mupi_1_groupstress, mupi_2_groupstress, etc.
# A negative coefficient for mupi_4_groupstress means stress mice
# have LESS pi_4 relative to pi_0 compared to naive mice.
#
# First, see what the coefficients are actually called:
# ===========================================================
cat("\n--- Coefficient names ---\n")
print(variables(fit_group))

# Test: is the group effect on each component significant?
# (credible interval excludes zero)
cat("\n--- Group effect per component (from fit_group) ---\n")
group_effects <- fixef(fit_group)
# Print rows containing "groupstress"
print(group_effects[grep("groupstress", rownames(group_effects)), ])

# Formal hypothesis tests (evidence ratio = Bayes factor analog)
cat("\n--- Hypothesis tests: group effect per component ---\n")

# You'll need to adjust these names based on what variables() shows.
# The pattern is typically mupi_N_groupstress where N is 1-4.
# Run variables(fit_group) first to get exact names, then update.
hyp_names <- paste0("mupi", 1:4, "_groupstress")

for (h in hyp_names) {
  cat("\n", h, "= 0:\n")
  print(hypothesis(fit_group, paste0(h, " = 0")))
}

# ===========================================================
# Post-hoc: which component's group effect is LARGEST?
# Compare group effects between components (e.g., c4 vs c1)
# ===========================================================
cat("\n--- Pairwise: is c4 group effect different from c1? ---\n")
print(hypothesis(fit_group, "mupi4_groupstress < mupi1_groupstress"))

cat("\n--- Pairwise: is c4 group effect different from c3? ---\n")
print(hypothesis(fit_group, "mupi4_groupstress < mupi3_groupstress"))

# ===========================================================
# Post-hoc: group × day interaction per component
# (from fit_interact)
# ===========================================================
cat("\n--- Interaction coefficients (group × day per component) ---\n")
interact_effects <- fixef(fit_interact)
print(interact_effects[grep("groupstress:day", rownames(interact_effects)), ])

# Formal test: is the group × day interaction on c4 significant?
cat("\n--- Hypothesis: group × day interaction on c4 ---\n")
print(hypothesis(fit_interact, "mupi4_groupstress:day = 0"))

# ===========================================================
# Predicted compositions for plotting
# ===========================================================
newdata <- expand_grid(
  group = c("naive", "stress"),
  day = c(1, 7, 14)
)

pred <- fitted(fit_group, newdata = newdata,
               re_formula = NA) # population-level predictions only

cat("\n--- Predicted compositions ---\n")
print(cbind(newdata, round(pred[, , "Estimate"], 4)))

# ===========================================================
# Save key results
# ===========================================================

# Group effects table
ge_df <- as.data.frame(group_effects[grep("groupstress", rownames(group_effects)), ])
ge_df$coefficient <- rownames(ge_df)
ge_df$component <- c("pi_1 (fast check, 0.84s)",
                     "pi_2 (within-bout, 2.27s)",
                     "pi_3 (between-bout, 6.80s)",
                     "pi_4 (disengaged, 21.19s)")
write_csv(ge_df, "bayes_group_effects.csv")

# Interaction effects table
ie_df <- as.data.frame(interact_effects[grep("groupstress:day", rownames(interact_effects)), ])
ie_df$coefficient <- rownames(ie_df)
write_csv(ie_df, "bayes_interaction_effects.csv")

# Full posterior summaries
write_csv(as.data.frame(fixef(fit_group)), "bayes_fixef_group.csv")
write_csv(as.data.frame(fixef(fit_interact)), "bayes_fixef_interact.csv")

cat("\nDone. Wrote bayes_group_effects.csv, bayes_interaction_effects.csv,",
    "bayes_fixef_group.csv, bayes_fixef_interact.csv\n")

# ===========================================================
# Model 3: Day as factor — per-day group contrasts on pi_4
# ===========================================================
sess$day_f <- factor(sess$day)

fit_day_f <- brm(
  Y ~ group * day_f + (1 | mouse),
  data = sess,
  family = dirichlet(),
  chains = 4,
  cores = 4,
  iter = 4000,
  warmup = 1000,
  seed = 42,
  control = list(adapt_delta = 0.95),
  file = "brm_dirichlet_day_factor"
)

summary(fit_day_f)

# Posterior predictions for each group × day
newdata <- expand_grid(
  group = factor(c("naive", "stress"), levels = c("naive", "stress")),
  day_f = factor(1:14)
)

pred <- fitted(fit_day_f, newdata = newdata,
               re_formula = NA, summary = FALSE)

# Category 5 = pi_4 (disengaged)
pi4_draws <- pred[, , 5]

# naive = odd rows, stress = even rows
naive_idx  <- seq(1, 28, by = 2)
stress_idx <- seq(2, 28, by = 2)

cat("\n===== GROUP DIFFERENCE IN pi_4 PER DAY =====\n")
results <- data.frame()
for (d in 1:14) {
  diff_draws <- pi4_draws[, naive_idx[d]] - pi4_draws[, stress_idx[d]]
  ci <- quantile(diff_draws, c(0.025, 0.975))
  prob_gt_zero <- mean(diff_draws > 0)
  results <- rbind(results, data.frame(
    day = d,
    median_diff = median(diff_draws),
    lower_95 = ci[1],
    upper_95 = ci[2],
    prob_naive_gt_stress = prob_gt_zero,
    sig = ifelse(ci[1] > 0 | ci[2] < 0, "*", "")
  ))
}
print(results, digits = 3)
write_csv(results, "bayes_c4_group_diff_by_day.csv")
 