library(survival)
library(timeROC)

# ==================
# Oracle Model
# ==================

set.seed(42)

n <- 20000
X <- rnorm(n) 

# Simulate Primary Event and Censoring
rate_cause1 <- exp(2 * X) 
T1 <- rexp(n, rate = rate_cause1)
C <- rexp(n, rate = 0.1)
#C <- rep(Inf, n)
eval_times <- seq(0.5, 3, by = 0.25)

# Baseline (No Competing Risks) ---
time_true <- pmin(T1, C)
event_true <- ifelse(T1 < C, 1, 0)
roc_true <- timeROC(T = time_true, delta = event_true, marker = X, 
                    cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)

# Compute roc for a simulated competing risk
simulate_cr_scenario <- function(cr_rate_multiplier) {
  # Simulate Competing Event (T2)
  rate_cause2 <- exp(cr_rate_multiplier)
  T2 <- rexp(n, rate = rate_cause2)
  time <- pmin(T1, T2, C)
  event <- ifelse(T1 < T2 & T1 < C, 1, 
                  ifelse(T2 < T1 & T2 < C, 2, 0))
  
  # Calculate ROC curves
  roc <- timeROC(T = time, delta = event, marker = X, 
                 cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)
  return(roc)
}

# Run a Low Rate and High Rate scenario
roc_low_cr  <- simulate_cr_scenario(-1.0)
roc_high_cr <- simulate_cr_scenario(1.5)

# Plot the graph
plot(eval_times, roc_true$AUC, type="b", col="gray", lwd=3, ylim=c(0.7, 1.05),
     ylab="Time-Dependent AUC", xlab="Time Horizon (t)", 
     main="Impact of Competing Risk on AUC metrics (Oracle Model)")

# Add AUC = 1.0 reference line
abline(h = 1.0, lty = 3, col = "black", lwd = 2)

# Plot Other AUC Lines
lines(eval_times, roc_low_cr$AUC_1, type="b", col="red", lwd=2, lty=2) # Cause-Specific
lines(eval_times, roc_low_cr$AUC_2, type="b", col="red", lwd=2, lty=1)  # Sub-dist
lines(eval_times, roc_high_cr$AUC_1, type="b", col="blue", lwd=2, lty=2) # Cause-Specific
lines(eval_times, roc_high_cr$AUC_2, type="b", col="blue", lwd=2, lty = 1)  # Sub-dist

# Add Legend
legend("bottomleft", 
       legend=c("Theoretical Maximum (AUC = 1.0)",
                "True Biological AUC (No Competing Risks)", 
                "High CR: Cause-Specific AUC", 
                "High CR: Sub-distribution AUC",
                "Low CR: Cause-Specific AUC",
                "Low CR: Sub-distribution AUC"),
       col=c("black", "gray", "blue", "blue", "red", "red"), 
       lty=c(3, 1, 2, 1, 2, 1), 
       lwd=c(2, 3, 2, 2, 1, 1), cex=0.7)

# ====================
# Model Simulation
# ====================

set.seed(42)

generate_cohort <- function(n, cr_rate_multiplier) {
  # Generate markers and data
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  rate_cause1 <- exp(1.5 * X1 + 0.8 * X2)
  T1 <- rexp(n, rate = rate_cause1)
  # Competing Event unrelated to markers
  rate_cause2 <- exp(cr_rate_multiplier)
  T2 <- rexp(n, rate = rate_cause2)
  C <- rexp(n, rate = 0.1)
  time <- pmin(T1, T2, C)
  event <- ifelse(T1 < T2 & T1 < C, 1, 
                  ifelse(T2 < T1 & T2 < C, 2, 0))
  return(data.frame(time, event, X1, X2))
}

# Generate 2000 patients for Training, 2000 for Testing
train_data <- generate_cohort(2000, 1.0)
test_data  <- generate_cohort(2000, 1.0)


# Run models
cox_model <- coxph(Surv(time, event == 1) ~ X1 + X2, data = train_data)
fg_data <- finegray(Surv(time, factor(event)) ~ ., data = train_data)
fg_model <- coxph(Surv(fgstart, fgstop, fgstatus) ~ X1 + X2, weight = fgwt, data = fg_data)

score_cox <- predict(cox_model, newdata = test_data, type = "risk")
score_fg  <- predict(fg_model,  newdata = test_data, type = "risk")

eval_times <- seq(0.5, 3, by = 0.5)

roc_cox <- timeROC(T = test_data$time, delta = test_data$event, marker = score_cox, 
                   cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)
roc_fg <- timeROC(T = test_data$time, delta = test_data$event, marker = score_fg, 
                  cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)

# Plot AUC curves
plot(eval_times, roc_cox$AUC_1, type="b", col="blue", lwd=2, ylim=c(0.8, 1.0),
     ylab="Time-Dependent AUC", xlab="Time Horizon (t)", 
     main="Cox vs. Fine-Gray Scores evaluated by AUC")

abline(h = 1.0, lty = 3, col = "black", lwd = 2)

# Plot Fine-Gray scores evaluated on Sub-distribution AUC
lines(eval_times, roc_fg$AUC_2, type="b", col="red", lwd=2)

# Plot Cox scores evaluated on Sub-distribution AUC
lines(eval_times, roc_cox$AUC_2, type="b", col="lightblue", lwd=2, lty=2)

lines(eval_times, roc_fg$AUC_1, type="b", col="orange", lwd=2, lty = 2)

legend("bottomleft", 
       legend=c("Cox Scores evaluated by Cause-Specific AUC", 
                "Fine-Gray Scores evaluated by Sub-dist AUC",
                "Fine-Gray Scores evaluared by Cause-Specific AUC",
                "Cox Scores evaluated by Sub-dist AUC"),
       col=c("blue", "red", "orange","lightblue"), lty=c(1, 1, 2, 2), lwd=2, cex=0.8)

# ============================
# Compare To Deterministic Bounds
# ============================

set.seed(42)

generate_data <- function(n, cr_rate_multiplier) {
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  
  true_marker <- X1 + X2
  rate_cause1 <- exp(4 * true_marker) 
  T1 <- T1 <- exp(-true_marker)
  
  # Competing Event
  rate_cause2 <- exp(cr_rate_multiplier)
  T2 <- rexp(n, rate = rate_cause2)
  
  C <- rexp(n, rate = 0.5)
  
  # No CR, No Censoring
  time_true <- pmax(T1, 1e-5) # Floor at 0.00001 to prevent absolute zeroes
  event_true <- rep(1, n)
  
  # CR + Censoring
  time_obs <- pmax(pmin(T1, T2, C), 1e-5)
  event_obs <- ifelse(T1 < T2 & T1 < C, 1, 
                      ifelse(T2 < T1 & T2 < C, 2, 0))
  
  return(data.frame(time_obs, event_obs, X1, X2, true_marker, time_true, event_true))
}

train <- generate_data(2500, 1.0)
test  <- generate_data(2500, 1.0)

# Train Models + Risk Scores
cox_model <- coxph(Surv(time_obs, event_obs == 1) ~ X1 + X2, data = train)
fg_data <- finegray(Surv(time_obs, factor(event_obs)) ~ ., data = train)
fg_model <- coxph(Surv(fgstart, fgstop, fgstatus) ~ X1 + X2, weight = fgwt, data = fg_data)

score_oracle <- test$true_marker
score_cox <- predict(cox_model, newdata = test, type = "risk")
score_fg  <- predict(fg_model, newdata = test, type = "risk")

eval_times <- seq(0.2, 1.5, by = 0.2)

# Evaluate Models
roc_baseline <- timeROC(T = test$time_true, delta = test$event_true, marker = score_oracle, 
                        cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)
roc_oracle_real <- timeROC(T = test$time_obs, delta = test$event_obs, marker = score_oracle, 
                           cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)
roc_cox_real <- timeROC(T = test$time_obs, delta = test$event_obs, marker = score_cox, 
                        cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)
roc_fg_real <- timeROC(T = test$time_obs, delta = test$event_obs, marker = score_fg, 
                       cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)

# Plot
plot(eval_times, roc_baseline$AUC, type="l", col="black", lwd=4, ylim=c(0.875, 1.01),
     ylab="Time-Dependent AUC", xlab="Time Horizon (t)", 
     main="Oracle vs. Cox vs. Fine-Gray under Competing Risks")

# Cause-Specific AUCs
lines(eval_times, roc_oracle_real$AUC_1, type="b", col="gray40", lwd=2, lty=2)
lines(eval_times, roc_cox_real$AUC_1, type="b", col="blue", lwd=2, lty=2)
lines(eval_times, roc_fg_real$AUC_1, type="b", col="green3", lwd=2, lty=2)

# Sub-distribution AUCs
lines(eval_times, roc_oracle_real$AUC_2, type="b", col="gray40", lwd=2, lty=1)
lines(eval_times, roc_cox_real$AUC_2, type="b", col="blue", lwd=2, lty=1)
lines(eval_times, roc_fg_real$AUC_2, type="b", col="green3", lwd=2, lty=1)

legend("bottomleft", 
       legend=c("True Biological Baseline (Oracle, No CR/Censoring)", 
                "Cause-Specific (Oracle)", "Cause-Specific (Cox)", "Cause-Specific (Fine-Gray)",
                "Sub-distribution (Oracle)", "Sub-distribution (Cox)", "Sub-distribution (Fine-Gray)"),
       col=c("black", "gray40", "blue", "green3", "gray40", "blue", "green3"), 
       lty=c(1, 2, 2, 2, 1, 1, 1), lwd=c(4, 2, 2, 2, 2, 2, 2), cex=0.7)

