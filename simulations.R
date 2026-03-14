library(survival)
library(timeROC)
library(survC1)

# ==================
# Oracle Model
# ==================
# This simulation compares the different AUC definitions using the marker as a predictor
# for the risk, without running any models. We compute this for a high-competing 
# risk and a low-competing risk,and also show the AUC of the same data where no 
# competing risk occured

set.seed(42)

n <- 20000
X <- rnorm(n) 

# Simulate Primary Event and Censoring
rate_cause1 <- exp(2 * X) 
T1 <- rexp(n, rate = rate_cause1)
C <- rexp(n, rate = 0.1)
eval_times <- seq(0.5, 3, by = 0.25)

# Baseline (No Competing Risks) ---
time_true <- pmin(T1, C)
event_true <- ifelse(T1 < C, 1, 0)
roc_true <- timeROC(T = time_true, delta = event_true, marker = X, 
                    cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)

# Compute roc for a simulated competing risk
simulate_cr_scenario <- function(cr_rate_multiplier) {
  # Simulate Competing Event (T2) and new time
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
# This simulation has T1 as an exponential with a function of X1 as a rate, and 
# T2 with a function of X2 as a rate. We compute Sub-distribution and Cause-specific AUCs
# for both Cox and Fine-Gray models, and then as a reference add a line to show the 
# Cause-Specific AUC for a Cox model where the competing risk does not occur

set.seed(42)

generate_cohort <- function(n, cr_rate_multiplier) {
  # Generate markers and data
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  rate_cause1 <- exp(1.5 * X1)
  T1 <- rexp(n, rate = rate_cause1)
  # Competing Event unrelated to markers
  rate_cause2 <- exp(cr_rate_multiplier*X2)
  T2 <- rexp(n, rate = rate_cause2)
  C <- rexp(n, rate = 0.1)
  time_true <- pmin(T1, C)
  time <- pmin(T1, T2, C)
  event <- ifelse(T1 < T2 & T1 < C, 1, 
                  ifelse(T2 < T1 & T2 < C, 2, 0))
  event_true <- ifelse(T1 < C, 1, 0)
  return(data.frame(time, event, X1, X2, time_true, event_true))
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

cox_model_true <- coxph(Surv(time_true, event_true == 1) ~ X1 + X2, data = train_data)

score_cox_true <- predict(cox_model, newdata = test_data, type = "risk")

eval_times <- seq(0.5, 3, by = 0.5)

roc_cox <- timeROC(T = test_data$time, delta = test_data$event, marker = score_cox, 
                   cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)
roc_fg <- timeROC(T = test_data$time, delta = test_data$event, marker = score_fg, 
                  cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)

roc_cox_true <- timeROC(T = test_data$time_true, delta = test_data$event_true, marker = score_cox_true, 
                   cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)

# Plot AUC curves
plot(eval_times, roc_cox$AUC_1, type="b", col="blue", lwd=2, ylim=c(0.5, 1.0),
     ylab="Time-Dependent AUC", xlab="Time Horizon (t)", 
     main="Cox vs. Fine-Gray Scores evaluated by AUC")

abline(h = 1.0, lty = 3, col = "black", lwd = 2)

# Plot Fine-Gray scores evaluated on Sub-distribution AUC
lines(eval_times, roc_fg$AUC_2, type="b", col="red", lwd=2)

# Plot Cox scores evaluated on Sub-distribution AUC
lines(eval_times, roc_cox$AUC_2, type="b", col="lightblue", lwd=2, lty=2)

lines(eval_times, roc_fg$AUC_1, type="b", col="orange", lwd=2, lty = 2)

# Plot Cox scores evaluated on Sub-distribution AUC
lines(eval_times, roc_cox_true$AUC, type="b", col="green", lwd=2, lty=3)


legend("bottomleft", 
       legend=c("Cox Scores evaluated by Cause-Specific AUC", 
                "Fine-Gray Scores evaluated by Sub-dist AUC",
                "Fine-Gray Scores evaluared by Cause-Specific AUC",
                "Cox Scores evaluated by Sub-dist AUC",
                "Cox Scores (data without competing risk)"),
       col=c("blue", "red", "orange","lightblue","green"), lty=c(1, 1, 2, 2,3), lwd=2, cex=0.8)

# ==============================
# Compare To Deterministic Bounds
# ==============================
# This experiment has T1 as a deterministic function of X1 and X2, while the competing 
# event depends only on X1. We see that the Cause-specific AUC of the Cox model here is = 1,
# while the sub-distribution AUC of the Cox model and both AUCs for the Fine-Gray model
# are below that.

set.seed(42)

generate_data <- function(n, cr_rate_multiplier) {
  X1 <- rnorm(n)
  X2 <- rnorm(n)
  # T1 is a deterministic function of X1 and X2
  T1 <- exp(X1+X2)
  
  # Competing Event
  rate_cause2 <- exp(2*X1)
  T2 <- rexp(n, rate = rate_cause2)
  
  C <- rexp(n, rate = 0.5)
  
  # CR + Censoring
  time_obs <- pmax(pmin(T1, T2, C), 1e-5)
  event_obs <- ifelse(T1 < T2 & T1 < C, 1, 
                      ifelse(T2 < C, 2, 0))
  
  return(data.frame(time_obs, event_obs, X1, X2))
}

train <- generate_data(2500, 1.0)
test  <- generate_data(2500, 1.0)

# Train Models + Risk Scores
cox_model <- coxph(Surv(time_obs, event_obs == 1) ~ X1 + X2, data = train)
fg_data <- finegray(Surv(time_obs, factor(event_obs)) ~ ., data = train)
fg_model <- coxph(Surv(fgstart, fgstop, fgstatus) ~ X1 + X2, weight = fgwt, data = fg_data)

score_cox <- predict(cox_model, newdata = test, type = "risk")
score_fg  <- predict(fg_model, newdata = test, type = "risk")

eval_times <- seq(0.2, 1.5, by = 0.1)

roc_cox_real <- timeROC(T = test$time_obs, delta = test$event_obs, marker = score_cox, 
                        cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)
roc_fg_real <- timeROC(T = test$time_obs, delta = test$event_obs, marker = score_fg, 
                       cause = 1, weighting = "marginal", times = eval_times, iid = FALSE)

# Plot
plot(eval_times, roc_cox_real$AUC_1, type="b", col="blue", lwd=2, lty=2, ylim=c(0.9, 1.01),
     ylab="Time-Dependent AUC", xlab="Time Horizon (t)", 
     main="Cox vs. Fine-Gray under Competing Risks w Perfect Marker")

# Cause-Specific AUCs
lines(eval_times, roc_fg_real$AUC_1, type="b", col="green3", lwd=2, lty=2)

# Sub-distribution AUCs
lines(eval_times, roc_cox_real$AUC_2, type="b", col="blue", lwd=2, lty=1)
lines(eval_times, roc_fg_real$AUC_2, type="b", col="green3", lwd=2, lty=1)

legend("bottomleft", 
       legend=c("Cause-Specific (Cox)", "Cause-Specific (Fine-Gray)",
                "Sub-distribution (Cox)", "Sub-distribution (Fine-Gray)"),
       col=c("blue", "green3", "blue", "green3"), 
       lty=c(2, 2, 1, 1), lwd=2, cex=0.7)

