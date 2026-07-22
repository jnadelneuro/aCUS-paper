library(fastFMM)

patch_fastFMM_parallel <- function() {
  message("Applying hotfix to fastFMM:::var_analytic...")
  
  va_fun <- fastFMM:::var_analytic
  va_src <- deparse(body(va_fun), width.cutoff = 500)
  
  # Find the line with the randintercept Z assignment and insert Z <- NULL before it
  target <- '    if (!randintercept) {'
  hit <- which(va_src == target)
  
  # There are two such blocks; we want the first one (around line 53 of body)
  # specifically the one followed by Z <- data_cov$Z_orig
  z_block <- which(va_src == '        Z <- data_cov$Z_orig')
  
  if (length(z_block) == 0) {
    warning("Patch Failed: Could not find Z assignment block. Version might differ.")
    return()
  }
  
  # The if (!randintercept) immediately before the Z assignment
  insert_after <- hit[which.min(abs(hit - z_block))] - 1
  
  va_src_patched <- c(
    va_src[1:insert_after],
    '    Z <- NULL',
    va_src[(insert_after + 1):length(va_src)]
  )
  
  body(va_fun) <- parse(text = paste(va_src_patched, collapse = "\n"))[[1]]
  assignInNamespace("var_analytic", va_fun, ns = "fastFMM")
  
  message("Success: Z <- NULL inserted before randintercept branch. parLapply can now evaluate Z = Z safely.")
}

patch_fastFMM_parallel()

# ======================================

# Proceed with your analysis
# ...
library(lme4)
library(parallel)
library(cAIC4)
library(magrittr)
library(dplyr)
library(mgcv) 
library(MASS)
library(lsei) 
library(refund)
library(stringr) 
library(Matrix) 
library(mvtnorm) 
#library(arrangements) 
library(progress) 
library(ggplot2)
library(gridExtra)
library(Rfast)
library(arrow)
library(feather)
library(data.table)
library(tidyr)
library(doParallel)
library(foreach)

rm(list=ls())
gc()

# ==============================================================================
# CONFIGURATION & FUNCTIONS
# ==============================================================================

# Set Base Directory
source("_config.R")
base_dir <- RI60_DIR
setwd(base_dir)

# Function: Clean Data via CSV (Required hack for FUI)
clean_via_csv <- function(df, filename_base) {
  fname <- paste0(filename_base, ".csv")
  write.csv(df, file = fname, row.names = FALSE)
  
  read_df <- read.csv(fname)
  # Filter NAs as per original script logic
  if("timeSince_ReNP" %in% names(read_df)){
    read_df <- read_df[!is.na(read_df$timeSince_ReNP), ]
  }
  return(read_df)
}

# Function: ROUT Outlier Removal (Session-Based)
# 1. Calculates mean per session (Mouse x Day)
# 2. Identifies outlier sessions using ROUT (Median/MAD + FDR)
# 3. Removes ALL trials for those sessions
remove_rout_outliers <- function(data, col_name, Q = 0.001) {
  if (!col_name %in% names(data)) return(data)
  
  # Ensure we have session identifiers
  if(!all(c("mouse", "dayOnType") %in% names(data))) {
    warning("Cannot perform session-based outlier removal: 'mouse' or 'dayOnType' columns missing.")
    return(data)
  }
  
  # 1. Summarize by Session (Mouse + Day)
  # We use dplyr::select to avoid conflict with MASS::select
  session_summary <- data %>%
    group_by(mouse, dayOnType) %>%
    summarise(mean_val = mean(.data[[col_name]], na.rm = TRUE), .groups = "drop") %>%
    filter(!is.na(mean_val))
  
  vals <- session_summary$mean_val
  
  # 2. Calculate Robust Statistics (Median and MAD)
  med_val <- median(vals, na.rm = TRUE)
  mad_val <- mad(vals, constant = 1.4826, na.rm = TRUE)
  
  if (mad_val == 0) {
    message(paste("MAD is 0 for", col_name, "- skipping outlier removal."))
    return(data)
  }
  
  # Calculate Robust Z-scores
  z_scores <- abs(vals - med_val) / mad_val
  
  # Calculate P-values (Two-tailed)
  p_vals <- 2 * (1 - pnorm(z_scores))
  
  # Apply FDR (Benjamini-Hochberg)
  is_outlier <- rep(FALSE, length(vals))
  valid_idx <- !is.na(p_vals)
  
  if (sum(valid_idx) > 0) {
    p_adj <- p.adjust(p_vals[valid_idx], method = "BH")
    is_outlier[valid_idx] <- p_adj < Q
  }
  
  # 3. Identify Outlier Sessions
  outlier_sessions <- session_summary[is_outlier, ]
  
  # Report and Filter
  n_sessions_removed <- nrow(outlier_sessions)
  
  if (n_sessions_removed > 0) {
    message(paste("ROUT (Q=", Q*100, "%): Removed", n_sessions_removed, "outlier session(s) for", col_name))
    
    # Create a composite key for filtering
    data <- data %>%
      mutate(session_id = paste(mouse, dayOnType, sep = "_"))
    
    outlier_ids <- paste(outlier_sessions$mouse, outlier_sessions$dayOnType, sep = "_")
    
    # Filter out all trials belonging to outlier sessions
    data_clean <- data %>% filter(!session_id %in% outlier_ids) %>% dplyr::select(-session_id)
    return(data_clean)
  }
  
  return(data)
}
run_fui_analysis <- function(data, fixed_formula, output_filename) {
  if (file.exists(output_filename)) {
    message(paste("File exists, skipping:", output_filename))
    return(NULL)
  }
  
  message(paste("Running FUI for:", output_filename))
  slurm_cores <- Sys.getenv("SLURM_CPUS_PER_TASK")
  
  if (slurm_cores == "") {
    n_cores <- 1L
  } else {
    n_cores <- as.numeric(slurm_cores)
  }
  # IMPORTANT: Tibble fix for fastFMM compatibility
  data <- as.data.frame(na.omit(data))
  
  full_formula <- as.formula(paste(fixed_formula, "+ (1 | mouse)"))
  
  # Run Model
  mod <- fui(formula = full_formula,
             data = data,
             analytic = TRUE, # Change to FALSE if you want bootstrap trust
             parallel = TRUE,
             n_cores = n_cores) # Keep FALSE to avoid "Object Z" bug with analytic=TRUE
  
  # Generate Plot Data
  plot_obj <- plot_fui(mod,
                       x_rescale = 51,
                       align_x = 2,
                       xlab = "Time (s)",
                       return = TRUE)
  
  # Save RDS
  saveRDS(plot_obj, file = output_filename)
}

# Function: run stress+naive FUI models with pooled z-scored predictor
# Does ROUT per group, computes pooled mean/SD across both post-ROUT groups,
# applies that same z-score to both groups, then fits each.
run_zscored_pair <- function(df_stress, df_naive, col_name, region_label, target_event, Q_rout = 0.001) {
  if (col_name == 'reward_rate') {
    df_stress <- df_stress[!is.nan(df_stress$reward_rate), ]
    df_naive  <- df_naive[!is.nan(df_naive$reward_rate), ]
  }
  # ROUT Outlier Removal (Session-Based, Q=0.1%)
  df_stress_clean <- remove_rout_outliers(df_stress, col_name, Q = Q_rout)
  df_naive_clean  <- remove_rout_outliers(df_naive,  col_name, Q = Q_rout)
  
  # Pooled z-params from post-ROUT data, applied identically to both groups
  pooled_vals <- c(df_stress_clean[[col_name]], df_naive_clean[[col_name]])
  pooled_vals <- pooled_vals[is.finite(pooled_vals)]
  pooled_mean <- mean(pooled_vals, na.rm = TRUE)
  pooled_sd   <- sd(pooled_vals, na.rm = TRUE)
  if (is.na(pooled_sd) || pooled_sd == 0) {
    warning(paste("Pooled SD is 0 or NA for", col_name, "in", region_label, "- skipping."))
    return(invisible(NULL))
  }
  message(sprintf("  [%s %s] pooled mean=%.4f sd=%.4f", region_label, col_name, pooled_mean, pooled_sd))
  df_stress_clean[[col_name]] <- (df_stress_clean[[col_name]] - pooled_mean) / pooled_sd
  df_naive_clean[[col_name]]  <- (df_naive_clean[[col_name]]  - pooled_mean) / pooled_sd
  
  formula_str <- paste0("trimTrace ~ ", col_name)
  out_stress <- paste0(target_event, "_FMM_stress_", region_label, "_", col_name, "_zscored.rds")
  out_naive  <- paste0(target_event, "_FMM_naive_",  region_label, "_", col_name, "_zscored.rds")
  run_fui_analysis(df_stress_clean, formula_str, out_stress)
  run_fui_analysis(df_naive_clean,  formula_str, out_naive)
}
# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================

# photoDF <- arrow::read_feather('photoDF_R.feather')
photoDF <- arrow::read_feather('photoDF_R_with_weights.feather')
target_event <- 'UnNP'

# Basic cleaning and filtering
photoDF <- photoDF %>%
  filter(event == target_event, sesType == 'RI60') %>%
  mutate(
    group = case_when(
      group == "naive" ~ 0,
      group == "stress" ~ 1,
      TRUE ~ as.numeric(group)
    ),
    sex = case_when(
      sex == "M" ~ 0,
      sex == "F" ~ 1,
      TRUE ~ as.numeric(sex)
    )
  )

# Split by recording location
TS_photoDF <- photoDF[photoDF$recordingLoc == 'TS', ]
DMS_photoDF <- photoDF[photoDF$recordingLoc == 'DMS', ]

# Downsample trace (keep every 2nd point)
TS_photoDF$trimTrace <- lapply(TS_photoDF$trimTrace, function(x) x[seq(1, length(x), by = 2)])
DMS_photoDF$trimTrace <- lapply(DMS_photoDF$trimTrace, function(x) x[seq(1, length(x), by = 2)])

# Remove unnecessary columns to save memory
drop_cols <- c('recordingLoc', 'event', 'sesType', 'trimTime')
TS_photoDF <- TS_photoDF[, !(names(TS_photoDF) %in% drop_cols)]
DMS_photoDF <- DMS_photoDF[, !(names(DMS_photoDF) %in% drop_cols)]

# Explode trace columns
TS_exploded <- TS_photoDF %>% unnest_wider(trimTrace, names_sep = "_")
DMS_exploded <- DMS_photoDF %>% unnest_wider(trimTrace, names_sep = "_")

# Create Subsets (Stress vs Naive) and Filter Time
# Helper to filter time range [0, 200]
filter_time <- function(df) {
  df[!is.na(df$timeSince_ReNP) & df$timeSince_ReNP >= 0 & df$timeSince_ReNP <= 200, ]
}

df_TS_stress <- filter_time(TS_exploded[TS_exploded$group == 1, ])
df_TS_naive  <- filter_time(TS_exploded[TS_exploded$group == 0, ])
df_DMS_stress <- filter_time(DMS_exploded[DMS_exploded$group == 1, ])
df_DMS_naive  <- filter_time(DMS_exploded[DMS_exploded$group == 0, ])

# Calculate Diff from 60
df_TS_stress$time_diff_from_60 <- abs(df_TS_stress$timeSince_ReNP - 60)
df_TS_naive$time_diff_from_60  <- abs(df_TS_naive$timeSince_ReNP - 60)
# Calculate Diff from 60
df_DMS_stress$time_diff_from_60 <- abs(df_DMS_stress$timeSince_ReNP - 60)
df_DMS_naive$time_diff_from_60  <- abs(df_DMS_naive$timeSince_ReNP - 60)

# ==============================================================================
# CSV WRITE/READ CYCLE (CRITICAL STEP)
# ==============================================================================

df_TS         <- clean_via_csv(TS_exploded, "TS_photoDF")
df_DMS        <- clean_via_csv(DMS_exploded, "DMS_photoDF")

df_TS_stress  <- clean_via_csv(df_TS_stress, "df_TS_stress")
df_TS_naive   <- clean_via_csv(df_TS_naive, "df_TS_naive")

df_DMS_stress <- clean_via_csv(df_DMS_stress, "df_DMS_stress")
df_DMS_naive  <- clean_via_csv(df_DMS_naive, "df_DMS_naive")

# ==============================================================================
# MODELING
# ==============================================================================
# Loop through predictors, pooled-z-scoring each across stress + naive so that
# the β(t) curves from the two group-specific models are directly comparable.
colnames <- c('pokeRate', 'entryRate', 'dayOnType', 'reward_rate', 'cumulative_poke', 'pi_0', 'pi_1', 'pi_2', 'pi_3', 'pi_4')
colnames <- c('reward_rate')

for (cn in colnames) {
  message(paste("=== Predictor:", cn, "==="))
  run_zscored_pair(df_TS_stress,  df_TS_naive,  cn, "TS",  target_event)
  run_zscored_pair(df_DMS_stress, df_DMS_naive, cn, "DMS", target_event)
}

# ==============================================================================
# EXPORT RDS RESULTS TO CSV
# ==============================================================================
# Define directory explicitly (or reuse base_dir if that was the intent)
rds_directory <- file.path(base_dir, "FMM_Models") 

if(dir.exists(rds_directory)) {
  rds_files <- list.files(path = rds_directory, pattern = "\\.rds$", full.names = TRUE)
  
  for (file in rds_files) {
    data <- readRDS(file)
    file_name <- tools::file_path_sans_ext(basename(file))
    
    # Iterate through objects in the RDS list
    for (name in names(data)) {
      df <- data[[name]]
      safe_name <- gsub("[^[:alnum:]_]", "_", name)
      csv_file_name <- file.path(rds_directory, paste0(file_name, "_", safe_name, ".csv"))
      
      # Check if object is data frame/matrix AND file doesn't exist
      if ((is.data.frame(df) || is.matrix(df)) && !file.exists(csv_file_name)) {
        write.csv(df, file = csv_file_name, row.names = FALSE)
      }
    }
  }
} else {
  warning(paste("Directory not found for final CSV export:", rds_directory))
}
