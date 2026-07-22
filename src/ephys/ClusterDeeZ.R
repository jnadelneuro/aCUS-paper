library(lme4)
library(ggplot2)
library(cluster)
library(mclust)
library(dplyr)
library(tidyr)
library(FactoMineR)
library(factoextra)
source("_config.R")

set.seed(15432)

setwd(file.path(INTRINSIC_DIR, "clustering"))
# Load dataset (assumed to be in a CSV file)
df <- read.csv("clusterPreppedBoi.csv")
df <- df[, !names(df) %in% c("X",'IR','mean_isi','isi_cv')]
# df <- df[, !names(df) %in% "X"]
original_data <- df

# Identify numeric electrophysiological features
firing_rate_features <- grep("FR_", names(df), value = TRUE)
other_features <- setdiff(names(df), c("cell.name", "mouse", "stress","sex", firing_rate_features))

# Standardize features
df[other_features] <- scale(df[other_features])
df[firing_rate_features] <- scale(df[firing_rate_features])

# Regress out stress effects using a linear mixed model
residuals_df <- df
for (feature in c(other_features, firing_rate_features)) {
  formula <- as.formula(paste(feature, "~ stress + (1 + stress | mouse)"))  # Random slope model
  model <- lmer(formula, data = df, REML = TRUE)
  residuals_df[[feature]] <- resid(model)
}

# Perform PCA on firing rate features
#pca_fr <- PCA(residuals_df[, firing_rate_features], scale.unit = TRUE, ncp = 3)
#fr_pca_components <- as.data.frame(pca_fr$ind$coord)
#names(fr_pca_components) <- paste0("FR_PC", 1:ncol(fr_pca_components))

# Replace firing rate features with PCA components
#residuals_df <- residuals_df[, !(names(residuals_df) %in% firing_rate_features)]
#residuals_df <- cbind(residuals_df, fr_pca_components)

# Perform PCA for overall dimensionality reduction
pca_overall <- PCA(residuals_df[, setdiff(names(residuals_df), c("cell.name", "mouse", "stress","sex"))], scale.unit = TRUE, ncp = 4)
principal_components <- as.data.frame(pca_overall$ind$coord)

# Fit Gaussian Mixture Model (GMM)
n_clusters <- 3  # Adjust based on evaluation metrics
kmeans_res <- kmeans(principal_components, centers = n_clusters, nstart = 25)
df$cluster <- as.factor(kmeans_res$cluster)  # Store cluster labels

pc_export <- principal_components %>%
  select(Dim.1, Dim.2) %>%
  mutate(cluster = df$cluster, cell.name = df$cell.name)

# Save to CSV
write.csv(pc_export, "PC1_PC2_clusters.csv", row.names = FALSE)

print("PC1 and PC2 coordinates with clusters saved to PC1_PC2_clusters.csv")

# Evaluate clustering with silhouette score
silhouette_score <- mean(silhouette(kmeans_res$cluster, dist(principal_components))[, 3])
print(paste("K-Means Silhouette Score:", round(silhouette_score, 3)))

# Save clustered data
write.csv(df, "clustered_cells.csv", row.names = FALSE)

# Visualize PCA results
ggplot(principal_components, aes(x = Dim.1, y = Dim.2, color = as.factor(df$cluster))) +
  geom_point() +
  labs(x = "PC1", y = "PC2", title = "PCA Projection with GMM Clusters", color = "Cluster") +
  theme_minimal()

# Compare cluster distribution
print(table(df$cluster))

# Compare key feature means across clusters
cluster_summary <- df %>% 
  group_by(cluster) %>% 
  summarise(across(where(is.numeric), mean, na.rm = TRUE))
print(cluster_summary)

# Boxplots for key features
firing_rate_features <- grep("^FR_", names(original_data), value = TRUE)  # Get all firing rate features
original_data <- original_data %>%
  left_join(df %>% select(cell.name, cluster), by = "cell.name")
firing_rate_long <- original_data %>%
  pivot_longer(
    cols = starts_with("FR_"),  # Select all firing rate columns
    names_to = "current",        # Create a column called "current"
    values_to = "firing_rate"    # Store values in "firing_rate"
  )

firing_rate_long <- firing_rate_long %>%
  mutate(current = as.numeric(gsub("FR_", "", current)))
ggplot(firing_rate_long, aes(x = current, y = firing_rate, color = as.factor(cluster), group = cluster)) +
  stat_summary(fun = mean, geom = "line", size = 1) +  # Plot mean firing rate per cluster
  stat_summary(fun.data = mean_cl_boot, geom = "ribbon", alpha = 0.2, aes(fill = as.factor(cluster))) +  # Confidence bands
  labs(x = "Current Injection (pA)", y = "Firing Rate (Hz)", color = "Cluster", fill = "Cluster",
       title = "Firing Rate vs. Current Injection Across Clusters") +
  theme_minimal()

for (feature in other_features) {
  p <- ggplot(original_data, aes(x = as.factor(cluster), y = .data[[feature]], fill = stress)) +
    geom_boxplot() +
    labs(title = paste("Comparison of", feature, "across Clusters"), x = "Cluster", y = feature, fill = "Stress Condition") +
    theme_minimal()
  
  print(p)
}


firing_rate_long$stress <- as.factor(firing_rate_long$stress)

sample_sizes <- firing_rate_long %>%
  group_by(cluster, stress) %>%
  summarise(n = n_distinct(cell.name), .groups = "drop")  # Count unique cells per group

# Merge n values back to use in legend labels
firing_rate_long <- left_join(firing_rate_long, sample_sizes, by = c("cluster", "stress"))

# Create legend labels with (n=X) appended
firing_rate_long$stress_n <- paste0(firing_rate_long$stress, " (n=", firing_rate_long$n, ")")

# Create separate plots for each cluster
plot_list <- list()

for (cl in unique(firing_rate_long$cluster)) {
  p <- ggplot(firing_rate_long %>% filter(cluster == cl), 
              aes(x = current, y = firing_rate, color = stress_n, group = stress_n)) +
    stat_summary(fun = mean, geom = "line", size = 1.2) +  # Thick mean line per stress group
    stat_summary(fun.data = mean_se, geom = "ribbon", alpha = 0.3, aes(fill = stress_n, color = NULL)) +  # SEM shading
    labs(x = "Current Injection (pA)", y = "Firing Rate (Hz)", 
         title = paste("Cluster", cl, "- Stress vs. Naive"),
         color = "Stress Group", fill = "Stress Group") +
    theme_minimal()
  
  plot_list[[paste0("Cluster_", cl)]] <- p  # Store plots in a list
}

# Print all plots
for (p in plot_list) {
  print(p)
}

# Check condition distribution across clusters
condition_cluster_table <- table(df$cluster, df$stress)
print(condition_cluster_table)
sex_cluster_table <- table(df$cluster, df$sex)
print(sex_cluster_table)

# Normalize by row for visualization
condition_cluster_table <- prop.table(condition_cluster_table, margin = 1)
condition_df <- as.data.frame(as.table(condition_cluster_table))
names(condition_df) <- c("Cluster", "Condition", "Proportion")

ggplot(condition_df, aes(x = as.factor(Cluster), y = Proportion, fill = Condition)) +
  geom_bar(stat = "identity", position = "fill") +
  labs(title = "Proportion of Stress/Naive in Each Cluster", x = "Cluster", y = "Proportion") +
  theme_minimal()

# Alternative clustering method: K-Means
kmeans_res <- kmeans(principal_components, centers = n_clusters)
kmeans_silhouette <- mean(silhouette(kmeans_res$cluster, dist(principal_components))[, 3])
print(paste("K-Means Silhouette Score:", round(kmeans_silhouette, 3)))

# Scree plot: Shows % variance explained by each PC
fviz_eig(pca_overall, addlabels = TRUE, ylim = c(0, 100)) +
  labs(title = "Scree Plot: Variance Explained by Each Principal Component")
# 
# # Extract variance explained by each PC
# var_explained <- pca_overall$eig[, 2]  # Percentage of variance explained
# cumulative_variance <- cumsum(var_explained)  # Cumulative sum
# 
# # Plot cumulative variance
# ggplot(data.frame(PC = 1:length(cumulative_variance), Variance = cumulative_variance), aes(x = PC, y = Variance)) +
#   geom_line() +
#   geom_point() +
#   labs(title = "Cumulative Variance Explained by Principal Components",
#        x = "Principal Component",
#        y = "Cumulative Variance (%)") +
#   theme_minimal()

# --- Pre/post stress-regression heatmaps for Prism ---
# pre  = z-scored features ; post = lmer residuals (exact PCA input from ClusterDeeZ.R)
library(lme4)

df <- read.csv("clusterPreppedBoi.csv")
df <- df[, !names(df) %in% c("X", "IR", "mean_isi", "isi_cv")]

fr    <- grep("^FR_", names(df), value = TRUE)
other <- c("RMP", "latency", "adaptation")
feats <- c(other, fr)

# step 1: z-score
dfz <- df
dfz[feats] <- scale(df[feats])

# step 2: regress out stress (identical models to the clustering script)
res <- dfz
for (f in feats) {
  m <- lmer(as.formula(paste(f, "~ stress + (1 + stress | mouse)")), data = dfz, REML = TRUE)
  res[[f]] <- resid(m)
}

# same column order for both panels: group by stress, then mouse
ord <- order(dfz$stress, dfz$mouse)
row_labels <- c("RMP", "Latency", "Adaptation", sub("FR_", "", fr))

export <- function(d, file) {
  m <- t(as.matrix(d[ord, feats]))
  colnames(m) <- d$cell.name[ord]
  write.csv(data.frame(Feature = row_labels, m, check.names = FALSE), file, row.names = FALSE)
}
export(dfz, "panelA_PRE_regression_prism.csv")
export(res, "panelA_POST_regression_prism.csv")

# column annotation (stress per column, in the same order) for the Prism color strip
write.csv(data.frame(cell.name = dfz$cell.name[ord], stress = dfz$stress[ord]),
          "panelA_column_stress_order.csv", row.names = FALSE)

