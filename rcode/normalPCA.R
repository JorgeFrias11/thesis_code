
setwd(Sys.getenv("THESIS_WORKDIR"))

libraries_used <- c("tidyr", "dplyr", "stats", "factoextra")

for (library_used in libraries_used) {
  if (!require(library_used, character.only = TRUE)) {
    install.packages(library_used)
    library(library_used, character.only = TRUE)
  }
}

data <- readRDS("data/predictors.rds")

# Choose reasonable time window to have enought observations
start_date <- as.Date("2018-06-01")
end_date <- as.Date("2025-04-30")

# Filter the data within the date window
trimdata <- data %>%
  filter(date >= start_date & date <= end_date)

n_dates <- length(unique(trimdata$date))

data_window <- trimdata %>%
  group_by(coinName) %>%
  filter(n_distinct(date) >= 0.95 * n_dates) %>%  # at least 95% obs available
  ungroup()

length(unique(data_window$coinName))   # 157 coins

# Extract only excess returns and make a new df
returns_df <- data_window %>%
  select(date, coinName, ret_excess) %>%
  pivot_wider(names_from = coinName, values_from = ret_excess) %>%
  fill(everything(), .direction = "down") %>%
  drop_na()

names(returns_df) <- gsub("-", "_", names(returns_df))

returns <- returns_df %>%
  select(-date) %>%
  as.matrix()

# Step 2: Run PCA
pca_result <- prcomp(returns, scale = TRUE)

head(pca_result$x)
head(pca_result$rotation)  # factor loadings

summary(pca_result)

fviz_eig(pca_result,
         addlabels = TRUE,
         ylim = c(0, 70))  # Scree plot. 
# PC1 (the first component) explains over 35% of the total variance

fviz_pca_var(pca_result, label = "none")  # Loadings
# Coins load positively on PC1. appears to represent a broad market component, 
# or common movement shared by most coins.
# The coin loadings are highly correlated with each other, pointing roughly in the same direction

#fviz_pca_ind(pca_result)  # Scores

# Save factors
dates <- returns_df$date
factors_df <- data_frame(date = dates, pca_result$x[, 1:3])




library(MASS)
data("biopsy")
head(biopsy)


# Flatten group_dict into variable-to-group named vector
group_dict <- list(
  core = c("mcap", "prc", "r14_1", "r21_1", "r30_1", "r30_14"),
  volrisk = c("beta", "ivol", "rvol", "retvol", "var", "delay"),
  activity = c("lvol", "volscaled", "turn", "std_vol", "cv_vol"),
  liquidity = c("bidask", "illiq", "sat", "dto", "volsh_30d"),
  pastreturns = c("r2_1", "r7_1", "r180_60", "dh90", "alpha"),
  distribution = c("skew", "kurt", "maxret", "minret"),
  news = c("nsi")
)

# Reverse the mapping: var name → group
var_to_group <- unlist(group_dict)  # names are lost
group_labels <- rep(names(group_dict), times = sapply(group_dict, length))
names(group_labels) <- var_to_group  # now: group_labels["mcap"] = "core"

# Reorder to match variable order in PCA
ordered_groups <- group_labels[colnames(factors_df)]








