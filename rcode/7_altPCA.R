setwd(Sys.getenv("THESIS_WORKDIR"))

packages_used <- c("tidyr", "dplyr")

for (pkg in packages_used) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

# Install package pcaMethods
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("pcaMethods")

library("pcaMethods", character.only = TRUE)


preds <- readRDS("data/weeklypredictors.rds")

returns <- preds %>%
  select(yyyyww, coinName, ret_excess)

returns_df <- preds %>%
  pivot_wider(names_from = coinName, values_from = ret_excess) %>%
  arrange(yyyyww)  # Ensure time is ordered



pc_f <- pcaMethods::pca(df, method = "nipals", nPcs = 7, scale = "uv")

###
data <- readRDS("data/valwei_factors.rds")
data <- data[-1]


pc_3 <- pcaMethods::pca(data, method = "nipals", nPcs = 3, scale = "uv")
pc_5 <- pcaMethods::pca(data, method = "nipals", nPcs = 5, scale = "uv")
pc_8 <- pcaMethods::pca(data, method = "nipals", nPcs = 8, scale = "uv")
###############################################################################
# Extract factors from the panel of returns
###############################################################################

data <- readRDS("data/predictors.rds")
data
# Remove the duplicate entries coin "float-protocol"

data <- data %>%
  filter(coinName != "float-protocol")

# Extract only excess returns and make a new df
returns_df <- data %>%
  select(date, coinName, ret_excess) %>%
  pivot_wider(names_from = coinName, values_from = ret_excess) 

names(returns_df) <- gsub("-", "_", names(returns_df))

returns_df <- returns_df[-1]

system.time(
  pc_ret <- pcaMethods::pca(returns_df, method = "nipals", 
                            nPcs = 7, scale = "uv")
)



pca_result <- prcomp(factors_df, scale. = TRUE)

summary(pca_result)

pca_eigplot <- fviz_eig(pca_result,
                        addlabels = TRUE,
                        ylim = c(0, 50)) 
pca_eigplot
#ggsave("plots/pca_eigplot.png", pca_eigplot)

# Define group_dict in R
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

pca_varplot <- fviz_pca_var(pca_result,
                         col.var = ordered_groups,
                         legend.title = "Factor Group",
                         repel = TRUE)

pca_varplot
# Save to file
#ggsave("plots/pca_varplot.png", pca_varplot)

