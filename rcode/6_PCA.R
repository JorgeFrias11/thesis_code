setwd(Sys.getenv("THESIS_WORKDIR"))

libraries_used <- c("tidyr", "dplyr", "stats", "factoextra", "dfms", "xts")

for (library_used in libraries_used) {
  if (!require(library_used, character.only = TRUE)) {
    install.packages(library_used)
    library(library_used, character.only = TRUE)
  }
}

data <- readRDS("data/valwei_factors.rds")

summary(data)
cat("Number of rows:", nrow(data), "\n")

factors_df <- na.omit(data[-1])
cat("Number of rows without NAs:", nrow(factors_df), "\n")

pca_result <- prcomp(factors_df, scale. = TRUE, center = TRUE)

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


