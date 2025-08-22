
packages_used <- c("dplyr", "readxl", "slider", "purrr", "moments", 
                   "quantmod", "bidask", "tidyr", "PerformanceAnalytics")

# for (package_used in packages_used) {
#   if (!require(package_used, character.only = TRUE)) {
#     #install.packages(package_used)
#     library(package_used, character.only = TRUE)
#   }
# }

Sys.setenv("R_LIBS_USER"="~/R_LIBS")

for (package_used in packages_used) {
  library(package_used, character.only = TRUE, lib.loc = "~/R_LIBS/")
}

setwd(Sys.getenv("THESIS_DATA_DIR"))
cat("Current working directory: \n")
getwd()


start_time <- Sys.time()

#######################################################################
#  Only edit these three variables
#######################################################################

cat("Reading file...", "\n")

coindata <- readRDS(file="data/coins_data_100mill.rds")
predictors <- readRDS(file="data/daily_predictors_100mill.rds")

head(predictors$prc)
dim(predictors)
#output_file <- "predictors2.rds"

cat("Constructing predictors...", "\n")

cat("Constructing size (1/7)...", "\n")

coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(prc = log(close))


cat("Size done...", "\n")

cat("Constructing Momentum (2/7)...", "\n")


coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(r14_0 = slide_dbl(.x = ret,
                           .f = PerformanceAnalytics::Return.cumulative,
                           .before = 13L,
                           .complete = TRUE))

predictors$prc = coindata$prc
head(predictors$prc)

predictors <- predictors %>%
  bind_cols(r14_0 = coindata$r14_0) %>%
  relocate(r14_0, .after = r7_0)

head(predictors)
dim(predictors)

saveRDS(predictors, "data/daily_predictors_100mill.rds")

