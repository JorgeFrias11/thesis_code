### ###########################################################################
# This code constructs factors following Fama-French based on weekly returns, 
# It constructs quintile portfolios that are rebalanced weekly. 
###############################################################################

libraries_used <- c("dplyr", "tidyr", "purrr", "lubridate")

for (library_used in libraries_used) {
  if (!require(library_used, character.only = TRUE)) {
    install.packages(library_used)
    library(library_used, character.only = TRUE)
  }
}

setwd(Sys.getenv("THESIS_WORKDIR"))

data <- readRDS("data/predictors_uncleaned.rds")

# Create year-week indicator
coindata <- data %>%
  mutate(
    yyyy = year(date),
    mm = month(date),
    #dd = day(date),
    #yyyymmdd = as.numeric(format(date, "%Y%m%d")),
    # Give weekly format: 1st week - first 7 days
    day_of_year = yday(date),
    days_in_year = if_else(leap_year(date), 366, 365),
    ww = case_when(
      day_of_year <= 357 ~ ceiling(day_of_year / 7),
      TRUE ~ 52),
    yyyyww = yyyy * 100 + ww
  ) %>%
  dplyr::select( -day_of_year, -days_in_year)

# Remove unnecesary predictors (match python IPCA
cols_drop <- c("ret_excess", "volume", "maxdprc", "prcvol", "stdprcvol", "GPRD", 
               "GPRD_MA7", "GPRD_MA30", "std_turn", "volsh_60d", "beta2",
               "open", "close", "high", "low", "mktret", "cmkt")  # extras

               
# Filter 
coindata <- coindata %>%
  select(-all_of(cols_drop)) 

## TEST
#coindata %>% select(date, coinName, volume, ret, mcap, yyyy, mm, ww, yyyyww)
#colnames(coindata)

first20 <- unique(coindata$coinName)[1:20]
coins20 <- coindata %>%
  filter(coinName %in% first20)
coins20

coindata <- coins20   # Replace to test
###############################################################################
# Construct the factors based on Fama-French
###############################################################################

# 1: Compute weekly return and market cap
weekly_ret_mcap <- coindata %>%
  group_by(coinName, yyyyww) %>%
  summarize(
    ret_w = prod(1 + ret, na.rm = TRUE) - 1,
    marketcap = last(marketcap),
    .groups = "drop"
  )

# 2: all characteristics (columns)
char_cols <- c(
  "lvol", "mcap", "prc", "r2_1", "dh90", "r7_1", "r14_1", "r21_1", "r30_1", "r30_14", "r180_60",
  "alpha", "beta", "ivol", "rvol", "retvol", "var", "delay", "bidask", "illiq",
  "turn", "cv_vol", "sat", "volsh_30d", "dto", "skew", "kurt", "maxret", "minret", "nsi"
)


# Get last value of each characteristic at each week
weekly_chars <- coindata %>%
  group_by(coinName, yyyyww) %>%
  summarize(across(all_of(char_cols), ~last(.x), .names = "{.col}"), .groups = "drop")

# 4: Merge with returns and compute next week's return
factor_data <- weekly_chars %>%
  inner_join(weekly_ret_mcap, by = c("coinName", "yyyyww")) %>%
  group_by(coinName) %>%
  arrange(yyyyww) %>%
  mutate(
    yyyyww_next = lead(yyyyww),
    ret_fwd = lead(ret_w),
    mcap_next = lead(marketcap)
  ) %>%
  ungroup()

# 5: Compute value-weighted long-short factor for one characteristic
compute_vw_factor <- function(data, char_name) {
  data <- data %>%
    filter(!is.na(.data[[char_name]]), !is.na(ret_fwd), !is.na(mcap_next), 
           !is.na(yyyyww_next)) %>%
    group_by(yyyyww) %>%
    mutate(
      group = ntile(.data[[char_name]], 5)
    ) %>%
    ungroup() %>%
    group_by(yyyyww_next) %>%
    summarize(
      high = if (sum(group == 5) > 0) weighted.mean(ret_fwd[group == 5], 
                                                    mcap_next[group == 5], 
                                                    na.rm = TRUE) else NA,
      low  = if (sum(group == 1) > 0) weighted.mean(ret_fwd[group == 1], 
                                                    mcap_next[group == 1], 
                                                    na.rm = TRUE) else NA,
      !!char_name := high - low,
      .groups = "drop"
    ) %>%
    rename(yyyyww = yyyyww_next) %>%  # Now this is safe
    dplyr::select(yyyyww, !!char_name)
  
  return(data)
}



# 6: Apply to all characteristics
cat("Computing long-short factors...")
factor_list <- map(char_cols, ~compute_vw_factor(factor_data, .x))

# 7: Merge all factors into one table
factors_merged <- reduce(factor_list, full_join, by = "yyyyww") %>%
  arrange(yyyyww)

cat("Done!")

factors_merged
summary(factors_merged)

#saveRDS(factors_merged, "data/valwei_factors.rds")