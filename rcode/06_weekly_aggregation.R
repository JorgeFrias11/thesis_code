# Code 4.1 to later include in the 4_predictors script
# Aggregate the predictors on a weekly basis

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
    # Give weekly format: 1st week - first 7 days
    day_of_year = yday(date),
    days_in_year = if_else(leap_year(date), 366, 365),
    ww = case_when(
      day_of_year <= 357 ~ ceiling(day_of_year / 7),
      TRUE ~ 52),
    yyyyww = yyyy * 100 + ww
  ) %>%
  dplyr::select( -day_of_year, -days_in_year, -yyyy, -ww)


# 2: Compute weekly return for each coin
weekly_returns <- coindata %>%
  group_by(coinName, yyyyww) %>%
  summarize(
    ret_w = prod(1 + ret, na.rm = TRUE) - 1,
    .groups = "drop"
  )

cat("Count of extreme outliers: \n")
sum(weekly_returns$ret_w > 5, na.rm = T)  # 360 extreme pos. returns
sum(weekly_returns$ret_w < -.99, na.rm = T)  # 38 extreme neg. returns

# Trim ourliers
weekly_returns <- weekly_returns %>%         
  mutate(ret_w = pmin(pmax(ret_w, -0.99), 5.0))  # trims to [-99%, +500%]

# 3: Compute weekly risk-free rate (last available each week)
rf_weekly <- coindata %>%
  group_by(yyyyww) %>%
  summarize(
    rf = dplyr::last(na.omit(rf)),
    .groups = "drop"
  )

# 4: Compute weekly excess return per coin
weekly_excess_ret <- weekly_returns %>%
  left_join(rf_weekly, by = "yyyyww") %>%
  mutate(ret_excess = ret_w - rf)

# 5: Create market excess return factor
cmkt_daily <- coindata %>%
  ungroup() %>%
  select(date, cmkt, yyyyww) %>%
  distinct(date, .keep_all = TRUE) %>%
  arrange(date)

cmkt_weekly <- cmkt_daily %>%
  group_by(yyyyww) %>%
  summarize(
    mktret_w = prod(1 + cmkt, na.rm = TRUE) - 1,
    .groups = "drop"
  ) %>%
  left_join(rf_weekly, by = "yyyyww") %>%
  mutate(cmkt = mktret_w - rf) %>%
  select(yyyyww, cmkt)

# Step 6: Aggregate all characteristics
exclude_cols <- c("date", "ret", "rf", "cmkt", "ret_excess",
                  "maxret", "minret", "yyyyww", "yyyy", "mm", "ww",
                  "open", "close", "high", "low")

cat("Computing weekly aggregation...\n")

weekly_data <- coindata %>%
  group_by(coinName, yyyyww) %>%
  summarize(
    # Safe max/min aggregation
    maxret = if (all(is.na(maxret))) NA_real_ else max(maxret, na.rm = TRUE),
    minret = if (all(is.na(minret))) NA_real_ else min(minret, na.rm = TRUE),
    
    # Mean for other numeric columns
    across(
      .cols = !any_of(exclude_cols),
      .fns = ~mean(.x, na.rm = TRUE)
    ),
    .groups = "drop"
  )

cat("Done!\n")

# Step 6: Combine all into a single panel
weekly_panel <- weekly_data %>%
  left_join(weekly_excess_ret, by = c("coinName", "yyyyww")) %>%
  relocate(coinName, yyyyww, ret_w, rf, ret_excess, cmkt)  # tidy column order

cat("Saving data...\n")
saveRDS(weekly_panel, "data/weeklypredictors.rds")



