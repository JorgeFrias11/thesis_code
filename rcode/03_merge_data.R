###############################################################################
# This code:
# - merges all coins' files and applies filters to clean the full dataset.
# - Saves the merged raw dataset and the cleaned one. 
# - Generates two plots to visualize the data. 
###############################################################################


libraries_used <- c("dplyr", "purrr", "zoo", "lubridate", "ggplot2")

for (library_used in libraries_used) {
  if (!require(library_used, character.only = TRUE)) {
    install.packages(library_used)
    library(library_used, character.only = TRUE)
  }
}

setwd(Sys.getenv("THESIS_WORKDIR"))
 
 ###############################################################################
 # Merging and saving data
 ###############################################################################
 
# coins.1 <- readRDS("data/rawData/1_coins_1_1000.rds")
# coins.2 <- readRDS("data/rawData/2_coins_1001_2000.rds")
# coins.3 <- readRDS("data/rawData/3_coins_2001_3000.rds")
# coins.4 <- readRDS("data/rawData/4_coins_3001_4000.rds")
# 
# missing_coins <- list.files("data/rawData", pattern = "^mis.*_coins_.*\\.rds$", 
#                             full.names = TRUE) %>%
#   map(readRDS) %>%
#   bind_rows()
# 
# coins4000 <- bind_rows(coins.1, coins.2, coins.3, coins.4,)
# 
# all_coins <- bind_rows(
#   coins.1,
#   coins.2,
#   coins.3,
#   coins.4,
#   missing_coins
# )

#saveRDS(coins4000, "data/raw_4000CoinsData.rds", compress = "gzip") # smaller set
#saveRDS(all_coins, "data/raw_CoinsData.rds", compress = "gzip")

### Load data and clean

coindata <- readRDS("data/raw_CoinsData.rds")

summary(coindata)
nrow(coindata)   # 3,592,292 observations
length(unique(coindata$coinName))    # 6370 coins

###############################################################################
# Apply filters: remove small marketcap coins and coins without at least at 
# least 365 consecutive  observations, and cap outliers. Similar to Bianchi and 
# Babiak (2021). For coins with vol-to-mktrcap ratio > 1, I flag these
# observations as NA and replace them by their last valid value (if a last value
# is not available, take the next one). If the ratio is still higher than 1, 
# I remove the observation (fake or erroneous volume)
###############################################################################

# Apply filters
clean_data <- coindata %>%
  filter(marketcap >= 1e6) %>%           # remove small marketcap coins
  group_by(coinName) %>%
  filter(date >= "2020-01-01") %>%    # for restricted dataset
  mutate(
    volume = if_else(volume / marketcap > 1, NA_real_, volume),
    volume = zoo::na.locf(volume, na.rm = FALSE),
    volume = zoo::na.locf(volume, fromLast = TRUE, na.rm = FALSE),
    vm = volume / marketcap  # recompute vol-to-mktcap AFTER volume fix
  ) %>%
  filter(vm <= 1) %>%  # remove coins violating this ratio
  ungroup() 

summary(clean_data$vm)   
clean_data$vm <- NULL    # remove the vm column


# Function to compute longest consecutive observations "streak"
longest_streak <- function(dates) {
  dates <- sort(unique(dates))
  streaks <- cumsum(c(1, diff(dates) != 1))
  max(tabulate(streaks))
}

# Compute longest streak. Select coins with more than 1 years of consecutive data
coins_streaks <- clean_data %>%
  group_by(coinName) %>%
  summarise(max_streak = longest_streak(as.integer(date)), .groups = "drop") %>%
  #filter(max_streak >= 730)
  filter(max_streak >= 365)

# Keep only coins with +365 consecutive observations
clean_data <- clean_data %>%
  semi_join(coins_streaks, by = "coinName")

# Calculate daily returns 
clean_data <- clean_data %>%
  group_by(coinName) %>%
  mutate(ret = close / lag(close) - 1)

cat("Count of extreme outliers: \n")
sum(clean_data$ret > 5, na.rm = T)  # 360 extreme pos. returns
sum(clean_data$ret < -.99, na.rm = T)  # 38 extreme neg. returns

# Trim ourliers
clean_data <- clean_data %>%         
  mutate(ret = pmin(pmax(ret, -0.99), 5.0))  # trims to [-99%, +500%]

nrow(clean_data)                       # 2,174,156 observations
length(unique(clean_data$coinName))    # 1592 coins - 1014 coins for 2 years
summary(clean_data)

#saveRDS(clean_data, "data/clean_CoinsData.rds")
saveRDS(clean_data, "data/CoinsData_2020sample.rds")

###############################################################################
# Visualize data
###############################################################################

clean_data <- readRDS("data/clean_CoinsData.rds")

# Count observations per asset (on filtered data)
observationsPerCoin <- clean_data %>%
  group_by(coinName) %>%
  summarise(n_obs = n(), .groups = "drop")

# Boxplot
boxplot(observationsPerCoin$n_obs,
        ylab = "Number of observations",
        notch = TRUE,
        col = "lightblue",
        boxwex = 0.3)
axis(side = 2, at = seq(0, max(observationsPerCoin$n_obs), by = 500))
# mtext("Figure 1: Distribution of the number of daily observations 
#       per cryptocurrency over the sample period",
#      side = 1, line = 2)


assets_per_year <- clean_data %>%
  mutate(year = year(date)) %>%
  group_by(year) %>%
  summarise(n_assets = n_distinct(coinName), .groups = "drop")

# Plot
ggplot(assets_per_year, aes(x = factor(year), y = n_assets)) +
  geom_col() +
  geom_text(aes(label = n_assets), vjust = -0.5, size = 6) +
  labs(
    x = "Year",
    y = "Number of coins"
  ) +
  scale_y_continuous(breaks = seq(0, 7) * 250) +
  theme_bw(base_size = 18)   # increases all text sizes simply


# ggplot(assets_per_year, aes(x = factor(year), y = n_assets)) +
#   geom_col() +
#   geom_text(aes(label = n_assets), vjust = -0.5, size = 4) +
#   labs(
#     x = "Year",
#     y = "Number of coins",
#     caption = "Figure 2: number of distinct cryptocurrencies for each year."
#   ) +
#   scale_y_continuous(breaks = seq(0, max(assets_per_year$n_assets), by = 250)) +
#   theme_bw(base_size = 13) +  # increases all text sizes simply
#   theme(plot.caption = element_text(hjust = 0.5, size = 12, vjust = -1.5))

