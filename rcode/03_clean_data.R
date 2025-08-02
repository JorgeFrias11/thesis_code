###############################################################################
# This code:
# - Applies filters to clean the full dataset.
# - Saves cleaned dataset. 
###############################################################################


libraries_used <- c("dplyr", "purrr", "zoo", "lubridate", "ggplot2")

for (library_used in libraries_used) {
  if (!require(library_used, character.only = TRUE)) {
    install.packages(library_used)
    library(library_used, character.only = TRUE)
  }
}

setwd(Sys.getenv("THESIS_DATA_DIR"))
 
 ###############################################################################
 # Load data and clean it
 ###############################################################################

coindata <- readRDS("data/coin_raw_data.rds")

summary(coindata)
nrow(coindata)  
length(unique(coindata$coinName))

###############################################################################
# Apply filters: remove small marketcap coins and coins without at least at 
# least 365 consecutive  observations, and cap outliers. Similar to Bianchi and 
# Babiak (2021). For coins with vol-to-mktrcap ratio > 1, remove these observations. 
# Other possibility is to I flag these
# observations as NA and replace them by their last valid value (if a last value
# is not available, take the next one). If the ratio is still higher than 1, 
# I remove the observation (fake or erroneous volume)
###############################################################################

# Apply filters
clean_data <- coindata %>%
  filter(marketcap >= 1e6) %>%           # remove small marketcap coins (observations)
  group_by(coinName) %>%
  mutate(
    #volume = if_else(volume / marketcap > 1, NA_real_, volume),
    # fill NAS in volume
    #volume = zoo::na.locf(volume, na.rm = FALSE),
    #volume = zoo::na.locf(volume, fromLast = TRUE, na.rm = FALSE),
    vm = volume / marketcap  # vol-to-mktcap ratio
  ) %>%
  filter(vm <= 1) %>%  # keep rows that satisfy the ratio <= 1
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
  filter(max_streak >= 365)

# Keep only coins with +365 consecutive observations
clean_data <- clean_data %>%
  semi_join(coins_streaks, by = "coinName")

# Calculate daily returns 
clean_data <- clean_data %>%
  group_by(coinName) %>%
  mutate(ret = close / lag(close) - 1)

cat("Count of extreme outlier returns: \n")
sum(clean_data$ret > 5, na.rm = T)  
sum(clean_data$ret < -.99, na.rm = T) 

# Trim ourliers
clean_data <- clean_data %>%         
  mutate(ret = pmin(pmax(ret, -0.99), 5.0))  # trims to [-99%, +500%]

cat("Total number of observations:", nrow(clean_data), "\n")                    
cat("Number of unique coins:", length(unique(clean_data$coinName)), "\n")
summary(clean_data)


saveRDS(clean_data, file = "data/coins_data.rds")
