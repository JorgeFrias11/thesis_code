###############################################################################
# Descriptive statistics and data visualization of the main data
###############################################################################

packages_used <- c("dplyr", "lubridate", "ggplot2")

for (package_used in packages_used) {
  if (!require(package_used, character.only = TRUE)) {
    install.packages(package_used)
    library(package_used, character.only = TRUE)
  }
}

setwd(Sys.getenv("THESIS_DATA_DIR"))
coindata <- readRDS("data/coins_data.rds")

# Count observations per asset (on filtered data)
observationsPerCoin <- coindata %>%
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


assets_per_year <- coindata %>%
  mutate(year = year(date)) %>%
  group_by(year) %>%
  summarise(n_assets = n_distinct(coinName), .groups = "drop")

# Plot
ggplot(assets_per_year, aes(x = factor(year), y = n_assets)) +
  geom_col() +
  geom_text(aes(label = n_assets), vjust = -0.5, size = 3.5) +
  labs(
    x = "Year",
    y = "Number of unique coins"
  ) +
  scale_y_continuous(
    breaks = seq(0, 1400, 250),
    limits = c(0, 1400)
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x = element_text(color = "black"),
    axis.text.y = element_text(color = "black"), 
    axis.title.x = element_text(size = 10),  # axis labels size
    axis.title.y = element_text(size = 10)   
  )

# ANtes, size 5 y base size =16

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



# Plot time series of number of coins
assets_per_day <- coindata %>%
  group_by(date) %>%
  summarise(n_assets = n_distinct(coinName), .groups = "drop")

# There is an outlier in this date, likely an error in the data. 
assets_per_day <- assets_per_day %>%
  mutate(n_assets = ifelse(date == as.Date("2020-09-17"), lag(n_assets), n_assets))

ggplot(assets_per_day, aes(x = date, y = n_assets)) +
  geom_line(color = "steelblue", linewidth = 1) +
  scale_x_date(
    date_breaks = "1 year",
    date_labels = "%Y"
  ) +
  scale_y_continuous(
    breaks = seq(0, max(assets_per_day$n_assets), by = 200)
  ) +
  labs(
    x = "Year",
    y = "Number of unique coins",
  ) +
  theme_bw(base_size = 12) +
  theme(
    axis.text.x = element_text(color = "black"),
    axis.text.y = element_text(color = "black"), 
    axis.title.x = element_text(size = 10),  # axis labels size
    axis.title.y = element_text(size = 10)   
  )

length(unique(coindata$coinName))

###############################################################################
# Descriptive statistics and data visualization of the filtered data
###############################################################################

# Panel of characteristics
coindata <- readRDS("data/daily_predictors.rds")

coindata <- coindata %>%
  select(-mcap, -open, -close, -high, -low, -mktret, -rf, -cmkt, 
         -logvol, -nsi, -GPRD, -GPRD_MA7, -GPRD_MA30)

coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(across(
    .cols = -ret_excess,       
    .fns  = ~ lag(.x, 1)      
  )) %>%
  ungroup() %>%
  rename(r2_1 = ret)           

coindata <- remove_missing(coindata)

dim(coindata)
#######################################
# MarketCap plot
######################################

mktcap <- read.csv("CoinGecko-GlobalCryptoMktCap.csv")
mktcap$snapped_at <- as.POSIXct(mktcap$snapped_at / 1000, 
                                origin = "1970-01-01", 
                                tz = "UTC")
names(mktcap)[1] <- "date"

head(coindata)



