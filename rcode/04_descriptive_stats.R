###############################################################################
# Descriptive statistics and data visualization
###############################################################################

setwd(Sys.getenv("THESIS_DATA_DIR"))
clean_data <- readRDS("data/coins_data.rda")

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