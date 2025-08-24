packages_used <- c("dplyr", "tidyr")

for (package_used in packages_used) {
  if (!require(package_used, character.only = TRUE)) {
    install.packages(package_used)
    library(package_used, character.only = TRUE, )
  }
}

setwd(Sys.getenv("THESIS_DATA_DIR"))

cat("Reading file...", "\n")
coindata <- readRDS(file="data/daily_predictors.rds")

#######################################################################
#  Market factor (CMKT) already calculated before
#######################################################################

cmkt_factor <- coindata %>% 
  ungroup() %>%
  select(date, cmkt) %>%
  distinct()

#######################################################################
#  Size factor (SMB)
#  Long-short portfolio: Long (short) on small (large) assets 
#######################################################################

smb_factor <- coindata %>%
  group_by(coinName) %>%
  arrange(date) %>%
  # use lagged (yesterday's) market cap
  mutate(marketcap_lag = dplyr::lag(marketcap)) %>%  
  ungroup() %>%
  filter(!is.na(ret), !is.na(marketcap)) %>%
  group_by(date) %>%
  # Sort by quintiles by market cap each day
  mutate(size_rank = ntile(marketcap, 5)) %>%
  group_by(date, size_rank) %>%
  # Value-weighted return within each quintile
  summarise(
    port_ret = sum(ret * marketcap, na.rm = TRUE) / sum(marketcap, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  # Wide format: one column per quintile
  tidyr::pivot_wider(names_from = size_rank, values_from = port_ret,
                     names_prefix = "Q") %>%
  # Long smallest coins (Q1) – short large coins (Q5)
  mutate(SMB = Q1 - Q5) %>%
  select(date, SMB)



#######################################################################
#  Momentum factor MOM following Liu et al. (2022)
#  r21_0 is 3 weeks return including today. We need to lag it so it takes returns
#  up to the last day: t-22 to t-1, and avoid look-ahead bias
#######################################################################

mom_factor <- coindata %>%
  filter(!is.na(ret), !is.na(marketcap), !is.na(r21_0)) %>%
  group_by(coinName) %>%
  arrange(date) %>%
  mutate(r21_lag = dplyr::lag(r21_0, 1)) %>%  # use lagged signal
  ungroup() %>%
  filter(!is.na(r21_lag)) %>%
  group_by(date) %>%
  # First split into size groups
  mutate(size_group = ifelse(marketcap <= median(marketcap, na.rm = TRUE), "Small", "Big")) %>%
  # Sort into momentum groups within each size group
  mutate(mom_group = ntile(r21_lag, 10),  
         mom_group = case_when(
           mom_group <= 3 ~ "Low",      # bottom 30%
           mom_group <= 7 ~ "Med",      # middle 40%
           TRUE ~ "High"                # top 30%
         )) %>%
  group_by(date, size_group, mom_group) %>%
  summarise(
    port_ret = sum(ret * marketcap, na.rm = TRUE) / sum(marketcap, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  tidyr::pivot_wider(names_from = c(size_group, mom_group),
                     values_from = port_ret) %>%
  mutate(
    CMOM = 0.5 * (Small_High + Big_High) - 0.5 * (Small_Low + Big_Low)
  ) %>%
  select(date, CMOM)

mom_factor

#######################################################################
#  Liquidity factor: long (short) on less liquid (more liquid) assets
#  A larger Amihud ratio represents lower liquidity 
#######################################################################

coindata <- coindata %>%
  mutate(
    vol_mill = volume / 1e6,       # convert to millions USD
    illiq = abs(ret) / vol_mill
  ) %>%
  group_by(coinName) %>%
  arrange(date) %>%
  mutate(
    illiq_lag = dplyr::lag(illiq)   # lag signal
  ) %>%
  ungroup()

liquid_factor <- coindata %>%
  filter(!is.na(ret), !is.na(marketcap), !is.na(illiq_lag)) %>%
  group_by(date) %>%
  mutate(illiq_q = ntile(illiq_lag, 5)) %>%    # 1=most liquid, 5=least liquid
  group_by(date, illiq_q) %>%
  summarise(
    port_ret = weighted.mean(ret, w = marketcap, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  pivot_wider(names_from = illiq_q, values_from = port_ret, names_prefix = "Q") %>%
  mutate(LIQ = Q5 - Q1) %>%    # long illiquid, short liquid
  select(date, LIQ)


#######################################################################
#  Volatility factor: long (short) on highest volatile (lowest volatile) assets
#  Use rvol_30d
#######################################################################

vol_factor <- coindata %>%
  group_by(coinName) %>%
  arrange(date) %>%
  mutate(rvol_30d_lag = dplyr::lag(rvol_30d)) %>%
  ungroup() %>%
  filter(!is.na(ret), !is.na(marketcap), !is.na(rvol_30d_lag)) %>%
  group_by(date) %>%
  mutate(vol_rank = ntile(rvol_30d_lag, 5)) %>%
  group_by(date, vol_rank) %>%
  summarise(
    port_ret = sum(ret * marketcap, na.rm = TRUE) / sum(marketcap, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  tidyr::pivot_wider(
    names_from = vol_rank, values_from = port_ret,
    names_prefix = "Q"
  ) %>%
  mutate(
    VOL = Q5 - Q1   # long high-volatility, short low-volatility
  ) %>%
  select(date, VOL)


# Merge factors 
factors_df <- cmkt_factor %>%
  inner_join(smb_factor, by = 'date') %>%
  inner_join(mom_factor, by = 'date') %>%
  inner_join(liquid_factor, by = "date") %>%
  inner_join(vol_factor, by = "date") %>%
  filter(date >= '2018-06-01')

saveRDS(factors_df, file="data/factors.rds")
