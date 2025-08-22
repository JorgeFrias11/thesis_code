#######################################################################
# This code constructs the predictors based on the categories
# in Liu et al. (2022), Bianchi (2021), and Mercik et al (2025)
# I calculate all characteristics up to time t (inclusive). Therefore, 
# lagging the characteristics is necessary to predict (excess) return_t with 
# Z_t-1, where Z_t-1 is the set of characteristics observable at time t-1
#######################################################################

packages_used <- c("dplyr", "readxl", "slider", "purrr", "moments", 
                    "quantmod", "bidask", "tidyr", "PerformanceAnalytics")

for (package_used in packages_used) {
  if (!require(package_used, character.only = TRUE)) {
    install.packages(package_used)
    library(package_used, character.only = TRUE)
  }
}

setwd(Sys.getenv("THESIS_DATA_DIR"))
cat("Current working directory: \n")
getwd()


start_time <- Sys.time()

#######################################################################
#  Only edit these three variables
#######################################################################

cat("Reading file...", "\n")

coindata <- readRDS(file="data/coins_data.rds")
all_vars_output_file <- "daily_predictors.rds"
data_dir <- "data"
#output_file <- "predictors2.rds"

cat("Constructing predictors...", "\n")


########################################################################
#  Size variables
########################################################################

cat("Constructing size (1/7)...", "\n")

coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(mcap = log(marketcap),
         prc = log(close), 
         maxdprc = slide_dbl(close, max, .before = 6L, .complete = TRUE))


cat("Size done...", "\n")

########################################################################
#  Momentum variables
########################################################################

cat("Constructing Momentum (2/7)...", "\n")

# calculate Momemtum rt_j
# calc_r_kj <- function(data, k, j){
#   stopifnot(k > j)  # Ensure the window makes sense
#   
#   var_name <- paste0("r", k, "_", j)
#   
#   data %>%
#     mutate(!!var_name :=
#              if_else(row_number() > k,
#                      purrr::reduce(
#                        purrr::map(j:(k - 1), ~1 + lag(ret, .x)), `*`) - 1, # include today
#                        #purrr::map((j+1):k, ~1 + lag(ret, .x)), `*`) - 1,
#                      NA_real_))
# }
# 
# coindata <- coindata %>%
#   group_by(coinName) %>%
#   mutate(r2_1 = lag(ret, 1),
#          dh90 = close / slide_dbl(.x = close,  # before, dplyr::lag(close)
#                                   .f = max,
#                                   .before = 89L,
#                                   #.after = -1L,
#                                   .complete = TRUE)) %>%
#   calc_r_kj(7, 0) %>%    
#   calc_r_kj(21, 0) %>%
#   calc_r_kj(30, 0) %>%
#   calc_r_kj(30, 14)  %>%   # r30_14: from t-30 to t-14
#   calc_r_kj(180, 60)      # r180_60: from t-180 to t-60


# Calculate Momentum signals
coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(r7_0 = slide_dbl(.x = ret,
                          .f = Return.cumulative,
                          .before = 6L,
                          .complete = TRUE),
         r14_0 = slide_dbl(.x = ret,
                          .f = Return.cumulative,
                          .before = 13L,
                          .complete = TRUE),
         r21_0 = slide_dbl(.x = ret,
                          .f = Return.cumulative,
                          .before = 20L,
                          .complete = TRUE),
         r30_0 = slide_dbl(.x = ret,
                           .f = Return.cumulative,
                           .before = 29L,
                           .complete = TRUE),
         r30_14 = slide_dbl(.x = ret,
                           .f = Return.cumulative,
                           .before = 29L,
                           .after = -14L,
                           .complete = TRUE),
         r30_14 = slide_dbl(.x = ret,
                            .f = Return.cumulative,
                            .before = 29L,
                            .after = -14L,
                            .complete = TRUE),
         r180_60 = slide_dbl(.x = ret,
                            .f = Return.cumulative,
                            .before = 179L,
                            .after = -60L,
                            .complete = TRUE), 
         dh90 = close / slide_dbl(.x = close,  # before, dplyr::lag(close)
                                  .f = max,
                                  .before = 89L,
                                  .complete = TRUE))

  
cat("Momentum done...", "\n")

# Check no extreme returns
summary(coindata$ret)

########################################################################
#  Volume variables
########################################################################

cat("Constructing Volume (3/7)...", "\n")

prcvol <- function(price, volume){
  log(mean(price * volume))
}

volscaled <- function(price, volume, marketcap){
  log(mean((price * volume)/ marketcap ))
}

coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(
    volume_7d = slide_dbl(.x = volume, 
                          .f = mean, 
                          .before = 6L, 
                          .complete = TRUE),
    volume_30d = slide_dbl(.x = volume,
                           .f = mean, 
                           .before = 29L, 
                           .complete = TRUE),
    logvol = slide_dbl(.x = volume,
                       .f = ~log(mean(.x)),
                       .before = 6L,
                       .complete = TRUE),
    prcvol = slide2_dbl(.x = close,
                           .y = volume,
                           .f = prcvol,
                           .before = 6L,
                           .complete = TRUE),
    volscaled = pslide_dbl(.l = list(close, volume, marketcap),
                           .f = ~ volscaled(..1, ..2, ..3 ),
                           .before = 6L,
                           .complete = TRUE)
  )

cat("Volume done...", "\n")

########################################################################
#  Volality and Risk 
########################################################################

cat("Constructing Volality and Risk (4/7)...", "\n")

# 1-month Treasury bill rate as Risk-free rate proxy
#getSymbols("DGS1MO", src = "FRED")
# save(DGS1MO, file = "tbill_1mo.rda")
load("tbill_1mo.rda")
tbill <- DGS1MO

# Risk-Free rate.  Fill NAs with previous obs and convert to dataframe
r_f <- na.locf(tbill["2013-12-31/2025-07-31"])[-1] / 100  
# Convert annualized yield into daily rate (252 trading days)
r_f <- (1 + r_f)^(1/252) - 1

r_f <- data.frame(date = index(r_f), rf = as.numeric(r_f))

# Crypto daily market return
r_mkt <- coindata %>%
  filter(!is.na(ret), !is.na(marketcap)) %>%
  group_by(date) %>%
  summarise(
    mktret = sum(ret * marketcap, na.rm = TRUE) / sum(marketcap, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(r_f, by = "date") %>%    # join rf rate
  mutate(
    rf = na.locf(rf),  # fill NAs from weekends
    cmkt = mktret - rf
  )

coindata <- coindata %>%
  left_join(r_mkt, by = "date") %>%
  mutate(
    ret_excess = ret - rf      # excess return of the coin
  ) %>%
  relocate(date, coinName, ret_excess)

### Functions

# CAPM: Obtain alpha, beta and ivol from the regression
# capm <- function(x, y){
#   reg <- lm(y ~ x)
#   alpha <- coefficients(reg)[[1]]
#   beta <- coefficients(reg)[[2]]
#   ivol <- sd(residuals(reg))
#   return(c("alpha" = alpha, "beta" = beta, "ivol" = ivol))
# }

capm <- function(x, y) {    # faster than linear regressions
  valid <- complete.cases(x, y)
  x <- x[valid]
  y <- y[valid]
  n <- length(x)
  
  X <- cbind(1, x)  # intercept and x
  coef <- solve(t(X) %*% X, t(X) %*% y)  # (X'X)^-1 X'y
  y_hat <- X %*% coef
  residuals <- y - y_hat
  ivol <- sd(residuals)
  
  return(c("alpha" = coef[1], "beta" = coef[2], "ivol" = ivol))
}


# Realized volatility
rvol <- function(open, high, low, close){   # From Yang and Zang (2000)
  n <- length(open)
  
  o <- log(open[-1]) - log(close[-n])   # normalized open  ln(o1) - ln(c0)
  u <- log(high[-1]) - log(open[-1])    # normalized high  ln(h1) - ln(o1)
  d <- log(low[-1]) - log(open[-1])     # normalized low   ln(l1) - ln(o1)
  c <- log(close[-1]) - log(open[-1])   # normalized close ln(c1) - ln(o1)
  
  vo <- var(o)  
  vc <- var(c)
  vrs <- mean(u * (u - c) + d * (d - c))  # Rogers et al (1994)
  k <- 0.34 / (1.34 + (n + 1) / (n - 1))
  
  sigma2 <- vo + k * vc + (1 - k) * vrs
  return(sqrt(sigma2))
}


delay <- function(x, y) {
  x_l1 <- dplyr::lag(x, 1)   # 1-day lag
  x_l2 <- dplyr::lag(x, 2)   # 2-day lag
  
  # Combine and remove rows with NA
  df <- data.frame(y = y, x = x, x_l1 = x_l1, x_l2 = x_l2)
  df <- na.omit(df)
  
  simp_reg <- lm(y ~ x, data = df)
  r2 <- summary(simp_reg)$r.squared
  
  mult_reg <- lm(y ~ x + x_l1 + x_l2, data = df)
  r2_mult <- summary(mult_reg)$r.squared
  
  return(r2_mult - r2)  # Improvement in R2
}


stdprcvol <- function(price, volume){
  log(sd(price * volume))
}

# Expected shortfall 
es_fun <- function(x, p = 0.05) {
  threshold <- quantile(x, probs = p, na.rm = TRUE)
  mean(x[x <= threshold], na.rm = TRUE)
}


### Create predictors
coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(
    capm_vars = slide2(.x = cmkt,    # slide2 as the result is a list
                       .y = ret_excess, 
                       .f = capm,
                       .before = 29L,
                       .complete = TRUE)
  ) %>%
  tidyr::unnest_wider(capm_vars) %>%       # alpha, beta, and ivol 
  mutate(
    beta2 = beta^2,
    rvol_7d = pslide_dbl(.l = list(open, high, low, close),
                          .f = ~rvol(..1, ..2, ..3, ..4),
                          .before = 6L,
                          .complete = TRUE),
    rvol_30d = pslide_dbl(.l = list(open, high, low, close),
                      .f = ~rvol(..1, ..2, ..3, ..4),
                      .before = 29L,
                      .complete = TRUE),
    retvol = slide_dbl(.x = ret, 
                       .f = sd, 
                       .before = 6L, 
                       .complete = TRUE),
    var = slide_dbl(.x = ret,
                    .f = ~quantile(.x, probs = 0.05, na.rm = TRUE),
                    .before = 89L,   
                    .complete = TRUE),
    es_5 = slide_dbl(.x = ret, 
                      .f = es_fun, 
                      .before = 89L, 
                      .complete = TRUE),
    delay = slide2_dbl(.x = cmkt,
                       .y = ret_excess, 
                       .f = delay,
                       .before = 29L,
                       .complete = TRUE),
    stdprcvol = slide2_dbl(.x = close, 
                           .y = volume, 
                           .f = stdprcvol,
                           .before = 6L,
                           .complete = TRUE)
  )

cat("Volatility and Risk done...", "\n")

########################################################################
#  Liquidity variables (Mercik et al. 2025)
########################################################################

cat("Constructing Liquidity (5/7)...", "\n")


illiq <- function(return, volume){
  mean(abs(return)/volume)
}

# trading volume over market capitalization
turnover <- function(volume, marketcap){
  dplyr::lag(volume) / marketcap
}


std_turn <- function(turnover){
  sd(turnover)
}


std_vol <- function(volume){  
  log(sd(volume))
}


cv_vol <- function(volume){
  sd(volume) / mean(volume)
}


## Create predictors
coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(
    bidask = bidask::edge_rolling(open, high, low, close, width = 30),
    illiq = slide2_dbl(.x = ret,
                       .y = volume, 
                       .f = ~illiq(return = .x, volume = .y),
                       .before = 89L,
                       .complete = TRUE),
    turn = turnover(volume, marketcap),
    turn_7d = slide_dbl(.x = turn, 
                        .f = mean, 
                        .before = 6L, 
                        .complete = TRUE), 
    std_turn = slide_dbl(.x = turn,
                         .f = std_turn,
                         .before = 29L, 
                         .complete = TRUE),
    std_vol = slide_dbl(.x = volume, 
                        .f = std_vol,
                        .before = 29L, 
                        .complete = TRUE),
    cv_vol = slide_dbl(.x = volume, 
                       .f = cv_vol,
                       .before = 29L,
                       .complete = TRUE),
    avg_turn = slide_dbl(.x = turn,     # used for sat
                         .f = mean,
                         .before = 30L, 
                         .after = -1L,  # 30 past days excluding today
                         .complete = TRUE),
    sigma_turn = slide_dbl(.x = turn,   # used for sat
                           .f = sd,
                           .before = 30L,
                           .after = -1L,   # 30 past days excluding today
                           .complete = TRUE),
    sat = (dplyr::lag(turn) - avg_turn) / sigma_turn,
    volsh_15d = log(volume) - slide_dbl(.x = volume,
                                         .f = ~log(mean(.x)),
                                         .before = 14L, 
                                         .complete = TRUE),
    volsh_30d = log(volume) - slide_dbl(.x = volume,
                                         .f = ~log(mean(.x)),
                                         .before = 29L, 
                                         .complete = TRUE)    
  ) %>%
  select( -avg_turn, -sigma_turn)


# market turnover to calculate DTO
market_turnover <- coindata %>%
  filter(!is.na(turn), !is.na(marketcap)) %>%
  group_by(date) %>%
  summarise(
    mkt_avg_turn = sum(turn * marketcap, na.rm = TRUE) / sum(marketcap, na.rm = TRUE),
    .groups = "drop"
  )
 
### Create dto predictor
coindata <- coindata %>%  
  left_join(market_turnover, by = "date") %>%    # add market turnover
  mutate(excess_turn = turn - mkt_avg_turn) %>%
  group_by(coinName) %>% 
  mutate(
    dto = excess_turn - slide_dbl(.x = excess_turn,
                                  .f = ~median(.x, na.rm = TRUE),
                                  .before = 89,
                                  .complete = TRUE)
    ) %>%
  ungroup() %>%
  select(-mkt_avg_turn, -excess_turn)  # remove extra cols


cat("Liquidity done...", "\n")

########################################################################
#  Distribution variables
########################################################################

cat("Constructing Distribution (6/7)...", "\n")

coindata <- coindata %>%
  group_by(coinName) %>%
  mutate(
    skew_7d = slide_dbl(ret, skewness, .before = 6L, .complete = TRUE),
    skew_30d = slide_dbl(ret, skewness, .before = 29L, .complete = TRUE),
    kurt_7d = slide_dbl(ret, kurtosis, .before = 6L, .complete = TRUE),
    kurt_30d = slide_dbl(ret, kurtosis, .before = 29L, .complete = TRUE),
    maxret_7d = slide_dbl(ret, max, .before = 6L, .complete = TRUE),
    maxret_30d = slide_dbl(ret, max, .before = 29L, .complete = TRUE),
    minret_7d = slide_dbl(ret, min, .before = 6L, .complete = TRUE), 
    minret_30d = slide_dbl(ret, min, .before = 29L, .complete = TRUE)
    )


cat("Distribution done...", "\n")

########################################################################
#  News variables
########################################################################

cat("Constructing News (7/7)...", "\n")

last_date <- max(coindata$date, na.rm = T)

gpr <- read_xlsx("data_gpr_daily_recent.xlsx", sheet = "Sheet1")
gpr <- gpr %>%
  mutate(date = as.Date(date)) %>%
  select(date, GPRD, GPRD_MA7, GPRD_MA30) %>% 
  filter(date >= "2014-01-01" & date <= last_date)

nsi <- read_xlsx("news_sentiment_data.xlsx", sheet = "Data")
nsi <- nsi %>%
  mutate(date = as.Date(date)) %>%
  select(date = date, nsi = 'News Sentiment') %>%
  filter(date >= "2014-01-01" & date <= last_date)

coindata <- left_join(
  coindata,
  left_join(gpr, nsi, by = "date"),
  by = "date"
)


cat("News done...", "\n")


########################################################################
#  Save data
########################################################################

cat("Saving data as:", file.path(data_dir, all_vars_output_file), "\n")

saveRDS(coindata, file.path(data_dir, all_vars_output_file))

# Remove support variables
#coindata <- coindata %>%
#  select(-open, -close, -high, -low, -marketcap, -mktret, 
#         -rf, -cmkt)

#saveRDS(coindata, file.path(data_dir, output_file))

end_time <- Sys.time()

print("Run time: \n")
print(end_time - start_time)
