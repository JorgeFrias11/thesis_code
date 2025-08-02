download_coin_data <- function(coin_vector, out_file, counter = 1,
                               start_date = "2014-01-01",
                               end_date = "2025-04-30") {
  
  # counter for tracking progress
  counter <- counter
  
  # Function to get API data
  get_CoinData <- function(coin, start_date, end_date) {
    Sys.sleep(4) # four seconds between requests
    cc_url <- "https://coincodex.com/api/coincodexcoins/get_historical_data_by_slug/%s/%s/%s"
    url <- sprintf(cc_url, coin, start_date, end_date)
    
    response <- GET(url)
    if (status_code(response) != 200) {
      cat(counter, status_code(response), "\n")
      counter <<- counter + 1
      stop("Failed to fetch data from API.")
    } else {
      cat(counter, "Getting", coin, "data\n")
      counter <<- counter +1
    }
    
    data_json <- content(response, "text", encoding = "UTF-8")
    data_parsed <- fromJSON(data_json)
    
    df <- as.data.frame(data_parsed$data) %>%
      mutate(
        date = as.Date(time_start),
        coinName = coin,
        open = price_open_usd,
        close = price_close_usd,
        high = price_high_usd,
        low = price_low_usd,
        volume = volume_usd,
        marketcap = market_cap_usd,
      ) %>%
      select(date, coinName, open, close, high, low, volume, marketcap) %>%
      filter(close > 0, volume > 0, marketcap > 0)
    
    return(df)
  }
  
  # Wrapper with error handling
  store_coin_data <- function(coin) {
    df <- try(get_CoinData(coin, start_date, end_date))
    if (!inherits(df, "try-error") && !is.null(df)) {
      return(df)
    } else {
      return(NULL)
    }
  }
  
  # Download data for all coins
  coin_dfs <- lapply(coin_vector, store_coin_data)
  coindata <- bind_rows(coin_dfs)
  
  # Save output
  data_dir <- "data"
  if (!dir.exists(data_dir)) dir.create(data_dir)
  
  saveRDS(coindata, file = file.path(data_dir, out_file))
  cat("Saved", length(unique(coindata$coinName)), "coins to", out_file, "\n")
}
