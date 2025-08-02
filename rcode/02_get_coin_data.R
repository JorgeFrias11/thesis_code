######################################################
# This code gets daily data from CoinCodex API
# We get closing prices, volume, and market cap of the first
# 100 coins by market cap
# We save the data to coinData.rds
######################################################

setwd(Sys.getenv("THESIS_CODE"))

libraries_used <- c("httr", "jsonlite", "dplyr", "tidyverse")

for (library_used in libraries_used) {
  if (!require(library_used, character.only = TRUE)) {
    install.packages(library_used)
    library(library_used, character.only = TRUE)
  }
}

counter <- 1

### function to obtain data
get_CoinData <- function(coin, start_date, end_date) {
  
  Sys.sleep(5)  # wait 5 seconds before each request
  
  # get data from API
  cc_url <- "https://coincodex.com/api/coincodexcoins/get_historical_data_by_slug/%s/%s/%s"
  url <- sprintf(cc_url, coin, start_date, end_date)
  
  response <- GET(url)
  
  if (status_code(response) != 200) {
    cat(counter, status_code(response), "\n")
    counter <<- counter + 1
    stop("Failed to fetch data from API.")
  }
  else {
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
      marketcap = market_cap_usd
    ) %>%
    select(date, coinName, open, close, high, low, volume, marketcap) %>%
    # remove obs where value <= 0
    filter(close > 0, volume > 0, marketcap > 0)
  
  return(df)
}

### Example
#bitcoin <- get_CoinData("bitcoin", "2019-09-20", "2019-10-07")
#head(bitcoin, 25)

######################################################
# Store dataframes in a list, then bind them
######################################################

# Define wrapper
store_coin_data <- function(coin, start_date, end_date) {
  df <- try(get_CoinData(coin, start_date, end_date))
  
  if (!inherits(df, "try-error") && !is.null(df)) {
    return(df)
  } else {
    return(NULL)
  }
}

# Date range from 2014, since there's no volume data before
start_date <- "2014-01-01"
end_date <- "2025-07-31"

coinNames <- readLines("rcode/coins_shortname.txt")

# Apply to all coins using lapply
coin_dfs <- lapply(coinNames, function(name) {
  store_coin_data(name, start_date, end_date)
})

# Combine into single data frame
coindata <- bind_rows(coin_dfs)

# number of unique coins
cat("Number of unique coins:", length(unique(coindata$coinName)))

data_dir <- Sys.getenv("THESIS_DATA_DIR")
out_file <- "coin_raw_data.rds"

saveRDS(coindata, file = file.path(data_dir, out_file))
