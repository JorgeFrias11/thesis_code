###############################################################################
# Code to get coin shortname from CoinCodex
# Short name is necessary to extract market data from API
# Coins are ordered by market capitalization
# For more coins, increase limit at the URL
###############################################################################

setwd(Sys.getenv("THESIS_CODE"))

libraries_used <- c("httr", "jsonlite", "dplyr")

for (library_used in libraries_used) {
  if (!require(library_used, character.only = TRUE)) {
    install.packages(library_used)
    library(library_used, character.only = TRUE)
  }
}

# Call full coin list endpoint. Max processable limit is 5000 
# Get coin names
allcoins_url <- "https://coincodex.com/api/v1/coins_compat/get_coin_list?limit=15000"
stablecoins_url <- "https://coincodex.com/api/v1/coins_compat/get_coin_list?categories=stablecoins"

# Parse
get_names <- function(url){
  response <- GET(url)
  if (status_code(response) == 200) {
    json <- content(response, as = "text", encoding = "UTF-8")
    data <- fromJSON(json)
    
    # Check data$data for all the columns available
    
    # Select useful fields
    coins_df <- data$data %>%
      select(symbol, name, shortname, market_cap_rank, 
             trading_since, last_market_cap_usd) %>%
      distinct()
    
    coins_df$trading_since <- as.Date(
      as.POSIXct(coins_df$trading_since, origin = "1970-01-01")
      )
    
    cat("Getting coin shortnames...", "\n") 
    # View
    #head(coins_df)
  } else {
    cat("Request failed:", status_code(response), "\n")
  }
  
  return(coins_df)
}

############## Test

response <- GET("https://coincodex.com/api/v1/coins_compat/get_coin_list?limit=200")
json <- content(response, as = "text", encoding = "UTF-8")
data <- fromJSON(json)
data
##############

###############################################################################
# The initial sample was coins4000, but it gets coins based on its last days'
# marketcapitalization. This may induce a "survivorship bias". 
# I get all coin names available - up to the 5000 limit.
###############################################################################

allnames <- get_names(allcoins_url)
stablecoins <- get_names(stablecoins_url)


allcoins <- allnames %>%
  filter(!is.na(trading_since)) %>% # remove coins with missing trading_since, only 3
  arrange(trading_since) %>%
  mutate(age_i = row_number()) 

# Remove stable coins:  4958 coins
filtered_coins <- allcoins %>%       
  filter(!(symbol %in% stablecoins$symbol))

summary(filtered_coins) # not all coins have marketcap rank or last market cap, possibly they're dead

# shortname is what we want to construct the URLs
writeLines(filtered_coins$shortname, "rcode/coins_shortname.txt")
