###############################################################################
# Code to get coin shortname from CoinCodex
# Short name is necessary to extract market data from API
# Coins are ordered by market capitalization
# For more coins, increase limit at the URL
###############################################################################

setwd(Sys.getenv("THESIS_WORKDIR"))

libraries_used <- c("httr", "jsonlite", "dplyr")

for (library_used in libraries_used) {
  if (!require(library_used, character.only = TRUE)) {
    install.packages(library_used)
    library(library_used, character.only = TRUE)
  }
}

# Call full coin list endpoint
# Get 4000 coin names
allcoins_url <- "https://coincodex.com/api/v1/coins_compat/get_coin_list?limit=15000"
coins4000_url <- "https://coincodex.com/api/v1/coins_compat/get_coin_list?limit=4000"
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
      select(symbol, name, shortname, trading_since, market_cap_rank,
             average_mktcap_all_time) %>%
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

response <- GET(stablecoins_url)
json <- content(response, as = "text", encoding = "UTF-8")
data <- fromJSON(json)
data
##############

###############################################################################
# The initial sample was coins4000, but it gets coins based on its last days'
# marketcapitalization. This may induce a "survivorship bias". 
# I get all coin names available, take the difference and store it. 
###############################################################################

coins4000 <- get_names(coins4000_url)   
allnames <- get_names(allcoins_url)
stablecoins <- get_names(stablecoins_url)


allcoins <- allnames %>%
  filter(!is.na(trading_since)) %>% # remove coins with missing trading_since
  arrange(trading_since) %>%
  mutate(age_i = row_number()) 


# different coinnames
diffcoins <- setdiff(allcoins$shortname, coins4000$shortname)

# Take diff and remove stable coins:  9691 coins
missingcoins <- allcoins %>%       
  filter(shortname %in% diffcoins) %>%
  filter(!(symbol %in% stablecoins$symbol))

summary(missingcoins) # all coins have no marketcap rank, possibly they're dead

# Remove stable coins from the list: 3963 out of 4000 coins
filtered_coins <- coins4000 %>%
  filter(!(symbol %in% stablecoins$symbol))


# shortname is what we want to construct the URLs
writeLines(filtered_coins$shortname, "code/coins_shortname.txt")

writeLines(missingcoins$shortname, "code/missingcoins_shortname.txt")
