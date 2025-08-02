#!/bin/bash

total_coins=$(wc -l < coins_slug.txt)
i=1

logfile="not_found.log"
> "$logfile"  # clear the log file at the start

while read -r coin; do	
  url="https://coincodex.com/api/coincodexcoins/get_historical_data_by_slug/$coin/2014-01-01/2025-04-30"
  echo "[$i/$total_coins] Checking $url"
  
  if curl --silent --fail "$url" > /dev/null; then
    echo "Exists"
  else
    echo "Not Found"
    echo "$coin" >> "$logfile"
  fi

  ((i++))
done < coins_shortname.txt
