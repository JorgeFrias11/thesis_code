###############################################################################
# The code reports and cites all the packages used in the project. 
# I use the package grateful to generate the Rmd paragraph and the bibtex file
###############################################################################

packages_used <- c("httr", "jsonlite", "tidyverse", "dplyr", "purrr", "zoo", 
                   "lubridate", "readxl", "slider", "purrr", "moments", "ggplot2",
                   "quantmod", "bidask", "PerformanceAnalytics", "pcaMethods")

library(grateful)

text <- grateful::cite_packages(output = "paragraph", pkgs = packages_used, 
                                out.format = "Rmd", out.dir = ".", omit = NULL,
                                cite.tidyverse = T, include.RStudio = T)

cat(text[1])


# Another option is to generate the bib.tex file with knitr

# To cite R
toBibtex(citation())

# To cite the packages used
knitr::write_bib(packages_used)

# To save the output to a file:
# knitr::write_bib(c(.packages(), "bookdown"), "packages.bib")







