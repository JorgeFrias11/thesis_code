# Install necessary packages in local library (read CLUSTER GUIDE cluster)

packages_used <- c("dplyr", "readxl", "slider", "purrr", "moments", "zoo", "lubridate",
                    "quantmod", "bidask", "tidyr", "PerformanceAnalytics")

for (pkg in packages_used) {
  install.packages(pkg, repo='https://cloud.r-project.org/', lib="~/R_LIBS")
}
