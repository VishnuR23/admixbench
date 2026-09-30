#!/usr/bin/env bash
# Install the R packages the benchmark calls through Rscript.
# Needs R >= 4.2 on PATH. On macOS with Homebrew R 4.6 and Apple clang 15, C
# packages fail to build under -std=gnu23. Put this in ~/.R/Makevars first:
#   CC = clang -std=gnu17
#   CC23 = clang -std=gnu17
set -euo pipefail
Rscript -e 'install.packages(c("remotes", "jsonlite"), repos = "https://cloud.r-project.org")'
Rscript -e 'remotes::install_github("uqrmaie1/admixtools", dependencies = NA, upgrade = "never")'
Rscript -e 'library(admixtools); library(jsonlite); cat("admixtools", as.character(packageVersion("admixtools")), "OK\n")'
