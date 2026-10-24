# Item 3b exploratory (pre-registration addition 4): per detector, Firth logistic on pooled O, (f), (d) texts,
#   detected ~ log(words) + specificity
#   Rscript analysis/item3b_firth.R <input.csv> <output.json>
suppressPackageStartupMessages({ library(logistf); library(jsonlite) })
args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1], stringsAsFactors = FALSE)
out <- lapply(split(dat, dat$key), function(d) {
  if (length(unique(d$detected)) < 2) return(list(key = d$key[1], n = nrow(d), note = "no variation in outcome"))
  f <- logistf(detected ~ logw + spec, data = d)
  t <- c("logw", "spec")
  list(key = d$key[1], n = nrow(d), n_detected = sum(d$detected),
       coef = as.list(setNames(unname(coef(f)[t]), t)),
       ci95 = list(logw = c(f$ci.lower["logw"], f$ci.upper["logw"]), spec = c(f$ci.lower["spec"], f$ci.upper["spec"])),
       p = as.list(setNames(unname(f$prob[t]), t)))
})
write(toJSON(unname(out), auto_unbox = TRUE, digits = NA), args[2])
