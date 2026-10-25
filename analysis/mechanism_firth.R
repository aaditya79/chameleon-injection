# Item 3 Amendment B1 (post-unblinding): Firth-penalized logistic models (logistf), one per spec.
#
#   Rscript analysis/mechanism_firth.R <input.csv> <specs.json> <output.json>
#
# Each spec selects rows (detector, goal_set, gated, scope, cells) and names the factors of a 2x2 (a, b). Model:
#   logistf(detected ~ a * b)  with a and b sum-coded (+0.5 / -0.5), or with a dummy-coded (level "+" = 0) when
#   spec$a_dummy is true, so the b coefficient is the b effect at a = "+" (used for V within D+).
# Profile penalized-likelihood CIs and p-values (logistf defaults).

suppressPackageStartupMessages({ library(logistf); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1], stringsAsFactors = FALSE)
specs <- fromJSON(args[2], simplifyVector = FALSE)

fit_spec <- function(s) {
  d <- dat[dat$detector == s$detector & dat$goal_set == s$goal_set & dat$cell %in% unlist(s$cells), ]
  if (isTRUE(s$gated)) d <- d[d$success == 1, ]
  if (s$scope == "fin+legal") d <- d[d$domain != "general", ]
  d$a <- d[[s$a]]
  d$b <- d[[s$b]]
  if (isTRUE(s$a_dummy)) d$a <- ifelse(d$a > 0, 0, 1)
  f <- logistf(detected ~ a * b, data = d)
  i <- match(s$term, names(coef(f)))  # coefficient names: (Intercept), a, b, a:b
  stopifnot(!is.na(i))
  list(id = s$id, n = nrow(d), n_detected = sum(d$detected), term = s$term,
       coef = unname(coef(f)[i]), ci95 = c(f$ci.lower[i], f$ci.upper[i]), p = unname(f$prob[i]),
       method = "logistf, profile penalized likelihood")
}

res <- lapply(specs, function(s) tryCatch(fit_spec(s), error = function(e) list(id = s$id, status = paste("error:", conditionMessage(e)))))
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, null = "null"), args[3])
