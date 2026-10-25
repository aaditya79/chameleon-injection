# Item 3 mixed models (P6; called by analysis/mechanism_detection_stats.py).
#
#   Rscript analysis/mechanism_glmm.R <input.csv> <specs.json> <output.json>
#
# Each spec selects rows (detector, goal_set, gated, scope) and a model:
#   primary     detected ~ D * V + (1 | task) + (1 | template)
#   acs_words   primary + acs_z + words_z            (A3/A4 sensitivity: ACS and length covariates)
#   success     primary + success                     (pre-detection note B.2 / D4; ungated only)
# D and V are sum-coded (+0.5 = D+ / V+, -0.5 = D- / V-), so the D and V coefficients are average log-odds effects
# over the other factor. Fallback order (P6): drop (1 | template); then GEE with an exchangeable task cluster.
# A fit counts as failed if lme4 reports a convergence message other than a singular (boundary) fit, or errors.
# Wald z tests throughout.

suppressPackageStartupMessages({ library(lme4); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
dat <- read.csv(args[1], stringsAsFactors = FALSE)
specs <- fromJSON(args[2], simplifyVector = FALSE)

fixed_terms <- list(primary = "", acs_words = " + acs_z + words_z", success = " + success")
contrasts_of <- list(
  D = c(D = 1), V = c(V = 1), DxV = c("D:V" = 1),
  "V|D+" = c(V = 1, "D:V" = 0.5), "V|D-" = c(V = 1, "D:V" = -0.5),
  "D|V+" = c(D = 1, "D:V" = 0.5), "D|V-" = c(D = 1, "D:V" = -0.5)
)

try_glmer <- function(f, d) {
  msgs <- character(0)
  m <- tryCatch(
    withCallingHandlers(
      glmer(f, data = d, family = binomial,
            control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))),
      warning = function(w) { msgs <<- c(msgs, conditionMessage(w)); invokeRestart("muffleWarning") }),
    error = function(e) { msgs <<- c(msgs, paste("error:", conditionMessage(e))); NULL })
  if (!is.null(m)) msgs <- c(msgs, unlist(m@optinfo$conv$lme4$messages))
  failed <- is.null(m) || any(!grepl("singular", msgs, ignore.case = TRUE))
  list(model = m, failed = failed, messages = unique(msgs), singular = !is.null(m) && isSingular(m))
}

fit_spec <- function(s) {
  d <- dat[dat$detector == s$detector & dat$goal_set == s$goal_set, ]
  if (isTRUE(s$gated)) d <- d[d$success == 1, ]
  if (s$scope == "fin+legal") d <- d[d$domain != "general", ]
  out <- list(id = s$id, n = nrow(d), n_detected = sum(d$detected))
  if (length(unique(d$detected)) < 2) { out$status <- "not_estimable_constant_outcome"; return(out) }
  rhs <- paste0("detected ~ D * V", fixed_terms[[s$model]])
  attempts <- list(
    list(label = "glmer (1|task)+(1|template)", f = paste(rhs, "+ (1 | task) + (1 | template)")),
    list(label = "glmer (1|task)", f = paste(rhs, "+ (1 | task)")))
  m <- NULL
  for (a in attempts) {
    r <- try_glmer(as.formula(a$f), d)
    out$fit_log <- c(out$fit_log, list(list(model = a$label, failed = r$failed, messages = r$messages)))
    if (!r$failed) { m <- r$model; out$model_used <- a$label; out$singular <- r$singular; break }
  }
  if (is.null(m)) {
    if (!requireNamespace("geepack", quietly = TRUE)) { out$status <- "gee_fallback_needed_geepack_missing"; return(out) }
    d <- d[order(d$task), ]
    g <- geepack::geeglm(as.formula(rhs), data = d, family = binomial, id = factor(d$task), corstr = "exchangeable")
    b <- coef(g); V <- vcov(g); out$model_used <- "GEE exchangeable (task)"
  } else {
    b <- fixef(m); V <- as.matrix(vcov(m))
    out$random_sd <- as.list(sapply(VarCorr(m), function(x) attr(x, "stddev")))
  }
  out$status <- "ok"
  out$coef_names <- names(b)
  out$estimates <- lapply(contrasts_of, function(cw) {
    w <- setNames(rep(0, length(b)), names(b)); w[names(cw)] <- cw
    est <- sum(w * b); se <- sqrt(as.numeric(t(w) %*% V %*% w)); z <- est / se
    list(coef = est, se = se, z = z, p = 2 * pnorm(-abs(z)))
  })
  for (cv in intersect(c("success", "acs_z", "words_z"), names(b))) {
    se <- sqrt(V[cv, cv]); out$estimates[[cv]] <- list(coef = b[[cv]], se = se, z = b[[cv]] / se, p = 2 * pnorm(-abs(b[[cv]] / se)))
  }
  out
}

res <- lapply(specs, function(s) tryCatch(fit_spec(s), error = function(e) list(id = s$id, status = paste("error:", conditionMessage(e)))))
writeLines(toJSON(res, auto_unbox = TRUE, digits = NA, null = "null"), args[3])
