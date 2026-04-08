# =============================================================================
# SPDR Sector Spillover Study — Phase 1: TVP-VAR Spillover Measurement
# =============================================================================
# Input  : etf_daily_prices.csv   (written by feature_collection.py)
# Outputs: data/spillover_targets.csv   (quarterly NET per sector)
#          data/tsi_daily.csv
#          data/tsi_quarterly.csv
#          data/directional_spillover.csv   (new: directional spillovers for lags)
# =============================================================================

options(xts.warn_dplyr_breaks_lag = FALSE)

library(ConnectednessApproach)
library(ggplot2)
library(lubridate)
library(dplyr)
library(tidyr)
library(zoo)
library(xts)

# ── 1. Configuration ──────────────────────────────────────────────────────────

DATA_DIR   <- "data"
OUT_DIR    <- "data"
PLOT_DIR   <- "plots"

dir.create(OUT_DIR,  showWarnings = FALSE, recursive = TRUE)
dir.create(PLOT_DIR, showWarnings = FALSE, recursive = TRUE)

SECTORS    <- c("XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLB", "XLU")
VOL_WINDOW <- 30    # days — realised vol window
FORECAST_H <- 10    # GFEVD forecast horizon

# ── 2. Load ETF prices ────────────────────────────────────────────────────────

cat("Loading ETF prices...\n")

prices_path <- file.path(DATA_DIR, "etf_daily_prices.csv")
if (!file.exists(prices_path)) {
  stop(paste("Cannot find:", prices_path,
             "\nRun feature_collection.py first."))
}

prices_raw <- read.csv(prices_path, stringsAsFactors = FALSE)

first_col    <- colnames(prices_raw)[1]
dates_parsed <- as.Date(as.character(prices_raw[[first_col]]),
                        format = "%Y-%m-%d")

if (all(is.na(dates_parsed)))
  dates_parsed <- as.Date(as.character(prices_raw[[first_col]]))

cat(sprintf("  Dates: %s to %s\n",
            as.character(min(dates_parsed, na.rm = TRUE)),
            as.character(max(dates_parsed, na.rm = TRUE))))

price_cols   <- colnames(prices_raw)[colnames(prices_raw) != first_col]
prices_clean <- as.data.frame(lapply(prices_raw[, price_cols, drop = FALSE],
                                     as.numeric))
valid        <- !is.na(dates_parsed)
prices_clean <- prices_clean[valid, ]
dates        <- sort(dates_parsed[valid])
prices_clean <- prices_clean[order(dates_parsed[valid]), ]
rownames(prices_clean) <- as.character(dates)

available <- intersect(SECTORS, colnames(prices_clean))
cat(sprintf("  Sectors: %s\n", paste(available, collapse = ", ")))
cat(sprintf("  %d trading days x %d sectors\n", nrow(prices_clean), length(available)))

prices <- prices_clean[, available, drop = FALSE]

# ── 3. Realised volatility (Zhang et al. 2025, eq.16) ────────────────────────

cat("Computing realised volatility...\n")

log_ret         <- rbind(NA, apply(log(prices), 2, diff))
rv              <- apply(log_ret, 2, function(x)
                    rollapply(x, VOL_WINDOW, sd, na.rm = TRUE,
                              fill = NA, align = "right")) * sqrt(252)
rv_df           <- as.data.frame(rv)
rownames(rv_df) <- as.character(dates)
rv_df           <- rv_df[complete.cases(rv_df), ]

cat(sprintf("  Realised vol: %d obs x %d sectors\n",
            nrow(rv_df), ncol(rv_df)))

# ── 4. TVP-VAR estimation ─────────────────────────────────────────────────────

cat("\n=== TVP-VAR Estimation (total volatility only) ===\n")
cat("  This may take 20-60 minutes — please wait\n\n")

rv_dates <- as.Date(rownames(rv_df), format = "%Y-%m-%d")
rv_xts   <- xts(as.matrix(rv_df), order.by = rv_dates)
rv_xts   <- rv_xts[apply(rv_xts, 1, function(r) all(is.finite(r))), ]

cat(sprintf("  Input: %d obs x %d sectors\n", nrow(rv_xts), ncol(rv_xts)))

t0  <- Sys.time()
fit <- tryCatch(
  suppressWarnings(
    ConnectednessApproach(
      x             = rv_xts,
      nlag          = 1,
      nfore         = FORECAST_H,
      model         = "TVP-VAR",
      connectedness = "Time",
      VAR_config    = list(
        TVPVAR = list(kappa1 = 0.99, kappa2 = 0.99,
                      prior = "BayesPrior", gamma = 0.01)
      ),
      Connectedness_config = list(
        TimeConnectedness = list(generalized = TRUE)
      )
    )
  ),
  error   = function(e) { cat("  ERROR:", e$message, "\n"); NULL },
  warning = function(w) { invokeRestart("muffleWarning") }
)

if (is.null(fit)) stop("TVP-VAR estimation failed. Check input data.")

cat(sprintf("  Done in %.1f min\n",
            as.numeric(difftime(Sys.time(), t0, units = "mins"))))

# ── 5. Extract NET spillover ──────────────────────────────────────────────────

cat("\nExtracting NET spillover...\n")

net_raw <- fit$NET %||% fit$net %||% fit$Net
if (is.null(net_raw)) {
  cat("  Available names:", paste(names(fit), collapse = ", "), "\n")
  stop("Cannot find NET in fit object.")
}

net_df           <- as.data.frame(net_raw)
colnames(net_df) <- available[seq_len(ncol(net_df))]
net_df$date      <- as.Date(rownames(net_df), format = "%Y-%m-%d")

# ── 6. Extract directional spillovers (FROM) ──────────────────────────────────
# For lagged directional spillover feature

cat("Extracting directional spillovers (FROM)...\n")

from_raw <- fit$FROM %||% fit$from %||% fit$From
if (is.null(from_raw)) {
  cat("  WARNING: FROM not found — using NET as fallback\n")
  from_df <- net_df
  colnames(from_df) <- colnames(net_df)
} else {
  from_df <- as.data.frame(from_raw)
  colnames(from_df) <- available[seq_len(ncol(from_df))]
}
from_df$date <- as.Date(rownames(from_df), format = "%Y-%m-%d")

# ── 7. Extract TCI ───────────────────────────────────────────────────────────

cat("Extracting Total Connectedness Index...\n")

tci_raw <- fit$TCI %||% fit$tci
if (is.null(tci_raw)) {
  cat("  WARNING: TCI not found — filling with NA\n")
  tci_vals <- rep(NA_real_, nrow(rv_df))
} else {
  tci_vals <- as.numeric(tci_raw)
}

tsi_daily <- data.frame(
  date = as.Date(rownames(rv_df), format = "%Y-%m-%d"),
  tsi  = tci_vals
)

# ── 8. Resample to quarterly ──────────────────────────────────────────────────

cat("Resampling to quarterly frequency...\n")

to_quarterly <- function(df) {
  df$date    <- as.Date(df$date)
  df$quarter <- as.Date(as.yearqtr(df$date), frac = 1)
  df %>%
    group_by(quarter) %>%
    summarise(across(where(is.numeric), \(x) mean(x, na.rm = TRUE)),
              .groups = "drop") %>%
    rename(date = quarter)
}

net_q <- to_quarterly(net_df)
from_q <- to_quarterly(from_df)
tsi_q <- to_quarterly(tsi_daily)

cat(sprintf("  %d quarters\n", nrow(net_q)))

# ── 9. Build long-format spillover targets with lags ─────────────────────────

cat("Building spillover targets with lagged features...\n")

# Current period targets
spillover_targets <- net_q %>%
  pivot_longer(cols = -date, names_to = "sector",
               values_to = "net_spillover") %>%
  mutate(is_transmitter = as.integer(net_spillover > 0))

# Directional spillover (FROM) for lagged feature
directional_targets <- from_q %>%
  pivot_longer(cols = -date, names_to = "sector",
               values_to = "directional_spillover")

# Add lagged features (shift by 1 quarter)
spillover_targets <- spillover_targets %>%
  left_join(directional_targets, by = c("date", "sector")) %>%
  group_by(sector) %>%
  arrange(date) %>%
  mutate(
    lagged_net_spillover = lag(net_spillover, 1),
    lagged_directional_spillover = lag(directional_spillover, 1)
  ) %>%
  ungroup()

cat(sprintf("  %d rows x %d cols\n",
            nrow(spillover_targets), ncol(spillover_targets)))

# ── 10. Save outputs ─────────────────────────────────────────────────────────

cat("\nSaving outputs...\n")

write.csv(spillover_targets,
          file.path(OUT_DIR, "spillover_targets.csv"), row.names = FALSE)
cat(sprintf("  -> %s\n", file.path(OUT_DIR, "spillover_targets.csv")))

write.csv(tsi_daily,
          file.path(OUT_DIR, "tsi_daily.csv"), row.names = FALSE)
write.csv(tsi_q,
          file.path(OUT_DIR, "tsi_quarterly.csv"), row.names = FALSE)
cat(sprintf("  -> %s\n", file.path(OUT_DIR, "tsi_daily.csv")))
cat(sprintf("  -> %s\n", file.path(OUT_DIR, "tsi_quarterly.csv")))

# Save directional spillover separately
write.csv(directional_targets,
          file.path(OUT_DIR, "directional_spillover.csv"), row.names = FALSE)
cat(sprintf("  -> %s\n", file.path(OUT_DIR, "directional_spillover.csv")))

# ── 11. Plots ─────────────────────────────────────────────────────────────────

cat("Generating plots...\n")

# TSI over time
p_tsi <- ggplot(tsi_daily %>% filter(!is.na(tsi)),
                aes(x = date, y = tsi)) +
  geom_ribbon(aes(ymin = 0, ymax = tsi), fill = "grey70", alpha = 0.6) +
  geom_line(colour = "#2E5FA3", linewidth = 0.8) +
  labs(title    = "Dynamic Total Connectedness Index — SPDR Sectors",
       subtitle = "TVP-VAR(1), forgetting factor kappa = 0.99",
       x = NULL, y = "TCI (%)") +
  theme_minimal(base_size = 11)

ggsave(file.path(PLOT_DIR, "tsi_plot.png"),
       p_tsi, width = 10, height = 5, dpi = 150)
cat(sprintf("  -> %s\n", file.path(PLOT_DIR, "tsi_plot.png")))

# NET spillover heatmap
net_long <- net_q %>%
  pivot_longer(cols = -date, names_to = "sector",
               values_to = "net_spillover")

p_heat <- ggplot(net_long, aes(x = date, y = sector, fill = net_spillover)) +
  geom_tile() +
  scale_fill_gradient2(low = "#185FA5", mid = "white", high = "#A32D2D",
                       midpoint = 0, name = "NET") +
  labs(title    = "Dynamic NET Spillover — SPDR Sectors",
       subtitle = "Red = net transmitter; Blue = net receiver",
       x = NULL, y = NULL) +
  theme_minimal(base_size = 11)

ggsave(file.path(PLOT_DIR, "net_spillover_heatmap.png"),
       p_heat, width = 12, height = 4, dpi = 150)
cat(sprintf("  -> %s\n", file.path(PLOT_DIR, "net_spillover_heatmap.png")))

cat("\n✓ Phase 1 complete. Next: run ml_pipeline.py\n")