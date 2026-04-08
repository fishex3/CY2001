library(ConnectednessApproach)
library(ggplot2)
library(lubridate)
library(dplyr)
library(tidyr)
library(zoo)
library(xts)

`%||%` <- function(a, b) {
  if (!is.null(a)) a else b
}

calculate_spillover <- function(prices_raw, sectors, vol_window, forecast_h, kappa1, kappa2) {
    sectors <- unlist(sectors)
    # Parse dates from rownames
    raw_dates <- rownames(prices_raw)
    dates_parsed <- as.Date(raw_dates, format = "%Y-%m-%d")

    if (any(is.na(dates_parsed))) {
      stop("Date conversion failed. Ensure Python index is YYYY-MM-DD.")
    }

    cat(sprintf("  Dates: %s to %s\n",
                as.character(min(dates_parsed, na.rm = TRUE)),
                as.character(max(dates_parsed, na.rm = TRUE))))

    # ✅ FIXED HERE
    price_cols <- colnames(prices_raw)

    prices_clean <- as.data.frame(lapply(prices_raw[, price_cols, drop = FALSE],
                                         as.numeric))
    valid        <- !is.na(dates_parsed)
    prices_clean <- prices_clean[valid, ]
    dates        <- sort(dates_parsed[valid])
    prices_clean <- prices_clean[order(dates_parsed[valid]), ]
    rownames(prices_clean) <- as.character(dates)

    available <- intersect(sectors, colnames(prices_clean))

    prices <- prices_clean[, available, drop = FALSE]
    log_ret         <- rbind(NA, apply(log(prices), 2, diff))
    rv              <- apply(log_ret, 2, function(x)
                        rollapply(x, vol_window, sd, na.rm = TRUE,
                                  fill = NA, align = "right")) * sqrt(252)
    rv_df           <- as.data.frame(rv)
    rownames(rv_df) <- as.character(dates)
    rv_df           <- rv_df[complete.cases(rv_df), ]
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
          nfore         = forecast_h,
          model         = "TVP-VAR",
          connectedness = "Time",
          VAR_config    = list(
            TVPVAR = list(kappa1 = kappa1, kappa2 = kappa2,
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

    # extract net spillover
    net_raw <- fit$NET %||% fit$net %||% fit$Net
    if (is.null(net_raw)) {
      stop("Cannot find NET in fit object.")
    }

    net_df           <- as.data.frame(net_raw)
    colnames(net_df) <- available[seq_len(ncol(net_df))]
    net_df$date      <- as.Date(rownames(net_df), format = "%Y-%m-%d")

    # extract directional spillover
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

    # extract TCI
    tci_raw <- fit$TCI %||% fit$tci
    if (is.null(tci_raw)) {
      cat("  WARNING: TCI not found — filling with NA\n")
      tci_vals <- rep(NA_real_, nrow(rv_df))
    } else {
      tci_vals <- as.numeric(tci_raw)
    }

    tci_daily <- data.frame(
      date = as.Date(rownames(rv_df), format = "%Y-%m-%d"),
      tci  = tci_vals
    )
    # resample to quarterly
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
    tci_q <- to_quarterly(tci_daily)

    cat(sprintf("  %d quarters\n", nrow(net_q)))

    # build long-format spillover targets with lags

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
    return(list(
        spillover_targets = spillover_targets,
        tci_daily = tci_daily,
        tci_quarterly = tci_q,        # Changed from tci_quarterly
        directional_spillover = directional_targets # Changed from directional_spillover
    ))
}

# ------------------------------------------------------------------------------
# CLI entrypoint (for calling from Python via Rscript)
# ------------------------------------------------------------------------------
parse_args <- function(args) {
  out <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (startsWith(key, "--")) {
      if (i == length(args)) stop(paste("Missing value for", key))
      val <- args[[i + 1]]
      out[[substr(key, 3, nchar(key))]] <- val
      i <- i + 2
    } else {
      i <- i + 1
    }
  }
  out
}

write_df <- function(df, path) {
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  write.csv(df, file = path, row.names = FALSE)
}

if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  kv <- parse_args(args)

  input_path <- kv$input
  output_dir <- kv$output
  sectors_arg <- kv$sectors
  vol_window <- as.integer(kv$vol_window)
  forecast_h <- as.integer(kv$forecast_h)
  kappa1 <- as.numeric(kv$kappa1)
  kappa2 <- as.numeric(kv$kappa2)

  if (is.null(input_path) || is.null(output_dir)) {
    stop("Usage: Rscript tvp_var_spillover.R --input prices.csv --output out_dir --sectors XLF,XLE --vol_window 30 --forecast_h 10 --kappa1 0.99 --kappa2 0.99")
  }

  if (is.null(sectors_arg) || nchar(sectors_arg) == 0) {
    stop("Missing --sectors")
  }

  sectors <- unlist(strsplit(sectors_arg, ",", fixed = TRUE))

  prices_in <- read.csv(input_path, stringsAsFactors = FALSE, check.names = FALSE)
  if (!("date" %in% colnames(prices_in))) {
    stop("Input CSV must include a 'date' column.")
  }
  rownames(prices_in) <- prices_in$date
  prices_in$date <- NULL

  res <- calculate_spillover(
    prices_raw = prices_in,
    sectors = sectors,
    vol_window = vol_window,
    forecast_h = forecast_h,
    kappa1 = kappa1,
    kappa2 = kappa2
  )

  out_dir <- normalizePath(output_dir, mustWork = FALSE)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  write_df(res$spillover_targets, file.path(out_dir, "spillover_targets.csv"))
  write_df(res$tci_daily, file.path(out_dir, "tci_daily.csv"))
  write_df(res$tci_quarterly, file.path(out_dir, "tci_quarterly.csv"))
  write_df(res$directional_spillover, file.path(out_dir, "directional_spillover.csv"))
}