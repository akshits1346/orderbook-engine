"""
Estimates the two Avellaneda-Stoikov parameters that must come from
data, not from guessing: sigma (volatility) and kappa (how fast fill
likelihood decays as your quote moves away from the best price).

This file is split into two layers deliberately:
  1. Pure-math fitting functions (fit_kappa_from_distance_counts,
     estimate_sigma_from_mid_series) that take plain numbers in and
     numbers out -- these are the correctness-critical, fully
     unit-testable part.
  2. Data-extraction functions (estimate_sigma_from_events,
     estimate_kappa_from_events) that replay LOBSTER events through
     the real order book to produce the mid-price series and
     trade-distance data the layer-1 functions need.

IMPORTANT HONESTY NOTE: with only a small LOBSTER sample (the
10-row/4-row synthetic files used elsewhere in this repo), these
estimates are not statistically meaningful -- there just isn't enough
data for sigma or kappa to mean anything. This file is built and
tested against small data so the MATH is verified correct, but a real
calibration for an actual strategy comparison needs a much larger,
real LOBSTER pull (thousands of events, ideally a full trading day).
Treat any sigma/kappa produced from the small sample files in this
repo as "the code runs and the formula is right," not "this is a
usable volatility estimate."
"""
import math
import numpy as np

import lob_engine as lob


# --- Layer 1: pure-math fitting, fully unit-testable ---

def estimate_sigma_from_mid_series(mid_prices: list, times: list) -> float:
    """
    Volatility estimator: std of mid-price changes, scaled by the
    average time step, giving sigma in price-units-per-sqrt(second) --
    the units Avellaneda-Stoikov's formula expects (sigma^2 * (T-t)
    needs to be in price^2, so sigma must carry units of price/sqrt(time)).

    This is a simple estimator (not accounting for overnight jumps,
    intraday seasonality in volatility, or microstructure noise from
    bid-ask bounce) -- adequate for demonstrating the calibration
    pipeline works, not a production-grade volatility model.
    """
    if len(mid_prices) < 2:
        raise ValueError("need at least 2 mid-price observations to estimate volatility")

    diffs = np.diff(np.array(mid_prices, dtype=float))
    dt = np.diff(np.array(times, dtype=float))
    dt = np.where(dt <= 0, 1e-6, dt)  # guard against zero/negative gaps from duplicate timestamps

    # Scale each price change by sqrt(its own dt) before taking std,
    # since Brownian-motion-style volatility scales as sqrt(time) --
    # this normalizes irregularly-spaced observations onto a common
    # per-sqrt-second basis before computing the standard deviation.
    normalized = diffs / np.sqrt(dt)
    return float(np.std(normalized))


def fit_kappa_from_distance_counts(distances: list, counts: list) -> float:
    """
    Fits intensity(distance) = A * exp(-kappa * distance) by linear
    regression on log(counts) vs distance -- log-linearizing turns the
    exponential fit into an ordinary least-squares line fit, where the
    fitted slope is -kappa.

    Zero-count buckets are dropped before fitting (log(0) is undefined,
    and a bucket with zero observed fills provides no information about
    the decay rate, not evidence that intensity is truly zero there).
    """
    distances = np.array(distances, dtype=float)
    counts = np.array(counts, dtype=float)

    mask = counts > 0
    if mask.sum() < 2:
        raise ValueError("need at least 2 nonzero-count distance buckets to fit kappa")

    log_counts = np.log(counts[mask])
    fit_distances = distances[mask]

    # np.polyfit(x, y, 1) returns [slope, intercept] for y = slope*x + intercept
    slope, _intercept = np.polyfit(fit_distances, log_counts, 1)
    kappa = -slope
    if kappa <= 0:
        raise ValueError(
            f"fitted kappa is non-positive ({kappa:.4f}) -- intensity should "
            "DECREASE with distance from best price; a non-positive kappa means "
            "either the data doesn't show that pattern or there isn't enough of it"
        )
    return float(kappa)


# --- Layer 2: extract the raw data from a replayed LOBSTER event stream ---

def _replay_and_track(events):
    """Replays events through a fresh OrderBook, recording (time, mid)
    at every point mid is computable, and (distance_from_mid, side) for
    every execution event. Shared helper so sigma and kappa estimation
    replay the data exactly once each, consistently."""
    book = lob.OrderBook()
    mid_series = []
    execution_distances = []

    for event in events:
        lob.apply_message(book, event)
        bid, ask = book.best_bid(), book.best_ask()
        if bid is not None and ask is not None:
            mid = (bid + ask) // 2
            mid_series.append((event.time, mid))

            is_execution = event.type in (lob.MessageType.VisibleExecution, lob.MessageType.HiddenExecution)
            if is_execution:
                distance = abs(event.price - mid)
                execution_distances.append(distance)

    return mid_series, execution_distances


def estimate_sigma_from_events(events) -> float:
    mid_series, _ = _replay_and_track(events)
    if len(mid_series) < 2:
        raise ValueError("not enough mid-price observations in this event stream to estimate sigma")
    times = [t for t, _ in mid_series]
    mids = [m for _, m in mid_series]
    return estimate_sigma_from_mid_series(mids, times)


def estimate_kappa_from_events(events, tick_size: int = 100, max_ticks: int = 20) -> float:
    _, distances = _replay_and_track(events)
    if not distances:
        raise ValueError("no execution events in this stream -- can't estimate kappa without any fills to measure")

    # Bucket distances into ticks (0, 1, 2, ... ticks from mid) and count occurrences per bucket
    bucket_counts = {}
    for d in distances:
        bucket = d // tick_size
        if bucket <= max_ticks:
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    buckets = sorted(bucket_counts.keys())
    counts = [bucket_counts[b] for b in buckets]
    return fit_kappa_from_distance_counts(buckets, counts)
