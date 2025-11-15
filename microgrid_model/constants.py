from __future__ import annotations

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]

HOURS_IN_MONTH = [31 * 24, 28 * 24, 31 * 24, 30 * 24, 31 * 24, 30 * 24,
                   31 * 24, 31 * 24, 30 * 24, 31 * 24, 30 * 24, 31 * 24]

# Default monthly capacity factors from the specification
DEFAULT_WIND_CF = [
    0.50, 0.45, 0.45, 0.33, 0.38, 0.28, 0.28, 0.28, 0.33, 0.38, 0.50, 0.55
]
DEFAULT_SOLAR_CF = [
    0.03, 0.05, 0.10, 0.16, 0.18, 0.18, 0.16, 0.14, 0.10, 0.07, 0.03, 0.02
]

HOURS_PER_YEAR = 8760

DAYLIGHT_HOURS_APPROX = [8, 9, 11, 13, 14, 15, 15, 14, 12, 10, 9, 8]

DAYLIGHT_START_HOUR = [
    8, 8, 7, 6, 5, 5, 5, 6, 6, 7, 8, 8
]
