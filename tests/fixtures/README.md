# Deterministic test inputs

These small, versioned JSON files are test specimens, not portfolio settings
or market-data inputs for run.py. Tests load them directly; no runtime override
of the user configuration or Git lookup is required.

- Provider fixtures contain 12 explicit SPX AM calls across four expiries,
  plus eight distinct price/reference instruments (20 merged requests and
  32 provider-symbol assignments).
- The portfolio fixture uses a synthetic 50/50 allocation.
- OIS fixtures contain two synthetic par quotes per currency, dated 9 October
  2026, for date selection and archive reuse tests.
- The Monte Carlo fixture starts on 9 October 2026; individual tests explicitly
  adjust their own simulation/calibration windows.

Assertions test these controlled examples, not the contents of config/.
