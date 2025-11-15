# Microgrid Simulation Engine

This repository contains a full-year (8,760 hour) microgrid simulation tailored for hyperscale data centres. The engine models wind, solar, battery storage, gas generators, grid imports, and optional energy exports to generate hourly dispatch, monthly reports, annual summaries, and a branded PDF document.

## Key capabilities
- Hourly dispatch of renewables, battery, gas, grid, export, and curtailment with strict SoC and import limits.
- Synthetic wind and solar profiles that honour user-supplied monthly capacity factors with realistic diurnal/seasonal variability.
- Monthly reporting of facility energy, Additional Energy Required (pre/post), Excess Renewable Energy, export/curtailment, battery cycling, costs, and carbon.
- Annual KPIs including blended £/kWh, carbon intensity, gas runtime, battery throughput/losses, longest battery-only window, and system flags (gas share, grid breaches, unserved energy, SoC violations).
- Detailed carbon/cost accounting including ETS adders, export treatments, and battery provenance tracking.
- Automated PDF report with all mandated charts, annual summary, formulas appendix, and scenario metadata timestamp.

## Getting started
1. Install dependencies (a virtual environment is recommended):
   ```bash
   pip install -r requirements.txt
   ```
2. Run the demonstration scenario (generates console output plus a PDF report under `artifacts/microgrid_report.pdf`):
   ```bash
   python run_simulation.py
   ```

## Customising scenarios
`run_simulation.py` builds an example `ScenarioInputs` object covering:
- Load assumptions (IT max MW, load factor, PUE, optional hourly load profile).
- Wind and solar build-out (installed MW, monthly CFs or hourly timeseries, LCOE, emissions).
- Battery envelope (power/energy, efficiency, throughput cost, charge-source rules, SoC limits).
- Gas plant sizing, runtime constraints, fuel cost, emissions, and ETS pricing.
- Grid tariffs (flat, time-of-use, or hourly) with import limits and monthly caps.
- Export settings (price, carbon displacement).
- Policy thresholds (e.g., max gas share).

Modify these inputs or construct new `ScenarioInputs` instances to simulate alternative portfolios. The `microgrid_model.simulation.run_scenario` function returns a `SimulationResult` containing:
- `hourly_data`: NumPy arrays for every dispatch/cost/carbon stream.
- `monthly`: A list of `MonthlyDispatch` records with the mandated KPI columns.
- `annual`: An `AnnualSummary` with energy balances, cost/carbon metrics, runtime statistics, and compliance flags.

Feed the result plus inputs into `microgrid_model.reporting.generate_pdf_report` to create the multi-page branded PDF.

## Project structure
```
microgrid_model/
  constants.py          # Date/time constants and default capacity factors
  simulation.py         # Dataclasses, profile synthesis, hourly dispatcher, reporting structures
  reporting.py          # Chart generation and PDF composition helpers
run_simulation.py       # Example scenario runner + PDF export
requirements.txt        # Python dependencies (numpy, matplotlib, reportlab)
```

## Notes
- Units: MW for power, MWh for energy, kgCO2e/kWh (reported in gCO2e/kWh), currency in GBP.
- The simulation enforces energy balance validation, SoC boundaries, import caps, and `AER_post ≤ AER_pre` for every month.
- Renewable generators can accept user-specified hourly profiles; otherwise, the synthesiser guarantees monthly CF preservation with realistic intra-day variability.
