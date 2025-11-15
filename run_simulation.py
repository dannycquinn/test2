from __future__ import annotations

from microgrid_model import ScenarioInputs, generate_pdf_report, run_scenario
from microgrid_model.simulation import (
    BatteryInput,
    ExportInput,
    GasInput,
    GridInput,
    LoadInput,
    RenewableInput,
    ThresholdsInput,
)


def build_demo_scenario() -> ScenarioInputs:
    load = LoadInput(it_max_mw=80, avg_it_load_factor=0.85, pue=1.3)
    wind = RenewableInput(installed_mw=60, cost_gbp_per_kwh=0.04)
    solar = RenewableInput(installed_mw=40, cost_gbp_per_kwh=0.05)
    battery = BatteryInput(
        power_mw_discharge=50,
        power_mw_charge=50,
        energy_mwh=200,
        soc_min=0.1,
        soc_max=0.9,
        throughput_cost_gbp_per_kwh=0.01,
        allow_grid_charge=False,
    )
    gas = GasInput(installed_mw=30, base_cost_gbp_per_kwh_elec=0.15)
    grid = GridInput(import_limit_mw=90, flat_tariff_gbp_per_kwh=0.20)
    export = ExportInput(allow_export=True, export_price_gbp_per_kwh=0.04, export_carbon_treatment="displace_grid_avg")
    thresholds = ThresholdsInput(max_gas_share_percent=30)
    return ScenarioInputs(load=load, wind=wind, solar=solar, battery=battery, gas=gas, grid=grid, export=export, thresholds=thresholds)


def main() -> None:
    scenario = build_demo_scenario()
    result = run_scenario(scenario)

    annual = result.annual
    print("=== Annual Summary ===")
    print(f"Facility energy: {annual.total_facility_mwh:,.0f} MWh")
    print(f"Wind: {annual.total_wind_mwh:,.0f} MWh | Solar: {annual.total_solar_mwh:,.0f} MWh")
    print(f"Gas: {annual.total_gas_mwh:,.0f} MWh | Grid: {annual.total_grid_mwh:,.0f} MWh")
    print(f"Export: {annual.total_export_mwh:,.0f} MWh | Curtailment: {annual.total_curtail_mwh:,.0f} MWh")
    print(f"Blended cost: £{annual.blended_cost_per_kwh:.3f}/kWh")
    print(f"Carbon intensity: {annual.carbon_intensity_g_per_kwh:.0f} gCO2e/kWh")
    print("Flags:")
    for key, value in annual.flags.items():
        print(f"  {key}: {'Yes' if value else 'No'}")

    pdf_path = generate_pdf_report(result, scenario, "artifacts/microgrid_report.pdf")
    print(f"PDF report saved to {pdf_path}")


if __name__ == "__main__":
    main()
