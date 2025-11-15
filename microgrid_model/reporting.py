from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas

from .simulation import MonthlyDispatch, ScenarioInputs, SimulationResult


FIGURE_STYLE = {
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
}


def _monthly_dataframe(monthly: List[MonthlyDispatch]) -> Dict[str, List[float]]:
    data = {field.name: [] for field in dataclasses.fields(MonthlyDispatch)}
    for entry in monthly:
        for field in data:
            data[field].append(getattr(entry, field))
    return data


def _save_chart(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_monthly_energy(months: List[str], facility: List[float], wind: List[float], solar: List[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(months, wind, label="Wind", color="#7fcdbb")
    ax.bar(months, solar, bottom=wind, label="Solar", color="#feb24c")
    ax.plot(months, facility, label="Facility Load", color="#2c7fb8", marker="o")
    ax.set_title("Monthly Renewable Output vs Facility Load")
    ax.set_ylabel("MWh")
    ax.legend(loc="upper right")
    ax.tick_params(axis="x", rotation=45)
    _save_chart(fig, path)


def _plot_aer(months: List[str], aer_pre: List[float], aer_post: List[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(months, aer_pre, label="AER_pre", marker="o")
    ax.plot(months, aer_post, label="AER_post", marker="s")
    ax.set_ylabel("MWh")
    ax.set_title("Additional Energy Required")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    _save_chart(fig, path)


def _plot_ere(months: List[str], ere: List[float], export: List[float], curtail: List[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(months, ere, label="Excess Renewable Energy", color="#fee391")
    ax.bar(months, export, label="Export", color="#80cdc1")
    ax.bar(months, curtail, bottom=export, label="Curtailment", color="#df65b0")
    ax.set_ylabel("MWh")
    ax.set_title("ERE vs Export/Curtailment")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    _save_chart(fig, path)


def _plot_battery(months: List[str], charge: List[float], discharge: List[float], losses: List[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    width = 0.25
    x = np.arange(len(months))
    ax.bar(x - width, charge, width, label="Batt In")
    ax.bar(x, discharge, width, label="Batt Out")
    ax.bar(x + width, losses, width, label="Losses")
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45)
    ax.set_ylabel("MWh")
    ax.set_title("Monthly Battery Cycling")
    ax.legend()
    _save_chart(fig, path)


def _plot_carbon_intensity(months: List[str], carbon_tonnes: List[float], facility: List[float], path: Path) -> None:
    intensity = [((c * 1000) / f if f else 0.0) for c, f in zip(carbon_tonnes, facility)]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(months, intensity, color="#3182bd")
    ax.set_ylabel("gCO2e/kWh")
    ax.set_title("Monthly Carbon Intensity")
    ax.tick_params(axis="x", rotation=45)
    _save_chart(fig, path)


def _plot_cost_breakdown(hourly_data: Dict[str, np.ndarray], path: Path) -> None:
    totals = {
        "Renewables": float(np.sum(hourly_data["renewable_cost_gbp"])),
        "Battery": float(np.sum(hourly_data["battery_cost_gbp"])),
        "Gas": float(np.sum(hourly_data["gas_cost_gbp"])),
        "Grid": float(np.sum(hourly_data["grid_cost_gbp"])),
        "Export Credit": float(-np.sum(hourly_data["export_revenue_gbp"])),
    }
    labels = list(totals.keys())
    values = list(totals.values())
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Annual Cost Breakdown")
    _save_chart(fig, path)


def _draw_section_heading(pdf: canvas.Canvas, text: str, y: float) -> float:
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(2 * cm, y, text)
    return y - 0.5 * cm


def generate_pdf_report(result: SimulationResult, inputs: ScenarioInputs, output_path: str) -> str:
    """Create a multi-page branded PDF report with charts and summaries."""

    monthly_df = _monthly_dataframe(result.monthly)
    months = monthly_df["month"]
    charts_dir = Path("artifacts/charts")

    plt.rcParams.update(FIGURE_STYLE)

    _plot_monthly_energy(months, monthly_df["facility_mwh"], monthly_df["wind_mwh"], monthly_df["solar_mwh"], charts_dir / "energy.png")
    _plot_aer(months, monthly_df["aer_pre_mwh"], monthly_df["aer_post_mwh"], charts_dir / "aer.png")
    _plot_ere(months, monthly_df["ere_mwh"], monthly_df["export_mwh"], monthly_df["curtail_mwh"], charts_dir / "ere.png")
    _plot_battery(months, monthly_df["battery_charge_mwh"], monthly_df["battery_discharge_mwh"], monthly_df["battery_losses_mwh"], charts_dir / "battery.png")
    _plot_carbon_intensity(months, monthly_df["monthly_carbon_tonnes"], monthly_df["facility_mwh"], charts_dir / "carbon.png")
    _plot_cost_breakdown(result.hourly_data, charts_dir / "cost.png")

    pdf_path = Path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    width, height = A4

    timestamp = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Cover page
    pdf.setFont("Helvetica-Bold", 20)
    pdf.drawString(2 * cm, height - 3 * cm, "Data Centre Microgrid Report")
    pdf.setFont("Helvetica", 12)
    pdf.drawString(2 * cm, height - 4 * cm, f"Scenario generated on {timestamp}")
    pdf.drawString(2 * cm, height - 4.7 * cm, f"IT Max Load: {inputs.load.it_max_mw:.1f} MW | PUE: {inputs.load.pue}")
    pdf.drawString(2 * cm, height - 5.4 * cm, f"Wind MW: {inputs.wind.installed_mw:.1f} | Solar MW: {inputs.solar.installed_mw:.1f} | Battery: {inputs.battery.energy_mwh:.1f} MWh")
    pdf.drawString(2 * cm, height - 6.1 * cm, f"Gas MW: {inputs.gas.installed_mw:.1f} | Grid limit: {inputs.grid.import_limit_mw:.1f} MW")
    pdf.showPage()

    # Charts page
    y = height - 2 * cm
    for title, image in [
        ("Monthly Renewable vs Load", charts_dir / "energy.png"),
        ("AER", charts_dir / "aer.png"),
        ("ERE", charts_dir / "ere.png"),
    ]:
        y = _draw_section_heading(pdf, title, y)
        pdf.drawImage(str(image), 2 * cm, y - 7 * cm, width=16 * cm, height=6 * cm, preserveAspectRatio=True)
        y -= 7.5 * cm
        if y < 8 * cm:
            pdf.showPage()
            y = height - 2 * cm

    pdf.showPage()

    y = height - 2 * cm
    for title, image in [
        ("Battery Cycling", charts_dir / "battery.png"),
        ("Carbon Intensity", charts_dir / "carbon.png"),
        ("Cost Breakdown", charts_dir / "cost.png"),
    ]:
        y = _draw_section_heading(pdf, title, y)
        pdf.drawImage(str(image), 2 * cm, y - 7 * cm, width=16 * cm, height=6 * cm, preserveAspectRatio=True)
        y -= 7.5 * cm
        if y < 8 * cm:
            pdf.showPage()
            y = height - 2 * cm

    pdf.showPage()

    # Annual summary page
    y = height - 2 * cm
    y = _draw_section_heading(pdf, "Annual Summary", y)
    summary_lines = [
        f"Facility energy: {result.annual.total_facility_mwh:,.0f} MWh",
        f"Renewables: Wind {result.annual.total_wind_mwh:,.0f} MWh | Solar {result.annual.total_solar_mwh:,.0f} MWh",
        f"Gas: {result.annual.total_gas_mwh:,.0f} MWh (runtime {result.annual.gas_runtime_hours:.0f} h)",
        f"Grid: {result.annual.total_grid_mwh:,.0f} MWh",
        f"Export: {result.annual.total_export_mwh:,.0f} MWh | Curtailment: {result.annual.total_curtail_mwh:,.0f} MWh",
        f"Battery throughput: {result.annual.battery_throughput_mwh:,.0f} MWh (losses {result.annual.total_battery_losses_mwh:,.0f} MWh)",
        f"Blended cost: £{result.annual.blended_cost_per_kwh:.3f}/kWh (total £{result.annual.total_cost_gbp:,.0f})",
        f"Carbon intensity: {result.annual.carbon_intensity_g_per_kwh:.0f} gCO2e/kWh",
        f"Longest battery-only run: {result.annual.longest_battery_only_run_hours} h",
        f"Energy balance error: {result.annual.energy_balance_error_mwh:.2f} MWh",
    ]
    pdf.setFont("Helvetica", 11)
    for line in summary_lines:
        pdf.drawString(2 * cm, y, line)
        y -= 0.7 * cm

    y -= 0.5 * cm
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(2 * cm, y, "Flags")
    y -= 0.6 * cm
    pdf.setFont("Helvetica", 11)
    for name, flag in result.annual.flags.items():
        pdf.drawString(2.5 * cm, y, f"{name.replace('_', ' ').title()}: {'Yes' if flag else 'No'}")
        y -= 0.6 * cm

    pdf.showPage()

    # Appendix
    y = height - 2 * cm
    y = _draw_section_heading(pdf, "Appendix: Key Formulae", y)
    pdf.setFont("Helvetica", 10)
    appendix_lines = [
        "Facility load (hourly) = IT Load × PUE",
        "AER_pre = max(0, Facility MWh − (Wind + Solar) MWh)",
        "AER_post = Gas + Grid + Unserved after dispatch",
        "Battery losses = Batt In − Batt Out",
        "Blended cost = Total cost / Facility MWh",
        "Carbon intensity = Total emissions / Facility kWh",
    ]
    for line in appendix_lines:
        pdf.drawString(2 * cm, y, f"• {line}")
        y -= 0.6 * cm

    pdf.setFont("Helvetica", 10)
    pdf.drawString(2 * cm, y - 0.5 * cm, "Scenario metadata captured at run time. Charts generated using hourly simulation outputs.")
    pdf.showPage()
    pdf.save()

    return str(pdf_path)
