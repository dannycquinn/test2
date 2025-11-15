from __future__ import annotations

import dataclasses
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .constants import (
    DAYLIGHT_HOURS_APPROX,
    DAYLIGHT_START_HOUR,
    DEFAULT_SOLAR_CF,
    DEFAULT_WIND_CF,
    HOURS_IN_MONTH,
    HOURS_PER_YEAR,
    MONTHS,
)


def _validate_length(name: str, values: Sequence[float], expected: int) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} must contain {expected} values, received {len(values)}")


@dataclass
class LoadInput:
    it_max_mw: float
    avg_it_load_factor: float
    pue: float | Sequence[float] = 1.3
    load_profile_timeseries_mw: Optional[Sequence[float]] = None


@dataclass
class RenewableInput:
    installed_mw: float
    capacity_factor_monthly: Optional[Sequence[float]] = None
    hourly_available_power_timeseries: Optional[Sequence[float]] = None
    cost_gbp_per_kwh: float = 0.0
    emission_kg_per_kwh: float = 0.0
    diurnal_shape_24h_fraction: Optional[Sequence[float]] = None


@dataclass
class BatteryInput:
    power_mw_discharge: float
    power_mw_charge: float
    energy_mwh: float
    soc_init: Optional[float] = None
    soc_min: float = 0.1
    soc_max: float = 0.9
    eff_charge: float = 0.95
    eff_discharge: float = 0.95
    throughput_cost_gbp_per_kwh: float = 0.0
    charge_from: Sequence[str] = dataclasses.field(default_factory=lambda: ["wind", "solar"])
    allow_grid_charge: bool = False
    allow_gas_charge: bool = False


@dataclass
class GasInput:
    installed_mw: float
    mode: str = "as_needed"
    expected_runtime_hours: float = 0.0
    base_cost_gbp_per_kwh_elec: float = 0.12
    emission_factor_kg_per_kwh_elec: float = 0.40


@dataclass
class GridInput:
    import_limit_mw: float
    flat_tariff_gbp_per_kwh: float = 0.25
    time_of_use_tariff_24h: Optional[Sequence[float]] = None
    hourly_tariff: Optional[Sequence[float]] = None
    grid_emission_kg_per_kwh: float | Sequence[float] = 0.233
    monthly_energy_cap_mwh: Optional[Sequence[float]] = None


@dataclass
class ExportInput:
    allow_export: bool = False
    export_price_gbp_per_kwh: float = 0.0
    export_carbon_treatment: str = "none"


@dataclass
class ETSInput:
    ets_price_gbp_per_tonne: float = 0.0

@dataclass
class ThresholdsInput:
    max_gas_share_percent: float = 100.0


@dataclass
class ScenarioInputs:
    load: LoadInput
    wind: RenewableInput
    solar: RenewableInput
    battery: BatteryInput
    gas: GasInput
    grid: GridInput
    export: ExportInput = field(default_factory=ExportInput)
    ets: ETSInput = field(default_factory=ETSInput)
    thresholds: ThresholdsInput = field(default_factory=ThresholdsInput)


@dataclass
class MonthlyDispatch:
    month: str
    facility_mwh: float
    wind_mwh: float
    solar_mwh: float
    aer_pre_mwh: float
    aer_post_mwh: float
    ere_mwh: float
    gas_mwh: float
    grid_mwh: float
    export_mwh: float
    curtail_mwh: float
    battery_charge_mwh: float
    battery_discharge_mwh: float
    battery_losses_mwh: float
    battery_round_trip_efficiency: float
    unserved_mwh: float
    monthly_cost_gbp: float
    monthly_carbon_tonnes: float


@dataclass
class AnnualSummary:
    total_facility_mwh: float
    total_wind_mwh: float
    total_solar_mwh: float
    total_gas_mwh: float
    total_grid_mwh: float
    total_export_mwh: float
    total_curtail_mwh: float
    total_battery_losses_mwh: float
    total_cost_gbp: float
    blended_cost_per_kwh: float
    total_carbon_tonnes: float
    carbon_intensity_g_per_kwh: float
    gas_runtime_hours: float
    battery_throughput_mwh: float
    longest_battery_only_run_hours: int
    energy_balance_error_mwh: float
    flags: Dict[str, bool]


@dataclass
class SimulationResult:
    hourly_data: Dict[str, np.ndarray]
    monthly: List[MonthlyDispatch]
    annual: AnnualSummary


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _expand_monthly(value: Sequence[float]) -> np.ndarray:
    _validate_length("monthly sequence", value, 12)
    expanded: List[float] = []
    for month_index, hours in enumerate(HOURS_IN_MONTH):
        expanded.extend([value[month_index]] * hours)
    return np.array(expanded)


def _prepare_tariff(grid: GridInput) -> np.ndarray:
    if grid.hourly_tariff is not None:
        _validate_length("hourly_tariff", grid.hourly_tariff, HOURS_PER_YEAR)
        return np.array(grid.hourly_tariff)
    if grid.time_of_use_tariff_24h is not None:
        _validate_length("time_of_use_tariff_24h", grid.time_of_use_tariff_24h, 24)
        return np.array(list(grid.time_of_use_tariff_24h) * (HOURS_PER_YEAR // 24))
    return np.full(HOURS_PER_YEAR, grid.flat_tariff_gbp_per_kwh, dtype=float)


def _prepare_grid_emissions(grid: GridInput) -> np.ndarray:
    emissions = grid.grid_emission_kg_per_kwh
    if isinstance(emissions, Sequence) and not isinstance(emissions, (str, bytes)):
        if len(emissions) == 12:
            return _expand_monthly(emissions)
        _validate_length("grid_emission_kg_per_kwh", emissions, HOURS_PER_YEAR)
        return np.array(emissions)
    return np.full(HOURS_PER_YEAR, float(emissions), dtype=float)


def _prepare_monthly_caps(cap: Optional[Sequence[float]]) -> Optional[List[float]]:
    if cap is None:
        return None
    if len(cap) != 12:
        raise ValueError("monthly_energy_cap_mwh must contain 12 values")
    return list(cap)


def _normalise_profile(profile: np.ndarray, target: float) -> np.ndarray:
    current = profile.sum()
    if current <= 0 or target <= 0:
        return np.zeros_like(profile)
    return profile * (target / current)


def _synthesise_solar(cfg: RenewableInput) -> np.ndarray:
    monthly_cf = cfg.capacity_factor_monthly or DEFAULT_SOLAR_CF
    _validate_length("solar monthly CF", monthly_cf, 12)
    if cfg.installed_mw == 0:
        return np.zeros(HOURS_PER_YEAR)
    profile = np.zeros(HOURS_PER_YEAR)
    hour_index = 0
    for month_index, hours in enumerate(HOURS_IN_MONTH):
        daylight_hours = DAYLIGHT_HOURS_APPROX[month_index]
        sunrise = DAYLIGHT_START_HOUR[month_index]
        sunset = sunrise + daylight_hours
        month_profile = []
        diurnal_shape = cfg.diurnal_shape_24h_fraction
        if diurnal_shape is not None:
            _validate_length("diurnal_shape_24h_fraction", diurnal_shape, 24)
            total_shape = sum(diurnal_shape)
            if not math.isclose(total_shape, 1.0, rel_tol=1e-3):
                diurnal_shape = [v / total_shape for v in diurnal_shape]
        days = hours // 24
        for day in range(days):
            noise = random.uniform(0.95, 1.15)
            for hour in range(24):
                if diurnal_shape is not None:
                    base = diurnal_shape[hour]
                elif sunrise <= hour < sunset:
                    progress = (hour - sunrise) / max(1, (sunset - sunrise))
                    base = math.sin(math.pi * progress)
                else:
                    base = 0.0
                month_profile.append(max(0.0, base * noise))
        month_profile = np.array(month_profile)
        target = cfg.installed_mw * monthly_cf[month_index] * hours
        if month_profile.sum() == 0:
            profile[hour_index: hour_index + hours] = 0.0
        else:
            scaled = _normalise_profile(month_profile, target)
            profile[hour_index: hour_index + hours] = scaled
        hour_index += hours
    return profile


def _synthesise_wind(cfg: RenewableInput) -> np.ndarray:
    monthly_cf = cfg.capacity_factor_monthly or DEFAULT_WIND_CF
    _validate_length("wind monthly CF", monthly_cf, 12)
    if cfg.installed_mw == 0:
        return np.zeros(HOURS_PER_YEAR)
    profile = np.zeros(HOURS_PER_YEAR)
    hour_index = 0
    for month_index, hours in enumerate(HOURS_IN_MONTH):
        base = []
        days = hours // 24
        for day in range(days):
            for hour in range(24):
                t = day * 24 + hour
                weather_wave = 0.5 + 0.5 * math.sin(2 * math.pi * t / (24 * 5))
                diurnal = 0.55 + 0.1 * math.cos(2 * math.pi * (hour / 24))
                noise = random.uniform(0.8, 1.2)
                val = weather_wave * diurnal * noise
                base.append(max(0.0, val))
        month_profile = np.array(base)
        target = cfg.installed_mw * monthly_cf[month_index] * hours
        if month_profile.sum() == 0:
            profile[hour_index: hour_index + hours] = 0.0
        else:
            scaled = _normalise_profile(month_profile, target)
            profile[hour_index: hour_index + hours] = np.clip(scaled, 0, cfg.installed_mw)
        hour_index += hours
    return profile


def _prepare_resource_profile(cfg: RenewableInput, default_cf: Sequence[float], synthesiser) -> np.ndarray:
    if cfg.hourly_available_power_timeseries is not None:
        _validate_length("hourly_available_power_timeseries", cfg.hourly_available_power_timeseries, HOURS_PER_YEAR)
        return np.minimum(np.array(cfg.hourly_available_power_timeseries), cfg.installed_mw)
    if cfg.capacity_factor_monthly is None:
        cfg.capacity_factor_monthly = list(default_cf)
    return synthesiser(cfg)


def _prepare_load(load: LoadInput) -> Tuple[np.ndarray, np.ndarray, float]:
    it_avg_mw = load.it_max_mw * load.avg_it_load_factor
    if load.load_profile_timeseries_mw is not None:
        _validate_length("load_profile_timeseries_mw", load.load_profile_timeseries_mw, HOURS_PER_YEAR)
        it_load = np.array(load.load_profile_timeseries_mw, dtype=float)
    else:
        it_load = np.full(HOURS_PER_YEAR, it_avg_mw, dtype=float)

    pue = load.pue
    if isinstance(pue, Sequence) and not isinstance(pue, (str, bytes)):
        if len(pue) == 12:
            pue_hourly = _expand_monthly(pue)
        else:
            _validate_length("pue", pue, HOURS_PER_YEAR)
            pue_hourly = np.array(pue, dtype=float)
    else:
        pue_hourly = np.full(HOURS_PER_YEAR, float(pue), dtype=float)

    facility_load = it_load * pue_hourly
    monthly_mwh = []
    idx = 0
    for hours in HOURS_IN_MONTH:
        monthly_mwh.append(float(facility_load[idx: idx + hours].sum()))
        idx += hours
    facility_energy = facility_load.sum()
    return facility_load, np.array(monthly_mwh), facility_energy


def _prepare_monthly_caps_remaining(grid: GridInput) -> Optional[List[float]]:
    if grid.monthly_energy_cap_mwh is None:
        return None
    _validate_length("monthly_energy_cap_mwh", grid.monthly_energy_cap_mwh, 12)
    return list(grid.monthly_energy_cap_mwh)


# ---------------------------------------------------------------------------
# Dispatch algorithm
# ---------------------------------------------------------------------------

def run_scenario(inputs: ScenarioInputs) -> SimulationResult:
    facility_load, facility_monthly_mwh, facility_energy = _prepare_load(inputs.load)
    solar_profile = _prepare_resource_profile(inputs.solar, DEFAULT_SOLAR_CF, _synthesise_solar)
    wind_profile = _prepare_resource_profile(inputs.wind, DEFAULT_WIND_CF, _synthesise_wind)

    tariff = _prepare_tariff(inputs.grid)
    grid_emissions = _prepare_grid_emissions(inputs.grid)
    monthly_grid_cap_remaining = _prepare_monthly_caps_remaining(inputs.grid)

    battery = inputs.battery
    soc = battery.soc_init if battery.soc_init is not None else battery.energy_mwh * 0.5
    soc_min = battery.energy_mwh * battery.soc_min
    soc_max = battery.energy_mwh * battery.soc_max
    soc = min(max(soc, soc_min), soc_max)
    stored_carbon = 0.0

    gas_energy_budget = None
    if inputs.gas.mode == "cap_by_expected_hours":
        gas_energy_budget = inputs.gas.installed_mw * inputs.gas.expected_runtime_hours
    gas_used_total = 0.0

    arrays: Dict[str, List[float]] = {
        "load_mw": [],
        "wind_mw": [],
        "solar_mw": [],
        "wind_used_mwh": [],
        "solar_used_mwh": [],
        "battery_charge_mwh": [],
        "battery_discharge_mwh": [],
        "gas_mwh": [],
        "grid_mwh": [],
        "export_mwh": [],
        "curtail_mwh": [],
        "unserved_mwh": [],
        "soc_mwh": [],
        "renewable_cost_gbp": [],
        "battery_cost_gbp": [],
        "gas_cost_gbp": [],
        "grid_cost_gbp": [],
        "export_revenue_gbp": [],
        "emissions_kg": [],
    }

    gas_runtime_hours = 0
    grid_limit_breached = False
    soc_violation = False
    unserved_flag = False

    month_hour_index = []
    for month_index, hours in enumerate(HOURS_IN_MONTH):
        month_hour_index.extend([month_index] * hours)
    month_hour_index = np.array(month_hour_index)

    monthly_charge = [0.0] * 12
    monthly_discharge = [0.0] * 12
    monthly_cost = [0.0] * 12
    monthly_carbon = [0.0] * 12
    monthly_aer_pre = []
    monthly_aer_post = [0.0] * 12
    monthly_ere = []

    monthly_start_hours = [0]
    for h in HOURS_IN_MONTH:
        monthly_start_hours.append(monthly_start_hours[-1] + h)

    for m in range(12):
        hours = HOURS_IN_MONTH[m]
        wind_mwh = inputs.wind.installed_mw * (inputs.wind.capacity_factor_monthly or DEFAULT_WIND_CF)[m] * hours
        solar_mwh = inputs.solar.installed_mw * (inputs.solar.capacity_factor_monthly or DEFAULT_SOLAR_CF)[m] * hours
        ren = wind_mwh + solar_mwh
        facility = facility_monthly_mwh[m]
        aer_pre = max(0.0, facility - ren)
        ere = max(0.0, ren - facility)
        monthly_aer_pre.append(aer_pre)
        monthly_ere.append(ere)

    ets_adder = inputs.gas.emission_factor_kg_per_kwh_elec / 1000 * inputs.ets.ets_price_gbp_per_tonne

    for hour in range(HOURS_PER_YEAR):
        month_idx = month_hour_index[hour]
        load = facility_load[hour]
        wind = wind_profile[hour]
        solar = solar_profile[hour]
        ren_total = wind + solar
        load_remaining = load
        wind_used = 0.0
        solar_used = 0.0
        wind_to_battery = 0.0
        solar_to_battery = 0.0
        battery_charge_total = 0.0
        battery_discharge = 0.0
        gas_used_for_load = 0.0
        gas_used_for_charge = 0.0
        grid_used_for_load = 0.0
        grid_used_for_charge = 0.0
        export = 0.0
        curtail = 0.0
        unserved = 0.0

        # Use renewables first
        if ren_total > 0:
            ren_used = min(load_remaining, ren_total)
            if ren_used > 0:
                wind_share = wind / ren_total if ren_total else 0.0
                wind_used = min(wind, ren_used * wind_share)
                solar_used = max(0.0, ren_used - wind_used)
            load_remaining -= ren_used

        # Surplus renewables for charging/export/curtailment
        surplus_wind = max(0.0, wind - wind_used)
        surplus_solar = max(0.0, solar - solar_used)
        surplus_total = surplus_wind + surplus_solar
        if surplus_total > 0:
            charge_potential = min(battery.power_mw_charge, surplus_total)
            max_energy_room = (soc_max - soc) / battery.eff_charge
            charge_energy = min(charge_potential, max(0.0, max_energy_room))
            if charge_energy > 0:
                # allocate proportionally between wind/solar respecting charge_from
                alloc_wind = 0.0
                alloc_solar = 0.0
                if "wind" in battery.charge_from and surplus_wind > 0:
                    alloc_wind = min(charge_energy, surplus_wind)
                if "solar" in battery.charge_from and surplus_solar > 0:
                    remaining = charge_energy - alloc_wind
                    alloc_solar = min(remaining, surplus_solar)
                if alloc_wind + alloc_solar < charge_energy:
                    charge_energy = alloc_wind + alloc_solar
                energy_added = charge_energy * battery.eff_charge
                if energy_added > 0:
                    soc += energy_added
                    battery_charge_total += charge_energy
                    wind_to_battery += alloc_wind
                    solar_to_battery += alloc_solar
                    carbon_energy = (
                        alloc_wind * inputs.wind.emission_kg_per_kwh +
                        alloc_solar * inputs.solar.emission_kg_per_kwh
                    )
                    carbon_factor = (carbon_energy / charge_energy) if charge_energy > 0 else 0.0
                    stored_carbon += energy_added * carbon_factor
                    surplus_wind -= alloc_wind
                    surplus_solar -= alloc_solar
                    surplus_total = surplus_wind + surplus_solar
            if surplus_total > 0:
                if inputs.export.allow_export:
                    export = surplus_total
                else:
                    curtail = surplus_total

        # Battery discharge if deficit remains
        soc_before_discharge = soc
        stored_carbon_before = stored_carbon
        if load_remaining > 0 and soc > soc_min:
            discharge_cap = min(battery.power_mw_discharge, load_remaining)
            max_discharge = (soc - soc_min) * battery.eff_discharge
            energy_from_battery = min(discharge_cap, max_discharge)
            if energy_from_battery > 0:
                load_remaining -= energy_from_battery
                soc_reduction = energy_from_battery / battery.eff_discharge
                soc -= soc_reduction
                if soc < soc_min - 1e-6:
                    soc_violation = True
                battery_discharge = energy_from_battery
                if soc_before_discharge > 0 and stored_carbon_before > 0:
                    carbon_release = stored_carbon_before * (soc_reduction / soc_before_discharge)
                    stored_carbon = max(0.0, stored_carbon_before - carbon_release)
                else:
                    carbon_release = 0.0
            else:
                carbon_release = 0.0
        else:
            carbon_release = 0.0

        # Gas dispatch
        gas_available = inputs.gas.installed_mw
        if gas_energy_budget is not None:
            gas_available = min(gas_available, max(0.0, gas_energy_budget - gas_used_total))
        if load_remaining > 0 and gas_available > 0:
            gas_used_for_load = min(load_remaining, gas_available)
            load_remaining -= gas_used_for_load
            gas_available -= gas_used_for_load
            gas_used_total += gas_used_for_load

        # Grid import for load
        grid_available = inputs.grid.import_limit_mw
        if monthly_grid_cap_remaining is not None:
            grid_available = min(grid_available, monthly_grid_cap_remaining[month_idx])
        if load_remaining > 0 and grid_available > 0:
            grid_used_for_load = min(load_remaining, grid_available)
            load_remaining -= grid_used_for_load
            grid_available -= grid_used_for_load
            if monthly_grid_cap_remaining is not None:
                monthly_grid_cap_remaining[month_idx] -= grid_used_for_load
                monthly_grid_cap_remaining[month_idx] = max(0.0, monthly_grid_cap_remaining[month_idx])
        if load_remaining > 0:
            grid_limit_breached = True

        # Any remaining deficit is unserved energy
        if load_remaining > 1e-9:
            unserved = load_remaining
            load_remaining = 0.0
            unserved_flag = True

        # Optional grid charging
        available_charge_room = min(battery.power_mw_charge - battery_charge_total,
                                    (soc_max - soc) / battery.eff_charge)
        available_charge_room = max(0.0, available_charge_room)
        if inputs.battery.allow_grid_charge and available_charge_room > 0 and grid_available > 0:
            grid_charge_cap = grid_available
            if monthly_grid_cap_remaining is not None:
                grid_charge_cap = min(grid_charge_cap, monthly_grid_cap_remaining[month_idx])
            grid_charge = min(available_charge_room, grid_charge_cap)
            if grid_charge > 0:
                energy_added = grid_charge * battery.eff_charge
                soc += energy_added
                battery_charge_total += grid_charge
                stored_carbon += energy_added * grid_emissions[hour]
                grid_used_for_charge = grid_charge
                grid_available -= grid_charge
                if monthly_grid_cap_remaining is not None:
                    monthly_grid_cap_remaining[month_idx] -= grid_charge
                    monthly_grid_cap_remaining[month_idx] = max(0.0, monthly_grid_cap_remaining[month_idx])
                available_charge_room = min(battery.power_mw_charge - battery_charge_total,
                                             (soc_max - soc) / battery.eff_charge)
                available_charge_room = max(0.0, available_charge_room)

        # Optional gas charging
        if inputs.battery.allow_gas_charge and available_charge_room > 0 and gas_available > 0:
            gas_charge = min(available_charge_room, gas_available)
            if gas_charge > 0:
                energy_added = gas_charge * battery.eff_charge
                soc += energy_added
                battery_charge_total += gas_charge
                stored_carbon += energy_added * inputs.gas.emission_factor_kg_per_kwh_elec
                gas_used_for_charge = gas_charge
                gas_used_total += gas_charge
                gas_available -= gas_charge

        gas_used = gas_used_for_load + gas_used_for_charge
        grid_used = grid_used_for_load + grid_used_for_charge
        if gas_used > 0:
            gas_runtime_hours += 1

        # Costs and emissions
        renewable_cost = (
            (wind_used + wind_to_battery) * inputs.wind.cost_gbp_per_kwh +
            (solar_used + solar_to_battery) * inputs.solar.cost_gbp_per_kwh
        )
        battery_cost = battery_discharge * battery.throughput_cost_gbp_per_kwh
        gas_cost = gas_used * (inputs.gas.base_cost_gbp_per_kwh_elec + ets_adder)
        grid_cost = grid_used * tariff[hour]
        export_revenue = export * inputs.export.export_price_gbp_per_kwh

        gas_emissions = gas_used * inputs.gas.emission_factor_kg_per_kwh_elec
        grid_emissions_hour = grid_used * grid_emissions[hour]
        emissions = gas_emissions + grid_emissions_hour + carbon_release
        if inputs.export.export_carbon_treatment == "displace_grid_avg" and export > 0:
            emissions -= export * grid_emissions[hour]
        emissions = max(0.0, emissions)

        # Track battery charge/discharge monthly
        monthly_charge[month_idx] += battery_charge_total
        monthly_discharge[month_idx] += battery_discharge
        monthly_cost[month_idx] += renewable_cost + battery_cost + gas_cost + grid_cost - export_revenue
        monthly_carbon[month_idx] += emissions / 1000
        monthly_aer_post[month_idx] += gas_used_for_load + grid_used_for_load + unserved

        arrays["load_mw"].append(load)
        arrays["wind_mw"].append(wind)
        arrays["solar_mw"].append(solar)
        arrays["wind_used_mwh"].append(wind_used)
        arrays["solar_used_mwh"].append(solar_used)
        arrays["battery_charge_mwh"].append(battery_charge_total)
        arrays["battery_discharge_mwh"].append(battery_discharge)
        arrays["gas_mwh"].append(gas_used)
        arrays["grid_mwh"].append(grid_used)
        arrays["export_mwh"].append(export)
        arrays["curtail_mwh"].append(curtail)
        arrays["unserved_mwh"].append(unserved)
        arrays["soc_mwh"].append(soc)
        arrays["renewable_cost_gbp"].append(renewable_cost)
        arrays["battery_cost_gbp"].append(battery_cost)
        arrays["gas_cost_gbp"].append(gas_cost)
        arrays["grid_cost_gbp"].append(grid_cost)
        arrays["export_revenue_gbp"].append(export_revenue)
        arrays["emissions_kg"].append(emissions)

    monthly_table: List[MonthlyDispatch] = []
    for idx, month in enumerate(MONTHS):
        charge = monthly_charge[idx]
        discharge = monthly_discharge[idx]
        losses = max(0.0, charge - discharge)
        rte = discharge / charge if charge > 0 else 0.0
        start = monthly_start_hours[idx]
        end = monthly_start_hours[idx + 1]
        monthly_table.append(
            MonthlyDispatch(
                month=month,
                facility_mwh=facility_monthly_mwh[idx],
                wind_mwh=inputs.wind.installed_mw * (inputs.wind.capacity_factor_monthly or DEFAULT_WIND_CF)[idx] * HOURS_IN_MONTH[idx],
                solar_mwh=inputs.solar.installed_mw * (inputs.solar.capacity_factor_monthly or DEFAULT_SOLAR_CF)[idx] * HOURS_IN_MONTH[idx],
                aer_pre_mwh=monthly_aer_pre[idx],
                aer_post_mwh=min(monthly_aer_pre[idx], monthly_aer_post[idx]),
                ere_mwh=monthly_ere[idx],
                gas_mwh=float(np.sum(arrays["gas_mwh"][start:end])) if arrays["gas_mwh"] else 0.0,
                grid_mwh=float(np.sum(arrays["grid_mwh"][start:end])) if arrays["grid_mwh"] else 0.0,
                export_mwh=float(np.sum(arrays["export_mwh"][start:end])) if arrays["export_mwh"] else 0.0,
                curtail_mwh=float(np.sum(arrays["curtail_mwh"][start:end])) if arrays["curtail_mwh"] else 0.0,
                battery_charge_mwh=charge,
                battery_discharge_mwh=discharge,
                battery_losses_mwh=losses,
                battery_round_trip_efficiency=rte,
                unserved_mwh=float(np.sum(arrays["unserved_mwh"][start:end])) if arrays["unserved_mwh"] else 0.0,
                monthly_cost_gbp=monthly_cost[idx],
                monthly_carbon_tonnes=monthly_carbon[idx],
            )
        )

    total_wind = float(np.sum(arrays["wind_mw"]))
    total_solar = float(np.sum(arrays["solar_mw"]))
    total_gas = float(np.sum(arrays["gas_mwh"]))
    total_grid = float(np.sum(arrays["grid_mwh"]))
    total_export = float(np.sum(arrays["export_mwh"]))
    total_curtail = float(np.sum(arrays["curtail_mwh"]))
    total_charge = float(np.sum(arrays["battery_charge_mwh"]))
    total_discharge = float(np.sum(arrays["battery_discharge_mwh"]))
    total_losses = max(0.0, total_charge - total_discharge)
    total_cost = float(np.sum(arrays["renewable_cost_gbp"]) + np.sum(arrays["battery_cost_gbp"]) +
                        np.sum(arrays["gas_cost_gbp"]) + np.sum(arrays["grid_cost_gbp"]) -
                        np.sum(arrays["export_revenue_gbp"]))
    total_emissions_tonnes = float(np.sum(arrays["emissions_kg"]) / 1000)

    blended_cost = total_cost / facility_energy if facility_energy > 0 else 0.0
    carbon_intensity = (total_emissions_tonnes * 1000) / facility_energy if facility_energy > 0 else 0.0

    gas_share = (total_gas / facility_energy * 100) if facility_energy > 0 else 0.0
    gas_share_flag = gas_share > inputs.thresholds.max_gas_share_percent

    gas_runtime = float(sum(1 for g in arrays["gas_mwh"] if g > 0))

    # Battery throughput
    battery_throughput = float(total_charge + total_discharge)

    # Longest battery-only window
    longest_window = 0
    current_window = 0
    for g, grd, unserved in zip(arrays["gas_mwh"], arrays["grid_mwh"], arrays["unserved_mwh"]):
        if g == 0 and grd == 0 and unserved == 0:
            current_window += 1
            longest_window = max(longest_window, current_window)
        else:
            current_window = 0

    energy_balance = (total_wind + total_solar + total_gas + total_grid - total_export - total_curtail - total_losses)
    energy_balance_error = energy_balance - facility_energy

    flags = {
        "gas_share_exceeded": gas_share_flag,
        "grid_import_limit_breached": grid_limit_breached,
        "unserved_energy": unserved_flag,
        "soc_violation": soc_violation,
    }

    annual = AnnualSummary(
        total_facility_mwh=facility_energy,
        total_wind_mwh=total_wind,
        total_solar_mwh=total_solar,
        total_gas_mwh=total_gas,
        total_grid_mwh=total_grid,
        total_export_mwh=total_export,
        total_curtail_mwh=total_curtail,
        total_battery_losses_mwh=total_losses,
        total_cost_gbp=total_cost,
        blended_cost_per_kwh=blended_cost,
        total_carbon_tonnes=total_emissions_tonnes,
        carbon_intensity_g_per_kwh=carbon_intensity * 1000,
        gas_runtime_hours=gas_runtime,
        battery_throughput_mwh=battery_throughput,
        longest_battery_only_run_hours=longest_window,
        energy_balance_error_mwh=energy_balance_error,
        flags=flags,
    )

    hourly_arrays = {k: np.array(v) for k, v in arrays.items()}
    return SimulationResult(hourly_data=hourly_arrays, monthly=monthly_table, annual=annual)
