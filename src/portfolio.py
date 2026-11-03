"""Portfolio input validation and the public analytical portfolio interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from copy import deepcopy
from pathlib import Path
import json
import math
import re

from .json_config import _unique_object, _invalid_constant

WEIGHT_TOLERANCE = 1e-8

if TYPE_CHECKING:
    # Type-only imports avoid coupling retrieval/configuration imports to the
    # optional numerical engines. Runtime imports remain inside public methods.
    from .data_cleaning import ProcessedArchive
    from .statistics import PortfolioStatistics
    from .monte_carlo import MonteCarloResult
    from .local_vol_real_world import RealWorldResult
    from .simulation_comparison import FinalAnalysis


@dataclass(frozen=True)
class PortfolioConfig:
    reporting_currency: str
    target_allocation: dict
    holdings: tuple
    reference_data: tuple
    nav: float | None

    @property
    def weights_ready(self):
        return bool(self.holdings) and all(h['weight'] is not None for h in self.holdings)


def load_portfolio(path, retrieval_config):
    """Weights are fractions of reporting-currency market value, not notional.

    Null weights are deliberate placeholders. Complete weights must add to one
    and match the configured allocation. No balancing or normalization occurs.
    """
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'),
                      object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    required = {'schema_version', 'reporting_currency', 'weight_basis', 'nav',
                'target_allocation', 'holdings', 'reference_data'}
    if not isinstance(data, dict) or not required <= data.keys() or set(data) - required - {'instructions'}:
        raise ValueError('Portfolio JSON contains missing or unknown fields.')
    if type(data['schema_version']) is not int or data['schema_version'] != 1:
        raise ValueError('Portfolio schema_version must be 1.')
    if not isinstance(data['reporting_currency'], str) or not re.fullmatch('[A-Z]{3}', data['reporting_currency']):
        raise ValueError('reporting_currency must be a three-letter currency code.')
    if data['weight_basis'] != 'reporting_currency_market_value':
        raise ValueError('Only reporting_currency_market_value weights are supported.')
    def number(value):
        return type(value) in (int, float) and math.isfinite(value)
    nav = data['nav']
    if nav is not None and (not number(nav) or nav <= 0):
        raise ValueError('nav must be a positive amount in reporting currency or null.')
    allocation = data['target_allocation']
    if (not isinstance(allocation, dict) or not allocation or
        any(not isinstance(k, str) or not k or not number(v) or not 0 <= v <= 1 for k, v in allocation.items()) or
        not math.isclose(math.fsum(allocation.values()), 1, abs_tol=WEIGHT_TOLERANCE, rel_tol=0)):
        raise ValueError('Target allocation must contain fractions adding to one.')
    requests = {r['id']: r for r in retrieval_config['requests']}
    holdings = data['holdings']
    if not isinstance(holdings, list) or not holdings:
        raise ValueError('Portfolio holdings must be a non-empty array.')
    seen, totals, missing = set(), dict.fromkeys(allocation, 0.0), False
    for h in holdings:
        if not isinstance(h, dict) or set(h) != {'request_id', 'bucket', 'weight'}:
            raise ValueError('Each holding needs request_id, bucket and weight.')
        key, bucket, weight = h['request_id'], h['bucket'], h['weight']
        if not isinstance(key, str) or key in seen or key not in requests:
            raise ValueError('Holding request_id must reference one distinct configured instrument.')
        if requests[key]['kind'] not in {'price', 'option'}:
            raise ValueError('Rates and FX observations belong in reference_data, not holdings.')
        if not isinstance(bucket, str) or bucket not in allocation:
            raise ValueError('Holding bucket must appear in target_allocation.')
        if weight is None: missing = True
        elif not number(weight) or not 0 <= weight <= 1:
            raise ValueError('Use long-only fractional weights between zero and one, or null.')
        else: totals[bucket] += weight
        seen.add(key)
    if any(totals[k] > allocation[k] + WEIGHT_TOLERANCE for k in totals):
        raise ValueError('Known holding weights exceed the target allocation.')
    if not missing:
        # Small rounding differences in separate buckets can add up to a material
        # error across the portfolio. Apply the same total-weight rule as MC.
        total = math.fsum(h['weight'] for h in holdings)
        if (not math.isclose(total, 1, abs_tol=WEIGHT_TOLERANCE, rel_tol=0)
                or any(not math.isclose(totals[k], allocation[k], abs_tol=WEIGHT_TOLERANCE, rel_tol=0) for k in totals)):
            raise ValueError('Complete holding weights must match target_allocation and add to one.')
    refs = data['reference_data']
    if not isinstance(refs, list) or any(not isinstance(k, str) or k not in requests for k in refs):
        raise ValueError('reference_data must contain configured request IDs.')
    if len(refs) != len(set(refs)) or seen.intersection(refs):
        raise ValueError('Reference data must be unique and separate from holdings.')
    return PortfolioConfig(data['reporting_currency'], dict(allocation), tuple(dict(h) for h in holdings), tuple(refs), nav)


@dataclass
class Portfolio:
    """Public analytical interface over one cleaned portfolio execution.

    The cleaned result remains the authority for holdings, weights, currencies
    and option-reference mappings. Construction does not calculate, download or
    write anything. Methods delegate to the existing engines; run.py still owns
    stage ordering, exception handling and persistence. In particular, the
    physical engine must validate the supplied risk-neutral evidence itself.
    """
    processed: ProcessedArchive
    source_folder: Path | None = None
    statistics: PortfolioStatistics | None = field(default=None, init=False, repr=False)
    simulations: dict[str, MonteCarloResult | RealWorldResult] = field(default_factory=dict, init=False, repr=False)
    final_analysis: FinalAnalysis | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        summary = getattr(self.processed, "summary", None)
        if not isinstance(summary, dict) or not isinstance(summary.get("portfolio"), dict):
            raise ValueError("Portfolio requires a cleaned portfolio definition.")
        if self.source_folder is not None:
            self.source_folder = Path(self.source_folder).resolve()

    @classmethod
    def from_processed(cls, processed, *, source_folder=None):
        """Use a cleaning result, without reloading today's portfolio JSON."""
        return cls(processed, source_folder)

    @property
    def definition(self):
        """A defensive view: editing this dictionary cannot change the inputs."""
        return deepcopy(self.processed.summary["portfolio"])

    @property
    def assets(self):
        """Analytical series whose statistics were computed on construction.

        AssetStatistics is the analytical price-series DataClass. Raw downloaded
        PriceSeries objects intentionally do not calculate unvalidated returns.
        No lazy calculation occurs merely by reading this property.
        """
        return dict(self.statistics.assets) if self.statistics is not None else {}

    def analyze(self, config_file):
        """Calculate existing statistics, without changing their definitions."""
        from .statistics import analyze_processed
        self.statistics = None  # A failed rerun must not expose an older success.
        self.final_analysis = None
        result = analyze_processed(self.processed, config_file)
        result.source_folder = self.source_folder
        self.statistics = result
        return result

    def simulate(self, config_file, *, method="historical", historical_config=None,
                 local_vol=None, surface=None, validation=None, progress=None):
        """Run one existing portfolio engine; never construct missing models.

        historical uses config_file as the historical MC JSON. physical_local_vol
        uses it as the physical JSON and requires historical_config for the shared
        drift/window/path policies. Risk-neutral MC is a preceding validation
        stage, not a third portfolio forecast. Supplied evidence is checked by
        the physical engine for identity, dates and the unchanged acceptance
        policy, including its explicitly configured diagnostic-only exception.
        """
        if method not in {"historical", "physical_local_vol"}:
            raise ValueError("Choose historical or physical_local_vol simulation.")
        self.simulations.pop(method, None)
        self.final_analysis = None
        if method == "historical":
            if any(x is not None for x in (historical_config, local_vol, surface, validation)):
                raise ValueError("Historical simulation does not consume local-volatility inputs.")
            from .monte_carlo import simulate_processed
            result = simulate_processed(self.processed, config_file)
            result.source_folder = self.source_folder
        else:
            if any(x is None for x in (historical_config, local_vol, surface, validation, self.source_folder)):
                raise ValueError("Physical local volatility requires a saved processing folder, historical configuration, surface, local volatility and risk-neutral validation.")
            from .local_vol_real_world import simulate_processed
            result = simulate_processed(self.processed, local_vol, surface, validation,
                historical_config, config_file, source_folder=self.source_folder, progress=progress)
        self.simulations[method] = result
        return result

    def build_final_analysis(self, stage_folders, config_file, *, issues=None):
        """Attach verified saved results from this portfolio's processing run.

        The reporter retains its checksum and transitive dependency checks.
        No folder discovery, model rerun or silent attachment of another
        portfolio is allowed. Missing optional stages remain legitimate.
        """
        from .simulation_comparison import build_analysis
        self.final_analysis = None
        folder = stage_folders.get("processing")
        if self.source_folder is None or folder is None or Path(folder).resolve() != self.source_folder:
            raise ValueError("Final analysis must reference this portfolio's saved processing folder.")
        result = build_analysis(str(self.processed.cob), stage_folders, config_file,
                                issues=issues, requested_portfolio=self.definition)
        # Matching weights alone do not establish that the report belongs to
        # these observations. Verify the saved cleaning identity as well.
        source = result.sources.get("processing") if result is not None else None
        if source is not None and source.summary.get("run_id") != self.processed.run_id:
            raise ValueError("Final analysis processing identity differs from the portfolio.")
        if result is not None and result.portfolio != self.definition:
            raise ValueError("Final analysis portfolio differs from the cleaned portfolio.")
        self.final_analysis = result
        return result

    def report(self, *, sections=None, include_figures=True):
        """Return final Markdown, or the existing statistics-only report.

        Section selection applies to Final Analysis. Before that stage, the
        default report is explicitly descriptive historical statistics only;
        simulation results are never silently presented as part of that text.
        """
        if self.final_analysis is not None:
            return self.final_analysis.report(sections=sections, include_figures=include_figures)
        if sections is not None:
            raise ValueError("Section selection requires a built Final Analysis.")
        if self.statistics is None:
            raise ValueError("No analysis is available; call analyze() or build_final_analysis().")
        return self.statistics.report()

    def plots_report(self, destination):
        """Return plot paths using existing writers and their safety checks.

        With Final Analysis, destination is the reports root and the immutable
        bundle is verified/published. Otherwise destination is the statistics
        plot directory. Neither path downloads, recalibrates or simulates.
        """
        if self.final_analysis is not None:
            return self.final_analysis.plots_report(destination)
        if self.statistics is None:
            raise ValueError("No analysis is available for plotting.")
        return self.statistics.plots_report(destination)
