"""Final offline synthesis of the current execution's saved analysis.

This module deliberately imports no provider, calibration or simulation engine.
It consumes checksummed artifacts, never searches for the 'latest' folder, and
never infers a successful model from a file merely existing. Missing optional
stages are ordinary report outcomes; corrupt supplied artifacts are integrity
errors. Neither is allowed to erase the independent historical evidence.

The common comparison unit is terminal portfolio return V(T)/V(0)-1. Physical
and historical engines retain their own clocks. We permit only an explicitly
labelled approximate terminal comparison when policy and metadata allow it;
we do not interpolate paths or compare their differently sampled drawdowns.
"""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import os
import re
from html.parser import HTMLParser
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties

from .portfolio import WEIGHT_TOLERANCE
from .mc_diagnostics import probability_of_loss, probability_label, confidence_label, _tail_mean


STAGES = {
    "processing": "Cleaned data", "statistics": "Historical statistics",
    "monte_carlo": "Historical portfolio Monte Carlo",
    "option_calibration": "OIS, forwards and individual IVs",
    "volatility_surface": "Fitted implied volatility",
    "local_volatility": "Local volatility and boundary validation",
    "local_vol_mc": "Risk-neutral underlying validation",
    "local_vol_real_world": "Physical local-volatility portfolio",
}


def _bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode("utf-8")


def _hash(path):
    # Stream large path archives instead of copying their bytes into memory.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_directory(staging, target):
    """Retain atomic publication through short Windows scanner/sync locks.

    This follows the existing storage writer's bounded backoff. It does not
    replace an existing report or change permissions; persistent failures are
    still raised to the caller. These are I/O retries, not model parameters.
    """
    for attempt in range(7):
        try:
            os.rename(staging,target)
            return
        except PermissionError:
            if attempt == 6 or target.exists():
                raise
            time.sleep(min(.05*2**attempt,1.0))


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError("Final-report configuration must be an object")
    expected = {"schema_version", "enabled", "primary_return_type", "reference_model",
                "confidence_levels", "sampling_confidence", "allow_approximate_terminal_horizon",
                "maximum_relative_horizon_difference", "material_loss_probability_difference",
                "minimum_expected_tail_paths", "maximum_chart_points", "surface_display_points", "simulation_plots"}
    if set(cfg) - expected - {"instructions","open_report_after_run","plot_windows_after_run"} or expected - set(cfg):
        raise ValueError("Invalid final-report configuration keys")
    if type(cfg["schema_version"]) is not int or cfg["schema_version"] != 1 or cfg["primary_return_type"] not in ("simple", "log"):
        raise ValueError("Unsupported final-report schema or return type")
    cfg.setdefault("open_report_after_run", False)
    windows = cfg.setdefault("plot_windows_after_run", [])
    if (not isinstance(windows,list) or
            any(not isinstance(w,str) or not re.fullmatch("[a-z][a-z0-9_]*",w) for w in windows)
            or len(windows) != len(set(windows))):
        raise ValueError("plot_windows_after_run must list distinct chart families")
    for key in ("enabled", "allow_approximate_terminal_horizon","open_report_after_run"):
        if not isinstance(cfg[key], bool):
            raise ValueError(f"{key} must be boolean")
    for key in ("sampling_confidence", "maximum_relative_horizon_difference", "material_loss_probability_difference"):
        if isinstance(cfg[key], bool) or not isinstance(cfg[key], (int, float)) or not 0 < cfg[key] < 1:
            raise ValueError(f"{key} must lie strictly between zero and one")
    for key in ("minimum_expected_tail_paths", "maximum_chart_points", "surface_display_points"):
        if type(cfg[key]) is not int or cfg[key] < 2:
            raise ValueError(f"{key} must be an integer of at least two")
    levels = cfg["confidence_levels"]
    if not isinstance(levels, list) or not levels or any(
            type(x) not in (float, int) or not 0.5 < x < 1 for x in levels):
        raise ValueError("Invalid tail confidence levels")
    if len(set(levels)) != len(levels):
        raise ValueError("Duplicate tail confidence levels")
    if not isinstance(cfg["reference_model"], str) or not cfg["reference_model"]:
        raise ValueError("A reference model name is required")
    plots = cfg["simulation_plots"]
    if not isinstance(plots, dict) or set(plots) != {"sample_paths", "histogram_bins", "fan_quantiles", "tail_display_fraction", "loss_thresholds", "smiles_per_figure"}:
        raise ValueError("Invalid simulation plot policy")
    for key in ("sample_paths", "histogram_bins", "smiles_per_figure"):
        if type(plots[key]) is not int or plots[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    q = plots["fan_quantiles"]
    if not isinstance(q, list) or len(q) != 2 or any(type(x) not in (int, float) for x in q) or not 0 < q[0] < .5 < q[1] < 1:
        raise ValueError("Fan quantiles must bracket the median")
    f = plots["tail_display_fraction"]
    if type(f) not in (int, float) or not 0 < f < .5:
        raise ValueError("Tail display fraction must lie between zero and one half")
    losses = plots["loss_thresholds"]
    if not isinstance(losses, list) or not losses or any(type(x) not in (int, float) or not 0 < x < 1 for x in losses) or len(set(losses)) != len(losses):
        raise ValueError("Loss thresholds must be distinct fractions between zero and one")
    return {k: v for k, v in cfg.items() if k != "instructions"}


@dataclass
class Source:
    folder: Path
    manifest: dict
    summary: dict
    files: dict
    fingerprint: str
    user_view: dict | None = None

    def table(self, key):
        """Load only recorded CSVs; identifiers remain strings, including 'NA'."""
        record = self.files.get(key) or self.files.get(key + ".csv")
        if record is None:
            return pd.DataFrame()
        try:
            # Read as text first so instrument identifiers such as '0005' are
            # never silently converted to numbers. Plot/metric columns are
            # converted explicitly at their point of use.
            return pd.read_csv(self.folder / record, dtype=str, keep_default_na=False)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()


def _read_source(folder, cob, processing_folder):
    folder = Path(folder).resolve()
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("complete") is not True:
        raise ValueError("Source was not completely saved")
    if not isinstance(manifest.get("files"), dict):
        raise ValueError("Source file inventory is missing or malformed")
    files = {}
    records = dict(manifest["files"])
    # Cleaning stores its report checksum separately; later stages include it
    # under files. Honor both existing formats without changing their writers.
    if isinstance(manifest.get("report"), dict):
        records.setdefault("report", manifest["report"])
    for key, entry in records.items():
        name, digest = (entry["file"], entry["sha256"]) if isinstance(entry, dict) else (key, entry)
        target = (folder / name).resolve()
        if not target.is_relative_to(folder) or target == manifest_path or not target.is_file():
            raise ValueError("Unsafe or missing source artifact")
        if _hash(target) != digest:
            raise ValueError(f"Source checksum mismatch: {name}")
        files[key] = name
    summary = manifest
    if "summary.json" in files:
        summary = json.loads((folder / files["summary.json"]).read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or summary.get("identity") != manifest.get("identity"):
            raise ValueError("Summary and manifest identities differ")
    identity = summary.get("identity", {})
    if not isinstance(identity, dict):
        raise ValueError("Source identity is malformed")
    # Reject malformed containers before they reach comparison or rendering.
    # Optional absent metadata is allowed; a supplied null/settings scalar is
    # not an executable configuration and must not poison independent stages.
    for key in ('settings','mc_config','config','portfolio'):
        if key in identity and not isinstance(identity[key],dict):
            raise ValueError(f"Malformed source identity: {key}")
    for meta in (summary,identity.get('settings',{}),identity.get('mc_config',{})):
        if meta.get('calibration') is not None and not isinstance(meta['calibration'],dict):
            raise ValueError('Malformed source calibration metadata')
    saved_cob = summary.get("cob", manifest.get("archive_cob", identity.get("cob")))
    if saved_cob != cob:
        raise ValueError("Source COB is missing or differs from the requested report COB")
    if processing_folder is not None:
        root = Path(processing_folder).resolve()
        if not folder.is_relative_to(root):
            raise ValueError("Source belongs to a different processed run")
        pid = identity.get("processing_id")
        if pid is not None and pid != root.name:
            raise ValueError("Processing identity mismatch")
        declared = identity.get("source_folder")
        if declared and Path(declared).resolve() != root:
            raise ValueError("Declared processing folder mismatch")
    source = Source(folder, manifest, summary, files, _hash(manifest_path))
    # Validate the columns consumed by this reporter. Missing optional tables
    # are acceptable; a supplied table with a different schema is not silently
    # interpreted. Empty tables are legitimate unavailable-stage artifacts.
    schemas = {
        "statistics": {"scope", "instrument_id", "return_type"},
        "dependence": {"instrument_i", "instrument_j", "correlation"},
        "rolling_volatility": {"scope", "instrument_id", "date"},
        "daily_statistics": {"scope", "instrument_id", "date", "drawdown"},
        "path_statistics": {"scope", "instrument_id", "path_status"},
        "surface_grid": {"group_id", "years", "k", "iv"},
        "local_vol_grid": {"group_id", "years", "k", "status", "local_volatility"},
    }
    for key, required in schemas.items():
        frame = source.table(key)
        if len(frame) and not required.issubset(frame.columns):
            raise ValueError(f"Malformed report input table: {key}")
        if key == "dependence" and len(frame) and frame.duplicated(
                [c for c in ("instrument_i", "instrument_j", "return_type") if c in frame]).any():
            raise ValueError("Duplicate dependence estimates")
    boundary = source.table("boundaries")
    if len(boundary):
        for key in ("paths_visiting_wings", "maximum_local_volatility"):
            if key in boundary and not np.isfinite(pd.to_numeric(boundary[key], errors="coerce")).all():
                raise ValueError("Malformed boundary diagnostics")
    return source


def _numbers(frame, column):
    return pd.to_numeric(frame[column], errors="coerce")


def _chosen(frame, cfg):
    if "return_type" in frame:
        return frame[frame.return_type == cfg["primary_return_type"]]
    return frame


def _signature(portfolio):
    # We never normalize the weights or silently omit an unavailable holding.
    holdings = portfolio.get("holdings", [])
    pairs = sorted((str(x["request_id"]), float(x["weight"])) for x in holdings)
    if not pairs or len({x[0] for x in pairs}) != len(pairs) or not all(
            math.isfinite(w) and 0 <= w <= 1 for _, w in pairs) or not math.isclose(
                math.fsum(w for _, w in pairs), 1, abs_tol=WEIGHT_TOLERANCE, rel_tol=0):
        raise ValueError("Invalid or incomplete portfolio weights")
    return pairs


def _same_calibration(historical, physical):
    left, right = historical.table("calibration_returns"), physical.table("calibration_returns")
    if "calibration_set" in left:
        left = left[left.calibration_set == "baseline"].drop(columns="calibration_set")
    columns = ["return_start_date", "date"]
    factors = sorted(c for c in left if c.startswith(("asset::", "fx::")))
    if not factors or set(left) != set(right) or not len(left) or len(left) != len(right):
        return False
    left, right = (x.sort_values(columns).reset_index(drop=True) for x in (left, right))
    return left[columns].equals(right[columns]) and np.allclose(
        left[factors].astype(float), right[factors].astype(float), rtol=1e-12, atol=1e-15)


def _compatibility(historical, physical, cfg):
    """No pooled ranking unless the economic comparison is actually defined."""
    if historical is None:
        return "standalone", "No historical simulation available for comparison"
    hs, ps = historical.summary, physical.summary
    hi, pi = hs["identity"], ps["identity"]
    hc, pc = hi["settings"], pi["mc_config"]
    checks = {
        "processing identity": hi.get("processing_id") == pi.get("processing_id") and bool(hi.get("processing_id")),
        "portfolio weights": _signature(hi["portfolio"]) == _signature(pi["portfolio"]),
        "currency": hs.get("simulation_currency") == ps.get("simulation_currency"),
        "investment convention": hc.get("portfolio_method") == pc.get("portfolio_method") == "buy_and_hold",
        "initial wealth": hc.get("initial_value") == pc.get("initial_value"),
        "forecast dates": all(hc.get(k) == pc.get(k) and hc.get(k) for k in ("forecast_start", "forecast_end")),
        "calibration observations": _same_calibration(historical, physical),
    }
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        return "not_comparable", "Different or missing " + ", ".join(failed)
    clock = ps.get("clock_comparison", {})
    htime, ptime = clock.get("historical_business_years"), clock.get("local_vol_elapsed_years")
    # NaN comparisons are always false: without explicit finite validation a
    # malformed clock could accidentally pass the maximum-difference gate.
    if any(type(x) not in (int,float) or not math.isfinite(x) or x <= 0 for x in (htime,ptime)) or clock.get("historical_session_steps") != hs.get("steps"):
        return "not_comparable", "Missing or inconsistent clock reconciliation"
    gap = abs(ptime / htime - 1)
    reason = f"Terminal horizon difference {gap:.4%}; exchange sessions versus ACT/365F. Intraday endpoints differ."
    if not cfg["allow_approximate_terminal_horizon"] or gap > cfg["maximum_relative_horizon_difference"]:
        return "not_comparable", reason
    return "approximate_terminal_only", reason


def _metrics(values, initial, cfg):
    """Compute identically defined terminal metrics for both physical engines.

    Loss L = 1 - V(T)/V(0). VaR is its empirical alpha quantile; ES is
    the mean of the worst exact (1-alpha) sample mass, with a fractional
    boundary observation, matching the existing simulation diagnostics. Negative VaR is
    retained when even the relevant lower-wealth quantile is a gain.
    Mean and Wilson intervals measure finite simulation sampling only.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all() or (values < 0).any() or not math.isfinite(initial) or not initial > 0:
        raise ValueError("Invalid terminal wealth sample")
    returns = values / initial - 1
    if not np.isfinite(returns).all():
        raise ValueError("Terminal returns overflow the reporting scale")
    n = len(values)
    # Same roundoff protection as the simulation engines: a flat deterministic
    # path must not become a loss because of floating point subtraction.
    p = probability_of_loss(-returns)
    z = NormalDist().inv_cdf((1 + cfg["sampling_confidence"]) / 2)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    radius = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    mean = float(np.mean(returns)); se = float(np.std(returns, ddof=1) / math.sqrt(n))
    out = dict(paths=n, mean_return=mean, median_return=float(np.median(returns)),
               mean_mc_standard_error=se, mean_ci_low=mean-z*se, mean_ci_high=mean+z*se,
               probability_loss=p, loss_probability_low=max(0., center-radius),
               loss_probability_high=min(1., center+radius))
    for alpha in cfg["confidence_levels"]:
        loss = -returns
        threshold = float(np.quantile(loss, alpha))
        out[f"loss_var_{probability_label(alpha)}"] = threshold
        out[f"loss_es_{probability_label(alpha)}"] = _tail_mean(loss, alpha)
        out[f"expected_tail_paths_{probability_label(alpha)}"] = n * (1-alpha)
    if not all(math.isfinite(value) for value in out.values()):
        raise ValueError("Non-finite terminal risk metrics")
    return out


@dataclass
class FinalAnalysis:
    cob: str
    cfg: dict
    sources: dict
    availability: list
    comparison: pd.DataFrame
    samples: dict
    findings: list
    identity: dict
    portfolio: dict
    run_id: str
    figures: dict = field(default_factory=dict)
    forecast_bands: pd.DataFrame = field(default_factory=pd.DataFrame)
    loss_thresholds: pd.DataFrame = field(default_factory=pd.DataFrame)
    chart_notes: list = field(default_factory=list)

    def report(self, *, sections=None, include_figures=True):
        """Return Markdown from the same content as HTML, without model reruns.

        Sections use HTML IDs: overview, modeling, forecast, vol, history,
        sources. PNG links are relative to the final bundle; save() or
        plots_report() publishes those files. Equations remain embedded SVGs.
        """
        if type(include_figures) is not bool:
            raise ValueError("include_figures must be boolean")
        # Validate selection before spending time rendering figures/equations.
        _MarkdownReport({}, sections, include_figures)
        if not self.figures:
            if include_figures:
                self.figures = _figures(self)
            else:
                # Hiding images must not hide numerical tables or warnings.
                # Preparation reads saved results only and draws no figures.
                _prepare_report_data(self)
        return _markdown(self, _render(self), sections, include_figures)

    def show(self, report_folder):
        """Open the completed offline HTML. Display consumes no simulation inputs."""
        import webbrowser
        file = Path(report_folder).resolve() / "report.html"
        if file.parent.name != self.run_id or not file.is_file():
            raise ValueError("Expected the saved HTML report for this analysis.")
        try:
            return webbrowser.open(file.as_uri(), new=2)
        except (OSError, webbrowser.Error):
            return False

    def show_plot_windows(self, report_folder):
        """Launch the native viewer using the images from this saved analysis."""
        from .plot_windows import launch
        folder = Path(report_folder).resolve()
        if folder.name != self.run_id:
            raise ValueError("Expected the saved folder for this analysis.")
        return launch(folder, self.cfg.get("plot_windows_after_run", []))

    def plots_report(self, reports_root):
        """Publish/verify the existing bundle and return its saved PNG paths."""
        folder = self.save(reports_root)
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        return [folder / name for name in manifest["files"] if name.endswith(".png")]

    def save(self, reports_root):
        """Publish a new immutable report; never overwrite a source or old run."""
        root = Path(reports_root).resolve()
        if date.fromisoformat(self.cob).isoformat() != self.cob or not self.run_id.isalnum():
            raise ValueError("Unsafe final-report output identity")
        for source in self.sources.values():
            if root.is_relative_to(source.folder):
                raise ValueError("Final reports must be separate from source artifacts")
        parent = root / self.cob / "final_analysis"
        parent.mkdir(parents=True, exist_ok=True)
        target = parent / self.run_id
        if target.is_symlink() or target.resolve().parent != parent.resolve():
            raise ValueError("Unsafe final-report output path")
        if target.exists():
            manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("identity") != self.identity or not manifest.get("complete"):
                raise ValueError("Existing final report has a different identity")
            if not {"report.html", "report.md", "summary.json", "comparison.csv", "availability.csv", "forecast_bands.csv", "loss_thresholds.csv"}.issubset(manifest.get("files", {})):
                raise ValueError("Existing final report is incomplete")
            for name, checksum in manifest["files"].items():
                file = (target / name).resolve()
                if not file.is_relative_to(target.resolve()) or _hash(file) != checksum:
                    raise ValueError("Existing final report failed verification")
            return target
        if not self.figures:
            self.figures = _figures(self)
        # Build completely in a temporary sibling. An interrupted report can
        # never be mistaken for an already completed result on the next run.
        with tempfile.TemporaryDirectory(prefix=".final-", dir=parent) as temp:
            staging = Path(temp) / "output"; staging.mkdir()
            self.comparison.to_csv(staging / "comparison.csv", index=False)
            self.forecast_bands.to_csv(staging / "forecast_bands.csv", index=False)
            self.loss_thresholds.to_csv(staging / "loss_thresholds.csv", index=False)
            pd.DataFrame(self.availability).to_csv(staging / "availability.csv", index=False)
            summary = dict(schema_version=1, scope="final_report_only", cob=self.cob, run_id=self.run_id,
                           identity=self.identity, portfolio=self.portfolio,
                           findings=self.findings, availability=self.availability,
                           chart_notes=self.chart_notes,
                           modeling_context=_modeling_context(self),
                           comparison=json.loads(self.comparison.to_json(orient="records")),
                           downloads_performed=False, simulations_performed=False)
            (staging / "summary.json").write_bytes(_bytes(summary))
            for name, (_, content) in self.figures.items():
                (staging / (name + ".png")).write_bytes(content)
            page = _render(self)
            (staging / "report.html").write_text(page, encoding="utf-8")
            (staging / "report.md").write_text(_markdown(self, page), encoding="utf-8")
            # Check again at publication: a user may have edited or removed an
            # upstream artifact while figures were rendering. Do not publish
            # a report that combines pre-edit metadata with post-edit values.
            for source in self.sources.values():
                current = _read_source(source.folder, self.cob, None)
                if current.fingerprint != source.fingerprint:
                    raise ValueError("A source changed during final-report generation")
            files = {f.name: _hash(f) for f in staging.iterdir()}
            (staging / "manifest.json").write_bytes(_bytes(dict(complete=True, identity=self.identity, files=files)))
            _publish_directory(staging, target)
        return target


def build_analysis(cob, stage_folders, config_file, *, issues=None, requested_portfolio=None):
    """Build from explicit current-run folders, including a legitimate empty run.

    A missing options stage is a capability limitation, not a reason to discard
    historical results. A supplied but corrupt stage is explicitly an error.
    No directory is discovered by sorting timestamps or walking old outputs.
    """
    cob = date.fromisoformat(str(cob)).isoformat()
    cfg = load_config(config_file)
    if not cfg["enabled"]:
        return None
    if set(stage_folders) - set(STAGES):
        raise ValueError("Unknown final-report source stage")
    sources, availability = {}, []
    issues = issues or {}
    processing_folder = stage_folders.get("processing")
    for name, label in STAGES.items():
        folder = stage_folders.get(name)
        row = dict(stage=name, label=label, status="not_available", reason=issues.get(name, "No result supplied by this execution"))
        if folder is not None:
            try:
                src = _read_source(folder, cob, processing_folder)
                sources[name] = src
                meta = src.summary
                status = meta.get("status", meta.get("portfolio_status", "saved"))
                if isinstance(meta.get("summary"), dict):
                    status = meta["summary"].get("status", "saved")
                row.update(status=status, reason=meta.get("reason") or "Verified saved artifacts")
                if name in issues:
                    row.update(status="saved_with_execution_error", reason=issues[name])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status="integrity_error", reason=str(exc))
        elif name in issues:
            row["status"] = "execution_error"
        availability.append(row)
    # Enforce the actual model chain, not only a shared folder. These checks
    # prevent accidental mixing of alternative fits within the same COB.
    dependencies=(("volatility_surface", "calibration_identity", "option_calibration"),
                                ("local_volatility", "source_surface_id", "volatility_surface"),
                                ("local_vol_mc", "source_local_id", "local_volatility"),
                                ("local_vol_real_world", "local_id", "local_volatility"),
                                ("local_vol_real_world", "surface_id", "volatility_surface"),
                                ("local_vol_real_world", "validation_id", "local_vol_mc"))
    # Propagate rejection to every downstream stage, regardless of edge order.
    # Intentionally omitted sources are permitted for standalone reporting;
    # a supplied source that fails integrity checks is a different situation.
    changed=True
    while changed:
        changed=False
        for child,key,parent in dependencies:
            if child not in sources:
                continue
            reason=None
            if parent in sources:
                # Surface fits store the full calibration identity; later
                # stages reference their parent's output-directory identifier.
                expected=sources[parent].manifest['identity'] if key=='calibration_identity' else sources[parent].folder.name
                if sources[child].manifest["identity"].get(key) != expected:
                    reason=f"Does not reference the supplied {parent} result"
            elif stage_folders.get(parent) is not None:
                reason=f"Required supplied {parent} result was rejected"
            if reason:
                del sources[child]
                next(r for r in availability if r["stage"] == child).update(
                    status="integrity_error", reason=reason)
                changed=True
    rows, samples, findings = [], {}, []
    historical = sources.get("monte_carlo")
    physical = sources.get("local_vol_real_world")
    if historical is not None:
        historical.user_view = _saved_view(historical.summary)
    for stage, src in (("monte_carlo", historical), ("local_vol_real_world", physical)):
        if src is None or not src.summary.get("simulation_performed"):
            continue
        try:
            s, identity = src.summary, src.summary["identity"]
            _signature(identity["portfolio"])
            settings = identity["settings" if stage == "monte_carlo" else "mc_config"]
            initial = float(settings["initial_value"])
            groups = []
            if stage == "monte_carlo":
                table = src.table("terminal_values")
                table = _verified_view_samples(src, table[table.scope == "portfolio"], settings)
                for model, group in table.groupby("model", sort=False):
                    if group.path.duplicated().any() or len(group) != settings["paths"]:
                        raise ValueError("Incomplete or duplicate terminal portfolio paths")
                    calibration_set = group.calibration_set.iloc[0] if "calibration_set" in group else "baseline"
                    if "calibration_set" in group and group.calibration_set.nunique() != 1:
                        raise ValueError("Mixed calibration sets in one scenario")
                    base_models = settings.get("models", [])
                    kind = "stress" if calibration_set != "baseline" else ("baseline" if model in base_models else "tail_sensitivity")
                    if src.user_view and model in src.user_view["models"]:
                        kind = "user_assumption"
                    groups.append((model, kind, calibration_set, _numbers(group, "terminal_value").to_numpy(), "same_historical_run", "Common historical simulation clock"))
            else:
                name = src.files.get("paths.npz")
                if not name:
                    raise ValueError("Physical terminal paths were not saved")
                with np.load(src.folder / name, allow_pickle=False) as archive:
                    values = archive["terminal_wealth"]
                    # The engine simulates positive-weight holdings in portfolio order.
                    # Older bundles may lack column labels; still check their shape.
                    holding_ids = [h["request_id"] for h in identity["portfolio"]["holdings"] if h["weight"] > 0]
                    if "holding_ids" in archive and archive["holding_ids"].tolist() != holding_ids:
                        raise ValueError("Physical terminal holding identifiers mismatch")
                    if values.shape != (settings["paths"], len(holding_ids) + 1):
                        raise ValueError("Physical terminal portfolio shape mismatch")
                    terminal = values[:, -1].copy()
                comparison, reason = _compatibility(historical, physical, cfg)
                groups.append(("physical_local_vol", "physical", "baseline", terminal, comparison, reason))
            # Assemble locally before publishing: an error cannot leave half
            # the scenarios from a malformed stage presented as valid results.
            staged_rows, staged_samples = [], {}
            for model, kind, calibration_set, values, comparison, reason in groups:
                row = dict(scenario=model, category=kind, calibration_set=calibration_set,
                           stage=stage, status=s["status"], currency=s["simulation_currency"],
                           forecast_start=settings["forecast_start"], forecast_end=settings["forecast_end"],
                           initial_value=initial, comparability=comparison, comparison_reason=reason)
                row.update(_metrics(values, initial, cfg))
                staged_rows.append(row); staged_samples[model] = values / initial - 1
            rows.extend(staged_rows); samples.update(staged_samples)
        except (OSError, ValueError, KeyError, AttributeError, TypeError, zipfile.BadZipFile, EOFError) as exc:
            next(r for r in availability if r["stage"] == stage).update(status="integrity_error", reason=str(exc))
            findings.append(f"{STAGES[stage]} excluded from terminal comparison: {exc}")
            sources.pop(stage,None)
            if stage=='monte_carlo':historical=None
            else:physical=None
    # An unavailable run still exports a readable, stable CSV schema. A bare
    # empty DataFrame would write a headerless file that read_csv cannot load.
    columns = ["scenario", "category", "calibration_set", "stage", "status", "currency",
               "forecast_start", "forecast_end", "initial_value", "comparability", "comparison_reason",
               "paths", "mean_return", "median_return", "mean_mc_standard_error", "mean_ci_low", "mean_ci_high",
               "probability_loss", "loss_probability_low", "loss_probability_high"]
    for alpha in cfg["confidence_levels"]:
        columns.extend(prefix + probability_label(alpha) for prefix in ("loss_var_", "loss_es_", "expected_tail_paths_"))
    comparison = pd.DataFrame(rows, columns=columns)
    if not rows:
        findings.append("No usable full-portfolio terminal simulation is available. Historical observations, if present, remain descriptive evidence only.")
    else:
        for row in rows:
            ref = _reference_for(row, comparison, cfg)
            if ref is not None:
                delta = row["probability_loss"] - ref.probability_loss
                adjective = "Above" if abs(delta) >= cfg["material_loss_probability_difference"] else "Below"
                findings.append(f"{row['scenario']}: loss probability differs from {ref.scenario} by {delta*100:+.2f} percentage points. {adjective} the configured descriptive materiality threshold; this is not a model-ranking or significance test.")
                alpha = max(cfg["confidence_levels"])
                es_key = "loss_es_" + probability_label(alpha)
                findings.append(f"At {confidence_label(alpha)} confidence, {row['scenario']} terminal ES is {row[es_key]:.2%} of initial wealth, versus {ref[es_key]:.2%} for {ref.scenario}. Loss frequency and tail severity answer different questions.")
        for category in ("stress", "tail_sensitivity"):
            group = comparison[comparison.category == category]
            if len(group):
                findings.append(f"{category.replace('_', ' ').capitalize()} scenarios: loss probabilities range from {group.probability_loss.min():.2%} to {group.probability_loss.max():.2%}. These are conditional scenario outcomes, not estimated probabilities of a crisis or proof of a preferred tail specification.")
        if not any(row['stage'] == 'local_vol_real_world' for row in rows):
            findings.append("No physical local-volatility comparison is available. The historical portfolio analysis remains usable on its own terms.")
        for row in rows:
            if row["status"] != "complete":
                findings.append(f"{row['scenario']} retains source status '{row['status']}'. It is included in the main analysis; the original validation qualification remains visible.")
            for alpha in cfg["confidence_levels"]:
                if row[f"expected_tail_paths_{probability_label(alpha)}"] < cfg["minimum_expected_tail_paths"]:
                    findings.append(f"{row['scenario']}: limited simulated tail support at {alpha:.0%}; inspect quantile and ES sampling sensitivity.")
    if historical and historical.user_view:
        view = historical.user_view
        if view["status"] == "complete":
            inputs = [f"{a['instrument']}: annual compound growth {a['effective_annual_compound_growth']:.2%}, annual volatility {a['effective_annual_volatility']:.2%}" for a in view["applied"]]
            findings.insert(0,"User assumptions for the full portfolio using the baseline calibration window: "+"; ".join(inputs)+".")
        else:
            findings.insert(0,"User scenario unavailable: "+view["reason"])
    portfolio = sources.get("processing").summary.get("portfolio", {}) if "processing" in sources else {}
    if not portfolio:
        for src in sources.values():
            portfolio = src.summary.get("identity", {}).get("portfolio", {})
            if portfolio:
                break
    # Current requested settings describe an empty run only. They must never
    # relabel saved results made with an earlier allocation or currency.
    if not portfolio:
        portfolio = requested_portfolio or {}
    if rows:
        currencies = sorted({r["currency"] for r in rows})
        reporting = portfolio.get("reporting_currency")
        if reporting and currencies != [reporting]:
            findings.insert(0, f"Requested reporting currency is {reporting}; available simulation currencies are {', '.join(currencies)}. These results do not measure {reporting} portfolio risk without the corresponding FX conversion.")
    if physical is not None:
        boundary = physical.table("boundaries")
        if {"instrument_id", "paths_visiting_wings"}.issubset(boundary.columns) and physical.summary.get("paths"):
            for item in boundary.to_dict("records"):
                fraction = float(item["paths_visiting_wings"]) / physical.summary["paths"]
                findings.append(f"{item['instrument_id']}: {fraction:.2%} of physical paths visit extrapolated wings. These paths depend on boundary assumptions beyond directly supported quotes.")
    findings.append("No automatic allocation recommendation: compare conditional downside, data coverage and model assumptions rather than choosing the model with the highest mean.")
    from importlib.metadata import version
    identity = dict(schema_version=1, cob=cob, settings=cfg, portfolio=portfolio,
                    source_manifests={k: dict(folder=str(v.folder), sha256=v.fingerprint) for k, v in sources.items()},
                    availability=availability, code_sha256={"simulation_comparison": _hash(Path(__file__)),
                                 "mc_diagnostics": _hash(Path(__file__).with_name("mc_diagnostics.py"))},
                    dependencies={n: version(n) for n in ("numpy", "pandas", "matplotlib")})
    run_id = hashlib.sha256(_bytes(identity)).hexdigest()[:16]
    return FinalAnalysis(cob, cfg, sources, availability, comparison, samples, findings, identity, portfolio, run_id)


def _png(fig):
    buffer = io.BytesIO(); FigureCanvasAgg(fig).print_png(buffer)
    return buffer.getvalue()


def _model_label(name):
    """Readable algorithm labels, never instrument-specific assumptions."""
    if "__" in name:
        period, model = name.rsplit("__", 1)
        return f"{period.replace('_', ' ')}: {_model_label(model)}"
    return {"gaussian": "Historical Gaussian", "student_t_truncated": "Historical Student-t",
            "physical_local_vol": "Physical local volatility"}.get(name, name.replace("_", " "))


PRIMARY = ("baseline","user_assumption","physical")


def _reference_for(row, comparison, cfg):
    """Match an assumed scenario with its baseline shock family for numerical deltas."""
    if row["category"] not in PRIMARY or row["comparability"] in ("not_comparable","standalone"):
        return None
    model = (row["scenario"].rsplit("__",1)[-1] if row["category"] == "user_assumption"
             else cfg["reference_model"])
    refs = comparison[(comparison.category == "baseline") & (comparison.scenario == model)]
    return refs.iloc[0] if len(refs) == 1 and row["scenario"] != model else None


def _saved_view(summary):
    """Use the saved run settings. Absent or null means there is no view to report."""
    settings = summary.get("identity",{}).get("settings",{})
    request = settings.get("user_scenario")
    if request is None:
        return None
    record = summary.get("user_scenario")
    def unavailable(reason):
        return dict(requested=request,status="unavailable",models=[],reason=str(reason))
    if not isinstance(record,dict) or record.get("requested") != request:
        return unavailable("Saved user-scenario metadata is missing or differs from the request.")
    if record.get("status") != "complete":
        return unavailable(record.get("reason") or "No completed user scenario.")
    if not isinstance(request,dict) or not isinstance(request.get("holdings"),dict) or not request["holdings"] or type(request.get("periods_per_year")) is not int or request["periods_per_year"] < 1:
        return unavailable("Invalid saved user-scenario holdings.")
    primary = settings.get("models")
    if not isinstance(primary,list) or not all(isinstance(m,str) for m in primary):
        return unavailable("Invalid primary models in saved historical settings.")
    models = [f"{request.get('name')}__{model}" for model in primary]
    fields = ("historical_annual_compound_growth","effective_annual_compound_growth",
              "historical_annual_volatility","effective_annual_volatility")
    applied = record.get("applied")
    # The engine and configuration may list the same two shock families in different orders.
    saved_models = record.get("models")
    if (not models or not isinstance(saved_models, list) or not all(isinstance(m, str) for m in saved_models)
            or sorted(saved_models) != sorted(models) or not isinstance(applied,list)
            or not all(isinstance(a,dict) and isinstance(a.get("instrument"),str)
                       and all(type(a.get(k)) in (int,float) and math.isfinite(a[k]) for k in fields) for a in applied)
            or len(applied) != len(request["holdings"])
            or {a["instrument"] for a in applied} != set(request["holdings"])):
        return unavailable("Incomplete saved user-scenario parameters or model identifiers.")
    return record.copy()


def _verified_view_samples(src, table, settings):
    """Keep baseline and crisis results when an optional view is incomplete."""
    view = src.user_view
    record = src.summary.get("user_scenario")
    listed = record.get("models") if isinstance(record,dict) else None
    suspects = {m for m in (listed if isinstance(listed,list) else []) if isinstance(m,str) and "__" in m}
    name = view["requested"].get("name") if view and isinstance(view["requested"],dict) else None
    declared = settings.get("stress_calibrations")
    stresses = {s["name"] for s in (declared if isinstance(declared,list) else [])
                if isinstance(s,dict) and isinstance(s.get("name"),str)}
    if isinstance(name,str) and name not in stresses and name != "baseline" and isinstance(settings.get("models"),list):
        suspects.update(name+"__"+m for m in settings.get("models",[]))
    if view and view["status"] == "complete":
        for model in view["models"]:
            group = table[table.model == model]
            values = _numbers(group,"terminal_value")
            if (len(group) != settings["paths"] or group.path.duplicated().any()
                    or "calibration_set" not in group or not group.calibration_set.eq(name).all()
                    or not np.isfinite(values).all() or (values < 0).any()):
                view.update(status="unavailable",models=[],reason="Invalid saved portfolio samples for "+model)
                break
    if view and view["status"] == "complete" and "calibration_set" in table:
        # Only declared primary models belong to the view. An unlisted row in
        # that same calibration must not leak into the historical stress bucket.
        table = table[(table.calibration_set != name) | table.model.isin(view["models"])]
    # Actual named stress rows always retain their original identity.
    if "stress_calibrations" in settings and "calibration_set" in table:
        allowed = {"baseline", *stresses}
        if view and view["status"] == "complete":
            allowed.add(name)
        table = table[table.calibration_set.isin(allowed)]
    if view is None or view["status"] != "complete":
        drop = table.model.isin(suspects)
        if "calibration_set" in table:
            drop &= ~table.calibration_set.isin(stresses | {"baseline"})
        return table[~drop]
    return table


def _stress_groups(result):
    """Compare each named stress with the same-run baseline, not with tail sweeps.

    Primary model names come from the SAVED simulation configuration. A stress
    prefix identifies a calibration window, not a new stochastic engine. Keep
    optional physical scenarios only when their existing compatibility check
    allows comparison. Nothing here reruns calibration or changes eligibility.
    """
    source = result.sources.get("monte_carlo")
    if source is None or result.comparison.empty:
        return {}
    models = source.summary.get("identity", {}).get("settings", {}).get("models", [])
    rows = result.comparison.to_dict("records")
    reference = [r for r in rows if r["category"] in PRIMARY
                 and r["comparability"] not in ("not_comparable", "standalone")]
    groups = {}
    for row in rows:
        name = row["calibration_set"]
        if (row["category"] == "stress" and row["scenario"] in
                {f"{name}__{model}" for model in models}):
            groups.setdefault(name, list(reference)).append(row)
    return groups


def _stress_context(result):
    """Expose requested and actual stress coverage, including failed attempts.

    The baseline lookback does not constrain a separately requested crisis
    window. Its availability depends on the archive and the engine's matched-
    interval minimum. Report actual dates so partial windows are not presented
    as complete historical coverage. Never substitute baseline returns for a
    missing stress calibration.
    """
    source = result.sources.get("monte_carlo")
    if source is None:
        return pd.DataFrame()
    saved = source.summary
    requested = saved.get("identity", {}).get("settings", {}).get("stress_calibrations", [])
    attempts = saved.get("stress_calibrations", [])
    requested = requested if isinstance(requested, list) else []
    attempts = attempts if isinstance(attempts, list) else []
    records = {r["name"]: r for r in requested if isinstance(r, dict) and isinstance(r.get("name"), str)}
    for attempt in attempts:
        if isinstance(attempt, dict) and isinstance(attempt.get("name"), str):
            records[attempt["name"]] = {**records.get(attempt["name"], {}), **attempt}
    groups = _stress_groups(result)
    for name in groups:
        records.setdefault(name, {"name": name})
    rows = []
    for name, record in records.items():
        calibration = record.get("calibration") or {}
        calibration = calibration if isinstance(calibration, dict) else {}
        warnings = record.get("warnings") or []
        reason = "; ".join(str(w) for w in warnings) if isinstance(warnings, list) else str(warnings)
        available = name in groups
        if not available:
            reason = reason or "No saved primary stress outcomes; no stress forecast is inferred."
        rows.append(dict(name=name, requested_start=record.get("start"), requested_end=record.get("end"),
                         drift=record.get("drift"), calibration_status=record.get("status", "not recorded"),
                         first_interval_start=calibration.get("first_start"), last_interval_end=calibration.get("last_end"),
                         matched_intervals=calibration.get("observations"),
                         comparison_status="available" if available else "unavailable", reason=reason))
    return pd.DataFrame(rows)


def _forecast_data(result, rows):
    """Read recorded portfolio trajectories; do not align or resimulate paths.

    Historical fans already summarize every simulation, whereas sample_paths
    stores only a display subset. Physical fans are computed across every
    column of saved portfolio_wealth. We NEVER estimate fans from the small
    display subset, nor interpret equal path IDs as paired random shocks.
    """
    output = {}
    policy = result.cfg["simulation_plots"]
    lower, upper = policy["fan_quantiles"]
    for row in rows:
        src = result.sources[row["stage"]]; name = row["scenario"]
        try:
            initial = row["initial_value"]
            if row["stage"] == "monte_carlo":
                table = src.table("bands")
                if table.empty:
                    result.chart_notes.append(f"{name}: no saved forecast bands; terminal outcomes are still included.")
                    continue
                table = table[(table.scope == "portfolio") & (table.model == name)].sort_values("date")
                if table.empty or table.date.duplicated().any():
                    raise ValueError("Missing or duplicated portfolio band dates")
                def column(q):
                    key = "quantile_" + probability_label(q)
                    # Older saved files also expose these same three quantiles
                    # under stable aliases. No other quantile is approximated.
                    if key not in table:
                        key = {.05:"p05", .5:"p50", .95:"p95"}.get(q, key)
                    if key not in table:
                        raise ValueError(f"Requested fan quantile {q} was not saved")
                    return _numbers(table, key).to_numpy(dtype=float)
                times = pd.DatetimeIndex(pd.to_datetime(table.date, utc=True)).tz_convert(None)
                values = np.column_stack((_numbers(table,"mean"), column(.5), column(lower), column(upper)))
                sample = src.table("sample_paths")
                trajectories = []
                if len(sample):
                    sample = sample[(sample.scope == "portfolio") & (sample.model == name)]
                    # Preserve saved IDs and choose a deterministic subset,
                    # not the prettiest/worst paths or a random second sample.
                    for _, path in list(sample.groupby("path", sort=False))[:policy["sample_paths"]]:
                        path = path.sort_values("date")
                        dates = pd.DatetimeIndex(pd.to_datetime(path.date, utc=True)).tz_convert(None)
                        if not dates.equals(times):
                            raise ValueError("Saved sample path dates differ from band dates")
                        trajectories.append(_numbers(path,"value").to_numpy(dtype=float))
                samples = np.array(trajectories).T if trajectories else np.empty((len(times),0))
            else:
                with np.load(src.folder / src.files["paths.npz"], allow_pickle=False) as archive:
                    if not {"years", "portfolio_wealth"}.issubset(archive.files):
                        result.chart_notes.append(f"{name}: terminal wealth was saved, but trajectory arrays are unavailable.")
                        continue
                    years = archive["years"]; paths = archive["portfolio_wealth"]
                if years.ndim != 1 or paths.shape != (len(years), row["paths"]) or len(years)<2 or years[0]!=0 or not np.isfinite(years).all() or not np.all(np.diff(years)>0):
                    raise ValueError("Invalid physical path time/shape metadata")
                if not np.isfinite(paths).all() or (paths<0).any() or not np.allclose(paths[0], initial):
                    raise ValueError("Invalid physical portfolio wealth paths")
                if not np.allclose(paths[-1]/initial-1, result.samples[name], rtol=1e-12, atol=1e-14):
                    raise ValueError("Physical paths disagree with saved terminal outcomes")
                # ACT/365F is the evaluator's existing clock, not a replacement
                # for the historical exchange-session convention. Plot each
                # engine at its own actual recorded dates with no regridding.
                start = pd.Timestamp(src.summary["start_timestamp"])
                end = pd.Timestamp(src.summary["end_timestamp"])
                if start.tzinfo is None or end.tzinfo is None:
                    raise ValueError("Physical timestamps must include timezone")
                times = pd.DatetimeIndex(start + pd.to_timedelta(years*365, unit="D")).tz_convert("UTC").tz_localize(None)
                if abs((times[-1].tz_localize("UTC")-end).total_seconds())>1:
                    raise ValueError("Physical path horizon differs from its end timestamp")
                quantiles = np.quantile(paths, [.5,lower,upper], axis=1)
                values = np.column_stack((paths.mean(axis=1), *quantiles))
                samples = paths[:,:min(policy["sample_paths"], paths.shape[1])]
            if not np.isfinite(values).all() or not np.isfinite(samples).all() or (values<0).any() or (samples<0).any():
                raise ValueError("Invalid forecast values")
            if np.any(values[:,2]>values[:,1]) or np.any(values[:,1]>values[:,3]):
                raise ValueError("Unordered forecast quantiles")
            if not np.isclose(values[-1,0]/initial-1, row["mean_return"],rtol=1e-10,atol=1e-12):
                raise ValueError("Forecast mean disagrees with terminal summary")
            bands = pd.DataFrame(values/initial*100, columns=["mean_index","median_index","lower_index","upper_index"])
            bands.insert(0,"timestamp", times);bands.insert(0,"scenario",name)
            bands["lower_quantile"]=lower;bands["upper_quantile"]=upper
            output[name] = (bands, samples/initial*100)
        except (OSError,ValueError,KeyError,AttributeError,zipfile.BadZipFile,EOFError) as exc:
            # Missing or malformed plotting inputs do not erase independently
            # verified terminal metrics. Explain why that particular chart is absent.
            result.chart_notes.append(f"{name}: trajectory chart unavailable ({exc}).")
    return output


def _mc_detail_figures(result, rows, index, create, figures, forecasts, prefix="mc"):
    """Growth, full empirical density and tails for each comparable model group."""
    policy=result.cfg["simulation_plots"]
    names=[row["scenario"] for row in rows]
    combined=np.concatenate([result.samples[n] for n in names])
    # Shared bins and N-based weights retain total probability mass. We do not
    # fit a normal/Student-t density to terminal wealth: those names refer to
    # the daily shock engines, whose terminal portfolio distributions differ.
    low,high=float(combined.min()),float(combined.max())
    if low==high:
        # A deterministic distribution is a point mass, not an infinite PDF.
        def density(ax):
            for i,name in enumerate(names):
                ax.axvline(result.samples[name][0]*100,color=f"C{i}",label=_model_label(name))
            ax.set_xlabel("Terminal return (%) - deterministic point masses");ax.legend()
    else:
        edges=np.linspace(low,high,policy["histogram_bins"]+1)
        def density(ax):
            for i,name in enumerate(names):
                values=result.samples[name]*100
                ax.hist(values,bins=edges*100,density=True,histtype="step",linewidth=1.8,
                        color=f"C{i}",label=_model_label(name))
                ax.axvline(np.median(values),color=f"C{i}",linestyle=":",alpha=.7)
            ax.axvline(0,color="black",linestyle="--",linewidth=.7)
            ax.set_xlabel("Terminal portfolio return (%)");ax.set_ylabel("Probability density per percentage point")
            ax.legend(fontsize=9)
    create(f"{prefix}_density_{index}","Terminal return distributions (dotted lines: medians)",density)
    if prefix != "user_mc":
        fig=Figure(figsize=(12,4.8),layout="constrained")
        axes=fig.subplots(1,2)
        left,right=np.quantile(combined,[policy["tail_display_fraction"],1-policy["tail_display_fraction"]])
        for i,name in enumerate(names):
            values,counts=np.unique(result.samples[name],return_counts=True);n=counts.sum()
            cumulative=np.cumsum(counts)
            # Aggregate ties: at a deterministic return the CDF is one, not a
            # misleading sequence of tiny probabilities at the same x-coordinate.
            for ax,mask,prob in ((axes[0],values<=left,cumulative/n),(axes[1],values>=right,(n-cumulative+counts)/n)):
                x=values[mask];y=prob[mask]
                keep=np.unique(np.linspace(0,len(x)-1,min(len(x),result.cfg["maximum_chart_points"])).astype(int)) if len(x) else []
                ax.semilogy(x[keep]*100,y[keep],color=f"C{i}",label=_model_label(name))
        for ax,title,label in zip(axes,["Downside tail","Upside tail"],["P(return <= x)","P(return >= x)"]):
            ax.set_title(title);ax.set_xlabel("Terminal return x (%)");ax.set_ylabel(label)
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        figures[f"{prefix}_tails_{index}"]=("Terminal tails on a logarithmic probability scale; all outcomes retained",_png(fig))
    # Numerical preparation is shared with the text-only report. Select this
    # comparable group without recalculating data or duplicating warnings.
    forecasts = {name: forecasts[name] for name in names if name in forecasts}
    if not forecasts:
        return
    if prefix != "user_mc":
        fig=Figure(figsize=(12,4.8),layout="constrained");axes=fig.subplots(1,2)
        for i,name in enumerate(names):
            if name not in forecasts:continue
            bands,_=forecasts[name]
            for ax,col in zip(axes,["mean_index","median_index"]):
                ax.plot(bands.timestamp,bands[col],color=f"C{i}",label=_model_label(name))
        for ax,title in zip(axes,["Expected portfolio wealth (mean)","Median portfolio wealth"]):
            ax.set_title(title);ax.set_ylabel("Wealth index (initial = 100)");ax.grid(alpha=.2);ax.legend(fontsize=8)
            ax.tick_params(axis="x",rotation=25)
        figures[f"{prefix}_growth_{index}"]=("Portfolio growth comparison at each engine's recorded dates",_png(fig))
    fig=Figure(figsize=(12,3.7*len(forecasts)),layout="constrained");axes=np.atleast_1d(fig.subplots(len(forecasts),1,sharex=True,sharey=True))
    for ax,(name,(bands,sample)) in zip(axes,forecasts.items()):
        color=f"C{names.index(name)}"
        ax.fill_between(bands.timestamp,bands.lower_index,bands.upper_index,color=color,alpha=.16,
                        label=f"{policy['fan_quantiles'][0]:.0%}-{policy['fan_quantiles'][1]:.0%} population band")
        ax.plot(bands.timestamp,sample,color=color,alpha=.2,linewidth=.65)
        ax.plot(bands.timestamp,bands.mean_index,color=color,linewidth=2,label="Mean")
        ax.plot(bands.timestamp,bands.median_index,color="black",linestyle="--",linewidth=1.1,label="Median")
        ax.axhline(100,color="grey",linestyle=":",linewidth=.8)
        ax.set_title(f"{_model_label(name)}: {sample.shape[1]} saved sample paths")
        ax.set_ylabel("Wealth index");ax.grid(alpha=.15);ax.legend(fontsize=8,loc="upper left")
    axes[-1].tick_params(axis="x",rotation=25)
    figures[f"{prefix}_paths_{index}"]=("Portfolio Monte Carlo paths and population bands; path IDs do not establish shock pairing across different engines",_png(fig))


def _smile_figures(result, figures):
    """Display stored observed/fitted IVs, with no new fit or IV inversion."""
    src=result.sources.get("volatility_surface")
    if src is None:return
    quotes=src.table("repricing")
    required={"group_id","expiration","k","observed_iv","fitted_iv"}
    if quotes.empty or not required.issubset(quotes):return
    count=result.cfg["simulation_plots"]["smiles_per_figure"]
    for cohort,(gid,group) in enumerate(quotes.groupby("group_id",sort=False)):
        expiries=sorted(group.expiration.unique())
        for page,offset in enumerate(range(0,len(expiries),count)):
            batch=expiries[offset:offset+count];cols=min(2,len(batch));nrows=math.ceil(len(batch)/cols)
            fig=Figure(figsize=(12,3.3*nrows),layout="constrained");axes=np.atleast_1d(fig.subplots(nrows,cols)).ravel()
            for ax,expiry in zip(axes,batch):
                q=group[group.expiration==expiry].sort_values("k",key=lambda x:pd.to_numeric(x,errors="coerce"))
                k=_numbers(q,"k");observed=_numbers(q,"observed_iv")*100;fitted=_numbers(q,"fitted_iv")*100
                ax.scatter(k,observed,s=8,alpha=.55,label="IV from observed quotes")
                ax.plot(k,fitted,color="C1",linewidth=1.6,label="Fitted smile")
                if {"bid_iv","ask_iv"}.issubset(q):
                    bid=_numbers(q,"bid_iv")*100;ask=_numbers(q,"ask_iv")*100
                    ax.fill_between(k,bid,ask,where=np.isfinite(bid)&np.isfinite(ask),alpha=.12,label="Bid/ask IV where available")
                ax.axvline(0,color="grey",linestyle=":");ax.set_title(str(expiry));ax.set_xlabel("log(K/F)")
                ax.set_ylabel("Annualized IV (%)");ax.grid(alpha=.15);ax.legend(fontsize=7)
            for ax in axes[len(batch):]:ax.set_visible(False)
            figures[f"smiles_{cohort}_{page}"]=(f"Observed and fitted smiles, cohort {gid}: expiries {offset+1}-{offset+len(batch)} of {len(expiries)}",_png(fig))


def _prepare_report_data(result):
    """Prepare tables and availability notes independently of image rendering.

    Reset derived outputs on each preparation so report/save call order cannot
    accumulate rows or warnings. Empty runs retain stable CSV schemas. Missing
    trajectories leave verified terminal outcomes usable and explicitly flagged.
    No simulations or provider calls occur here.
    """
    result.chart_notes=[]
    result.forecast_bands=pd.DataFrame(columns=["scenario","timestamp","mean_index","median_index","lower_index","upper_index","lower_quantile","upper_quantile"])
    thresholds=[]
    for row in result.comparison.to_dict("records"):
        values=result.samples[row["scenario"]]
        for loss in result.cfg["simulation_plots"]["loss_thresholds"]:
            count=int(np.count_nonzero(values < -loss))
            thresholds.append(dict(scenario=row["scenario"],loss_threshold=loss,loss_events=count,paths=len(values),probability=count/len(values)))
    result.loss_thresholds=pd.DataFrame(thresholds,columns=["scenario","loss_threshold","loss_events","paths","probability"])
    # Include primary stress trajectories once; sensitivity sweeps remain in
    # their separate terminal tables. This also works for text-only reports.
    selected = {r["scenario"]: r for r in result.comparison.to_dict("records")
                if r["category"] in PRIMARY}
    for group in _stress_groups(result).values():
        selected.update({r["scenario"]: r for r in group})
    rows = list(selected.values())
    forecasts = _forecast_data(result, rows)
    if forecasts:
        result.forecast_bands = pd.concat([item[0] for item in forecasts.values()], ignore_index=True)
    return forecasts


def _terminal_figures(result, rows, bucket, index, create, prefix=""):
    """Compare the same scenario set in both the ECDF and loss-probability chart."""
    def ecdf(ax):
        for row in rows:
            values = np.sort(result.samples[row["scenario"]])
            indices = np.unique(np.linspace(0, len(values)-1, min(len(values), result.cfg["maximum_chart_points"])).astype(int))
            ax.plot(values[indices]*100, (indices+1)/len(values), label=row["scenario"])
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set_xlabel("Terminal portfolio return (%)"); ax.set_ylabel("Cumulative simulated probability")
        ax.legend(fontsize=8)
    create(f"{prefix}terminal_distribution_{index}", f"Terminal distributions: {bucket}", ecdf)
    def losses(ax):
        y = np.arange(len(rows)); p = np.array([r["probability_loss"] for r in rows])
        lo = np.array([r["loss_probability_low"] for r in rows]); hi = np.array([r["loss_probability_high"] for r in rows])
        ax.errorbar(p*100, y, xerr=np.maximum(0,np.array([p-lo, hi-p])*100), fmt="o", capsize=4)
        ax.set_yticks(y, [r["scenario"] for r in rows]); ax.set_xlim(0,100)
        ax.set_xlabel(f"Loss probability (%) with {result.cfg['sampling_confidence']:.0%} sampling interval")
    create(f"{prefix}loss_probability_{index}", f"Probability of finishing below initial wealth: {bucket}", losses)


def _figures(result):
    """Render from recorded data only. Downsampling changes display, not metrics."""
    figures = {}
    forecasts = _prepare_report_data(result)
    def create(key, title, draw):
        fig = Figure(figsize=(10, 5.3), layout="constrained")
        ax = fig.add_subplot(111)
        draw(ax); ax.set_title(title); ax.grid(alpha=.18)
        figures[key] = (title, _png(fig))
    def thin(frame):
        return frame.iloc[np.unique(np.linspace(0, len(frame)-1, min(len(frame), result.cfg["maximum_chart_points"])).astype(int))] if len(frame) else frame
    stats = result.sources.get("statistics")
    if stats:
        dependence = _chosen(stats.table("dependence"), result.cfg)
        if not dependence.empty:
            matrix = dependence.assign(value=_numbers(dependence, "correlation")).pivot(index="instrument_i", columns="instrument_j", values="value")
            def heat(ax):
                im = ax.imshow(matrix.astype(float), vmin=-1, vmax=1, cmap="RdBu_r")
                ax.set_xticks(range(len(matrix.columns)), matrix.columns)
                ax.set_yticks(range(len(matrix)), matrix.index)
                for i in range(len(matrix)):
                    for j in range(len(matrix.columns)):
                        ax.text(j, i, f"{matrix.iloc[i,j]:.2f}", ha="center", va="center")
                ax.figure.colorbar(im, ax=ax, label="Correlation")
            create("historical_correlation", "Historical correlation on common return intervals", heat)
        rolling = _chosen(stats.table("rolling_volatility"), result.cfg)
        volcols = [c for c in rolling if c.startswith("volatility_")]
        if len(rolling) and volcols:
            column = volcols[-1]
            def lines(ax):
                for (scope, instrument), group in rolling.groupby(["scope", "instrument_id"], sort=False):
                    group = thin(group.sort_values("date"))
                    ax.plot(pd.to_datetime(group.date), _numbers(group, column)*100, label=f"{scope}: {instrument}")
                ax.set_ylabel("Annualized volatility (%)"); ax.legend(fontsize=8)
            create("rolling_volatility", f"Historical rolling volatility ({column.split('_')[-1]} observations)", lines)
        daily = stats.table("daily_statistics")
        paths = stats.table("path_statistics")
        if len(daily) and len(paths):
            # Never bridge a known gap in the selected-price history. Restrict
            # path charts to series the statistics engine itself accepted.
            good = set(zip(paths[paths.path_status == "available"].scope, paths[paths.path_status == "available"].instrument_id))
            if good:
                def history(ax):
                    for key, group in daily.groupby(["scope", "instrument_id"], sort=False):
                        if key in good:
                            group = thin(group.sort_values("date"))
                            ax.plot(pd.to_datetime(group.date), _numbers(group, "drawdown")*100, label=f"{key[0]}: {key[1]}")
                    ax.set_ylabel("Drawdown from preceding peak (%)"); ax.legend(fontsize=8)
                create("historical_drawdown", "Observed historical drawdowns (individual coverage periods)", history)
    comp = result.comparison
    if len(comp):
        # Separate stress, sensitivities and incompatible scenarios so a shared
        # chart cannot accidentally imply an apples-to-apples comparison.
        buckets = {}
        for row in comp.to_dict("records"):
            category = row["category"]
            if row["comparability"] in ("not_comparable", "standalone"):
                bucket = row["scenario"] + " (standalone)"
            else:
                bucket = "portfolio models" if category in PRIMARY else category
            buckets.setdefault(bucket, []).append(row)
        for index, (bucket, rows) in enumerate(buckets.items()):
            if any(row["category"] in PRIMARY for row in rows):
                _mc_detail_figures(result, rows, index, create, figures, forecasts)
            _terminal_figures(result, rows, bucket, index, create)
    # Separate panels make conditional crisis assumptions explicit, while
    # preserving the original baseline versus physical-model comparisons.
    for index, (name, rows) in enumerate(_stress_groups(result).items()):
        _mc_detail_figures(result, rows, index, create, figures, forecasts, prefix="stress_mc")
        _terminal_figures(result, rows, str(name)+" stress comparison", index, create, prefix="stress_mc_")
    user_rows = comp[comp.category == "user_assumption"]
    if len(user_rows):
        models = set(user_rows.scenario.str.split("__").str[-1])
        baseline = comp[(comp.category == "baseline") & comp.scenario.isin(models)]
        rows = pd.concat([baseline,user_rows]).to_dict("records")
        _mc_detail_figures(result, rows, 0, create, figures, forecasts, prefix="user_mc")
    for stage, table_name, zcol in (("volatility_surface", "surface_grid", "iv"),
                                    ("local_volatility", "local_vol_grid", "local_volatility")):
        src = result.sources.get(stage)
        if not src:
            continue
        grid = src.table(table_name)
        if not {"group_id", "years", "k", zcol}.issubset(grid.columns):
            continue
        for number, (gid, group) in enumerate(grid.groupby("group_id", sort=False)):
            if "status" in group:
                group = group[group.status == "available"]
            xyz = group[["years", "k", zcol]].apply(pd.to_numeric, errors="coerce").dropna()
            if xyz.empty:
                continue
            xyz = xyz.iloc[np.unique(np.linspace(0,len(xyz)-1,min(len(xyz),result.cfg["surface_display_points"])).astype(int))]
            fig = Figure(figsize=(10,6), layout="constrained"); ax = fig.add_subplot(111, projection="3d")
            im = ax.scatter(xyz.k, xyz.years, xyz[zcol]*100, c=xyz[zcol]*100, cmap="viridis", s=7)
            ax.set_xlabel("log(K/F)" if stage == "volatility_surface" else "log(S/F)")
            ax.set_ylabel("Time (years)"); ax.set_zlabel("Annualized volatility (%)")
            metadata = src.table("cohort_status" if stage == "volatility_surface" else "cohorts")
            matched = metadata[metadata.group_id == gid] if "group_id" in metadata else pd.DataFrame()
            row = group.iloc[0] if "underlying" in group else (matched.iloc[0] if len(matched) else {})
            description = f"{row.get('underlying', gid)}, {row.get('valuation_date', '')}"
            kind = "Implied volatility" if stage == "volatility_surface" else "Local volatility"
            title = f"{kind}: {description} (supported grid only)"
            ax.set_title(title, fontsize=11); fig.colorbar(im, ax=ax, shrink=.55, pad=.1)
            figures[f"{stage}_{number}"] = (title, _png(fig))
    _smile_figures(result,figures)
    return figures


def _table(frame, columns=None, percent=()):
    if frame.empty:
        return '<p class="muted">No usable observations supplied for this section.</p>'
    frame = frame.copy()
    if columns is not None:
        frame = frame[[c for c in columns if c in frame]]
    for c in frame:
        if c == "scenario":
            frame[c] = frame[c].map(_model_label)
        if c in percent:
            frame[c] = pd.to_numeric(frame[c], errors="coerce").map(lambda x: f"{x:.2%}" if pd.notna(x) else "Unavailable")
    # CSVs retain stable machine-readable names; HTML uses readable labels.
    frame.columns = [str(c).replace("_", " ").capitalize() for c in frame.columns]
    return '<div class="table-wrap">' + frame.to_html(index=False, escape=True, border=0, na_rep="Unavailable", float_format=lambda x: f"{x:.5g}") + '</div>'


def _outcome_interpretation(result):
    """Numerical comparisons, not a claim that the best-looking model is true."""
    data=result.comparison
    if data.empty:return []
    alpha=max(result.cfg["confidence_levels"]);key="loss_es_"+probability_label(alpha)
    statements=[]
    for row in data.to_dict("records"):
        ref = _reference_for(row,data,result.cfg)
        if ref is None:
            continue
        statements.append(f"{_model_label(row['scenario'])}: expected growth {row['mean_return']:.2%}, median growth {row['median_return']:.2%}, and loss probability {row['probability_loss']:.2%}. The corresponding {_model_label(ref.scenario)} estimates are {ref.mean_return:.2%}, {ref.median_return:.2%}, and {ref.probability_loss:.2%}.")
        if row["probability_loss"] < ref.probability_loss and row[key] > ref[key]:
            statements.append(f"{_model_label(row['scenario'])} has fewer losses overall in this sample, but a more severe extreme downside: {confidence_label(alpha)} expected shortfall is {row[key]:.2%} versus {ref[key]:.2%}. A narrower central distribution or higher median therefore does not establish lower tail risk.")
    return statements


def _modeling_context(result):
    """Describe the settings and actual coverage of the SAVED simulations.

    Never read today's model JSON here: it may have changed since the run.
    The identity stores the executable JSON settings used by each engine.
    Requested dates describe intent; calibration summaries/forecast grids
    describe what was actually used. Missing metadata stays explicitly unknown.
    """
    records=[]
    for stage in ('monte_carlo','local_vol_real_world'):
        src=result.sources.get(stage)
        if src is None:
            continue
        s=src.summary;identity=s.get('identity',{})
        cfg=identity.get('settings' if stage=='monte_carlo' else 'mc_config',{})
        cal=s.get('calibration') or {};policy=cfg.get('calibration') or {}
        dates=s.get('forecast_dates') or []
        first=cal.get('first_interval') or [];last=cal.get('last_interval') or []
        actual_start=cal.get('first_start') if stage=='monte_carlo' else (first[0] if first else None)
        actual_end=cal.get('last_end') if stage=='monte_carlo' else (last[-1] if last else None)
        # A verified artifact can still record an unavailable simulation.
        # Do not turn its requested settings into a claim of completed work.
        used=bool(s.get('simulation_performed'))
        record=dict(stage=stage,label=STAGES[stage],status=s.get('status','unknown'),
            terminal_results_included=bool(len(result.comparison) and (result.comparison.stage==stage).any()),
            simulation_performed=used,requested_forecast_start=cfg.get('forecast_start'),
            requested_forecast_end=cfg.get('forecast_end'),
            actual_forecast_start=(dates[0] if dates else None) if stage=='monte_carlo' else s.get('start_timestamp'),
            actual_forecast_end=(dates[-1] if dates else None) if stage=='monte_carlo' else s.get('end_timestamp'),
            paths=cfg.get('paths'),calendar=cfg.get('calendar'),portfolio_method=cfg.get('portfolio_method'),
            seed=cfg.get('seed'),initial_value=cfg.get('initial_value'),currency=s.get('simulation_currency'),
            requested_history_start=cal.get('requested_start'),
            requested_history_end=cal.get('requested_end',result.cob),
            history_policy=(f"Explicit start {policy['start']} (overrides lookback)" if policy.get('start') else
                            f"Trailing {policy['lookback_years']} calendar years" if policy.get('lookback_years') is not None else 'Not recorded'),
            actual_history_start=actual_start,actual_history_end=actual_end,
            matched_daily_intervals=cal.get('observations'),
            minimum_observations=policy.get('minimum_observations'),
            clock=('Exchange-session daily log increments' if stage=='monte_carlo' else s.get('day_count','Not recorded')),
            steps=s.get('steps') if stage=='monte_carlo' else s.get('fine_steps'),
            student_t_settings=cfg.get('student_t') if stage=='monte_carlo' else None,
            physical_settings=identity.get('config') if stage=='local_vol_real_world' else None,
            stress_calibrations=s.get('stress_calibrations',[]) if stage=='monte_carlo' else [])
        if src.user_view:
            record["user_scenario"] = src.user_view
        records.append(record)
    return records


def _equation(tex, description):
    """Embed typeset vector equations without JavaScript, CDN or network use.

    Matplotlib's existing math renderer supplies fraction bars, Greek letters,
    roots and subscripts as scalable SVG. The plain-language alternative text
    also makes the formula meaningful to screen readers. These are explanatory
    equations only; the simulator and its random numbers are untouched.
    """
    buffer=io.BytesIO()
    math_to_image('$'+tex+'$',buffer,prop=FontProperties(size=18),format='svg')
    # Rendering time is not financial provenance. Remove SVG's wall-clock
    # metadata so repeated HTML/Markdown views contain identical equations.
    svg = re.sub(rb'<dc:date>.*?</dc:date>', b'', buffer.getvalue())
    encoded=base64.b64encode(svg).decode('ascii')
    return '<div class="equation"><img alt="'+html.escape(description,quote=True)+'" src="data:image/svg+xml;base64,'+encoded+'"/></div>'


def _modeling_section(result):
    """Explain equations without pretending Student-t shocks are Brownian."""
    esc=lambda value:html.escape(str(value),quote=True)
    records=_modeling_context(result)
    parts=['<section id="modeling"><h2>Simulation method, forecast period and calibration data</h2>',
           '<p>These settings come from the JSON snapshots saved with each result, not from possibly edited current configuration files. Requested windows and actual usable data are shown separately. Descriptive historical statistics may use a longer history than the Monte Carlo calibration window.</p>']
    if not records:
        parts.append('<p>No saved Monte Carlo settings or completed simulation were supplied. No forecast period or calibration history is assumed.</p>')
    for r in records:
        parts.append(f'<h3>{esc(r["label"])}</h3>')
        if not r['terminal_results_included']:
            parts.append('<p>No usable terminal simulation results from this stage are included. The settings below describe the saved request, not evidence of a successful forecast.</p>')
        fields=[('Source status',r['status']),('Requested forecast',f"{r['requested_forecast_start']} to {r['requested_forecast_end']}"),
                ('Actual simulated period',f"{r['actual_forecast_start']} to {r['actual_forecast_end']}" if r['actual_forecast_start'] and r['actual_forecast_end'] else 'Not recorded'),
                ('Paths requested per scenario',r['paths']),('Time convention',r['clock']),('Simulated steps',r['steps']),
                ('Exchange calendar',r['calendar']),('Portfolio convention',r['portfolio_method']),
                ('Initial wealth / currency',f"{r['initial_value']} / {r['currency']}"),('Random seed',r['seed']),
                ('Historical window policy',r['history_policy']),
                ('Requested calibration interval',f"{r['requested_history_start']} to {r['requested_history_end']}"),
                ('Actual common historical interval',f"{r['actual_history_start']} to {r['actual_history_end']}" if r['actual_history_start'] and r['actual_history_end'] else 'Not recorded'),
                ('Matched daily return intervals',r['matched_daily_intervals']),('Minimum required intervals',r['minimum_observations'])]
        parts.append(_table(pd.DataFrame([{'Setting':k,'Saved value':'Not recorded' if v is None else v} for k,v in fields])))
        if r['stage']=='monte_carlo':
            parts.append('<p><strong>Gaussian MC and its SDE interpretation.</strong> Normalized holding/factor wealth follows a diffusion with correlated Brownian shocks:</p>')
            parts.append(_equation(r'\frac{\mathrm{d}G_i(t)}{G_i(t)}=\mu_i\,\mathrm{d}t+\sigma_i\,\mathrm{d}W_i(t),\qquad \mathrm{d}W_i\,\mathrm{d}W_j=\rho_{ij}\,\mathrm{d}t', 'Gaussian SDE: relative wealth change equals drift times dt plus volatility times a correlated Brownian increment.'))
            parts.append('<p>The code simulates the joint log increments directly, one exchange session at a time:</p>')
            parts.append(_equation(r'\Delta\log\mathbf{G}=\mathbf{m}+B\mathbf{z},\qquad\mathbf{z}\sim\mathcal{N}(\mathbf{0},I),\qquad BB^{\mathsf{T}}=C', 'Daily joint log increments equal the historical log mean vector plus covariance-scaled standard normal shocks.'))
            parts.append(_equation(r'\mu_i=m_i+\frac{1}{2}C_{ii}\quad\mathrm{(daily\ units)}', 'In daily units, arithmetic SDE drift equals mean log return plus half the historical log-return variance.'))
            parts.append('<p>Here m is the historical daily log-return mean vector, C is its covariance matrix, B is a covariance factor and z is a vector of independent standard normals. G denotes normalized wealth and rho denotes Brownian correlation. The half-variance correction is already embedded in the log formulation and is not subtracted again. Holdings and required FX factors evolve jointly before portfolio aggregation.</p>')
            parts.append('<p><strong>Student-t extension.</strong> The same daily mean/covariance are used with fitted, radially truncated multivariate Student-t shocks, rescaled to retain the target covariance. This is a discrete-time fat-tail return model, not an exact Brownian SDE solution. The fitted degrees of freedom and truncation are recalculated from the applicable calibration sample.</p>')
            if r['student_t_settings']:
                parts.append('<details><summary>Saved Student-t assumptions</summary><pre>'+esc(json.dumps(r['student_t_settings'],indent=2))+'</pre></details>')
            view = r.get("user_scenario")
            if view:
                parts.append("<h4>User assumptions on the historical forecast</h4>")
                if view["status"] == "complete":
                    periods = view["requested"]["periods_per_year"]
                    parts.append(f'<p>Baseline historical calibration: {esc(r["actual_history_start"])} to {esc(r["actual_history_end"])} ({esc(r["matched_daily_intervals"])} matched intervals). Forecast: {esc(r["actual_forecast_start"])} to {esc(r["actual_forecast_end"])} with {esc(r["paths"])} paths per shock model. Annual compound growth g sets daily log drift to log(1+g)/{esc(periods)}. Annual native volatility is divided by the square root of {esc(periods)}. Baseline correlations, FX parameters and Student-t tail parameters are retained.</p>')
                    applied = pd.DataFrame(view["applied"])
                    parts.append(_table(applied,percent=[c for c in applied if c.startswith(("historical_","effective_"))]))
                else:
                    parts.append("<p>Unavailable: "+esc(view["reason"])+"</p>")
            stresses=r['stress_calibrations']
            if stresses:
                parts.append('<p>Stress windows are separate conditional experiments, not extra observations added to the baseline calibration. A baseline drift choice borrows the baseline daily log mean; a stress drift choice uses the stress-period mean.</p>')
                parts.append(_table(pd.DataFrame([dict(name=x.get('name'),start=x.get('start'),end=x.get('end'),drift=x.get('drift'),status=x.get('status'),matched_intervals=(x.get('calibration') or {}).get('observations')) for x in stresses])))
        else:
            parts.append('<p><strong>Physical local-volatility SDE.</strong> Historical drift is combined with a volatility coefficient that can change with time and the simulated state:</p>')
            parts.append(_equation(r'\frac{\mathrm{d}G_i(t)}{G_i(t)}=\mu_i\,\mathrm{d}t+\sigma_i(t,\mathbf{G}(t))\,\mathrm{d}W_i(t),\qquad\mathrm{d}W_i\,\mathrm{d}W_j=\rho_{ij}\,\mathrm{d}t', 'Physical SDE: historical arithmetic drift plus state-dependent volatility times correlated Brownian increments.'))
            parts.append('<p>The numerical log-Euler update evaluates volatility at the beginning of each step:</p>')
            parts.append(_equation(r'\Delta\log G_i=\left(\mu_i-\frac{1}{2}\sigma_i^2(t,\mathbf{G}(t))\right)\Delta t+\sigma_i(t,\mathbf{G}(t))\sqrt{\Delta t}\,z_i', 'Log-Euler: drift minus half current variance times the time step, plus current volatility times square root of the time step times a correlated normal shock.'))
            parts.append(_equation(r'\mathbf{z}\sim\mathcal{N}(\mathbf{0},R),\qquad R_{ij}=\rho_{ij},\qquad\mu_i=A m_i+\frac{1}{2}A C_{ii}', 'Shocks have historical correlation matrix R; annual arithmetic drift uses A times daily log mean plus half A times daily variance.'))
            parts.append('<p>Mapped holdings use the calibrated local-volatility evaluator, including the configured underlying/proxy mapping represented by the state-dependent coefficient. Unmapped holdings and FX retain historical volatility. Historical shock correlation is held fixed while the state-dependent covariance changes with local volatility. Unlike the historical formula above, z here denotes already correlated standard-normal shocks.</p>')
            cfg=r['physical_settings'] or {};annual=cfg.get('annualization_periods','not recorded')
            parts.append(f'<p>The saved annualization factor A is {esc(annual)}; historical annual variance is A times the daily variance. The local-volatility clock remains ACT/365F. Historical drift with option-calibrated diffusion is a physical scenario assumption, not an empirical estimate of future local volatility. Buy-and-hold aggregation applies initial weights once; total-return proxies avoid adding dividends twice.</p>')
    parts.append('<p>Missing options or local volatility do not prevent historical simulation reporting. Missing simulations do not prevent available descriptive statistics from being reported. Unavailable stages retain their reasons in the availability section.</p></section>')
    return ''.join(parts)


def _render(result):
    esc = lambda value: html.escape(str(value), quote=True)
    parts = [f"<header><p>ACADEMIC PROJECT / FINAL ANALYSIS</p><h1>Portfolio evidence and simulation comparison</h1><p>Archive COB {esc(result.cob)} | Analysis {esc(result.run_id)}</p></header>",
             '<nav><a href="#overview">Overview</a><a href="#modeling">Method and dates</a><a href="#forecast">Portfolio simulations</a><a href="#vol">Volatility study</a><a href="#history">Historical evidence</a><a href="#sources">Sources and policy</a></nav>',
             '<main><section id="overview"><h2>What can this run tell us?</h2>',
             '<p>This report uses the saved outputs of this execution. Generating this summary made no provider calls and did not rerun models. Missing optional data does not invalidate independent historical evidence.</p>',
             '<ul>' + ''.join(f'<li>{esc(x)}</li>' for x in result.findings) + '</ul>',
             '<h3>Portfolio requested</h3>', _table(pd.DataFrame(result.portfolio.get("holdings", [])), ["request_id","bucket","weight"], ["weight"]),
             f'<p>Requested reporting currency: <strong>{esc(result.portfolio.get("reporting_currency", "Unavailable"))}</strong>. Actual analysis/simulation currencies are shown below. Weights were not renormalized by this report.</p>',
             '<details><summary>Stage availability and exclusions</summary>', _table(pd.DataFrame(result.availability)), '</details></section>']
    parts.append(_modeling_section(result))
    def figures(prefix, exact=False):
        for key, (caption, content) in result.figures.items():
            if (exact and key == prefix) or (not exact and key.startswith(prefix)):
                parts.append(f'<figure><img alt="{esc(caption)}" src="data:image/png;base64,{base64.b64encode(content).decode()}"/><figcaption>{esc(caption)}</figcaption></figure>')
    stats = result.sources.get("statistics")
    history_start = len(parts)
    parts.append('<section id="history"><h2>What does observed history show?</h2>')
    if stats:
        parts.append(f'<p>Daily {esc(result.cfg["primary_return_type"])} returns. Historical analysis currency: <strong>{esc(stats.summary.get("analysis_currency"))}</strong>. Skewness describes asymmetry; excess kurtosis above zero indicates heavier fourth-moment tails than a normal distribution. These are sample estimates, not future probabilities.</p>')
        data = _chosen(stats.table("statistics"), result.cfg)
        columns = ["scope","instrument_id","currency","observations","mean_daily","volatility_annualized","skewness","excess_kurtosis","worst_daily","status"]
        parts.append(_table(data, columns, ["mean_daily","volatility_annualized","worst_daily"]))
        parts.append('<h3>Coverage, growth and drawdown</h3><p>Compare CAGR only over matching dates. A worst daily loss is different from a peak-to-trough drawdown, which can span many sessions.</p>')
        parts.append(_table(stats.table("path_statistics"), ["scope","instrument_id","first_price_date","last_price_date","cagr","max_drawdown","path_status"], ["cagr","max_drawdown"]))
        figures("historical_"); figures("rolling_")
        parts.append('<details><summary>Stress-period historical statistics and correlations</summary>')
        parts.append(_table(_chosen(stats.table("stress_statistics"), result.cfg), ["name","start","end","scope","instrument_id","observations","volatility_annualized","skewness","excess_kurtosis","coverage","status"], ["volatility_annualized"]))
        parts.append(_table(_chosen(stats.table("dependence"), result.cfg), ["instrument_i","instrument_j","observations","correlation"]))
        parts.append('</details>')
    else:
        parts.append('<p>Historical statistics are unavailable. No distribution or correlation has been inferred from missing data.</p>')
    parts.append('</section>')
    history_parts=parts[history_start:]
    del parts[history_start:]
    parts.append('<section id="forecast"><h2>What do the portfolio simulations imply?</h2><p>These are conditional model outcomes, not guarantees. Returns are V(T)/V(0)-1. Mean, median and standard errors below are expressed as percentages of initial wealth. Stress scenarios and tail sensitivities are separate experiments. Risk-neutral underlying paths are excluded from this portfolio forecast comparison.</p>')
    comp = result.comparison
    metrics = ["scenario","category","status","currency","paths","mean_return","median_return","probability_loss","mean_mc_standard_error"]
    core = comp[comp.category.isin(PRIMARY)] if len(comp) else comp
    parts.append(_table(core, metrics, ["mean_return","median_return","probability_loss","mean_mc_standard_error"]))
    parts.append('<ul>'+''.join(f'<li>{esc(x)}</li>' for x in _outcome_interpretation(result))+'</ul>')
    secondary = comp[comp.category.isin(["stress","tail_sensitivity"])]
    if len(secondary):
        parts.append('<details><summary>Stress and tail-sensitivity scenario results</summary>')
        parts.append(_table(secondary,metrics,["mean_return","median_return","probability_loss","mean_mc_standard_error"]))
        parts.append('</details>')
    parts.append('<h3>Monte Carlo portfolio paths and expected growth</h3><p>The shaded fan summarizes the full simulated population at each recorded time. The thin lines are a saved display subset, not all simulated paths. A small path sample can miss rare losses; use the full-population tail plots and probability tables for those. The mean and median curves answer different questions: expected wealth versus the middle simulated outcome. Models keep their original clocks; connecting plotted observations does not create a common daily return grid.</p>')
    figures("mc_growth_"); figures("mc_paths_")
    if result.chart_notes:
        parts.append('<ul>'+''.join(f'<li>{esc(x)}</li>' for x in result.chart_notes)+'</ul>')
    parts.append('<h3>How do the distributions and tails differ?</h3><p>These are empirical terminal portfolio distributions from the simulations, not fitted normal or Student-t PDFs. Gaussian/Student-t describe the shock engines. The same histogram bins cover every observed outcome. In the log-scale tail plots, a higher curve at a fixed loss level means more probability in that downside tail.</p>')
    parts.append(f'<p>Tail panels focus on the lowest/highest {result.cfg["simulation_plots"]["tail_display_fraction"]:.0%} of pooled terminal returns for display. Probabilities always use the entire sample for each model; no outcomes are dropped from risk estimates.</p>')
    figures("mc_density_"); figures("mc_tails_")
    if len(result.loss_thresholds):
        tails=result.loss_thresholds[result.loss_thresholds.scenario.isin(core.scenario)]
        display=tails.pivot(index="scenario",columns="loss_threshold",values="probability")
        display.columns=[f"Loss greater than {x:.0%}" for x in display.columns]
        parts.append('<h3>Probability of specified terminal losses</h3><p>A zero observed count means no such event occurred in this finite sample; it is not proof that the true probability is zero.</p>')
        parts.append(_table(display.reset_index(),percent=list(display.columns)))
    parts.append('<p>Mean MC standard error measures precision of the simulated mean, not the volatility of individual outcomes. All probability intervals are sampling-only; drift estimation, calibration uncertainty and model risk remain outside them.</p>')
    historical_source = result.sources.get("monte_carlo")
    view = historical_source.user_view if historical_source else None
    if view:
        parts.append('<h3>User assumptions versus historical baseline</h3>')
        if view["status"] != "complete":
            parts.append(f'<p>Unavailable: {esc(view["reason"])}</p>')
        else:
            parts.append('<p>Annual native-currency assumptions for the selected holdings. Compound growth sets the daily mean log return. The scenario retains baseline correlations, FX inputs, weights and forecast dates.</p>')
            tail_key = "loss_es_"+probability_label(max(result.cfg["confidence_levels"]))
            columns = ["mean_return","median_return","probability_loss",tail_key]
            baseline = comp[comp.category == "baseline"].set_index("scenario")
            comparisons = []
            for row in comp[comp.category == "user_assumption"].to_dict("records"):
                model = row["scenario"].rsplit("__",1)[-1]
                if model in baseline.index:
                    base = baseline.loc[model]
                    comparisons.append(dict(scenario=model, **{k:base[k] for k in columns}))
                    comparisons.append(dict(scenario=row["scenario"], **{k:row[k] for k in columns},
                        loss_change_pp=100*(row["probability_loss"]-base["probability_loss"])))
            parts.append(_table(pd.DataFrame(comparisons),percent=columns))
            parts.append('<p>Loss change is measured in percentage points against the corresponding baseline shock model.</p>')
            figures("user_mc_paths_"); figures("user_mc_density_")
    stress_context = _stress_context(result)
    if not stress_context.empty:
        parts.append('<h3>Historical stress Monte Carlo comparisons</h3><p>The baseline lookback and each stress calibration window are independent JSON inputs. A shorter baseline lookback does not disable a crisis scenario if the saved archive still contains sufficient matched observations in that crisis window. Missing or insufficient history makes that stress calibration unavailable; baseline analysis continues. Actual interval coverage is reported below, without inventing missing dates.</p>')
        parts.append(_table(stress_context))
        parts.append('<p>These are conditional stress experiments over the same forecast horizon, not forecasts that another crisis will occur. Drift = stress uses the stress-period mean; drift = baseline retains the baseline mean and changes the risk calibration. Tail-cutoff sensitivity variants remain in the separate results above. Optional physical models retain their original eligibility and clock warnings.</p>')
        for index, (name, rows) in enumerate(_stress_groups(result).items()):
            parts.append(f'<h4>Baseline versus {esc(name)} stress</h4>')
            if any(r["category"] == "user_assumption" for r in rows):
                parts.append("<p>The user scenario is a baseline-window alternative shown alongside the crisis calibration. Its assumptions are not applied to the crisis window.</p>")
            group = pd.DataFrame(rows)
            parts.append(_table(group, metrics, ["mean_return","median_return","probability_loss","mean_mc_standard_error"]))
            for kind in ("growth", "paths", "density", "tails", "terminal_distribution", "loss_probability"):
                figures(f"stress_mc_{kind}_{index}", exact=True)
            losses = result.loss_thresholds[result.loss_thresholds.scenario.isin(group.scenario)]
            if len(losses):
                display = losses.pivot(index="scenario", columns="loss_threshold", values="probability")
                display.columns = [f"Loss greater than {x:.0%}" for x in display.columns]
                parts.append(_table(display.reset_index(), percent=list(display.columns)))
    parts.append('<h3>Forecast and comparison eligibility</h3>')
    parts.append(_table(comp, ["scenario","initial_value","forecast_start","forecast_end","comparability","comparison_reason"]))
    parts.append('<p>Approximate terminal comparison does not authorize comparisons of daily losses or maximum drawdown across different observation grids. No such ranking is made here.</p>')
    for alpha in result.cfg["confidence_levels"]:
        parts.append(f'<h3>Terminal downside at {confidence_label(alpha)} confidence</h3><p>Positive VaR/ES is a loss fraction of initial wealth. ES averages the worst exact tail probability, including a fractional boundary observation; it is not a maximum possible loss.</p>')
        parts.append(_table(comp, ["scenario",f"loss_var_{probability_label(alpha)}",f"loss_es_{probability_label(alpha)}",f"expected_tail_paths_{probability_label(alpha)}"], [f"loss_var_{probability_label(alpha)}",f"loss_es_{probability_label(alpha)}"]))
    figures("terminal_"); figures("loss_probability_")
    parts.append('<details><summary>Sampling intervals and calibration windows</summary>')
    parts.append(_table(comp, ["scenario","mean_ci_low","mean_ci_high","loss_probability_low","loss_probability_high"], ["mean_ci_low","mean_ci_high","loss_probability_low","loss_probability_high"]))
    for stage in ("monte_carlo", "local_vol_real_world"):
        src = result.sources.get(stage)
        if src:
            parts.append(f'<h4>{esc(STAGES[stage])}</h4><pre>{esc(json.dumps(src.summary.get("calibration", {}), indent=2))}</pre>')
            if src.summary.get("stress_calibrations"):
                parts.append(f'<pre>{esc(json.dumps(src.summary["stress_calibrations"], indent=2))}</pre>')
    parts.append('</details></section>')
    parts.extend(history_parts)
    parts.append('<section id="vol"><h2>What is available from the volatility study?</h2><p>Option observations can be later than the archive COB. Intraday approximations, boundary assumptions and source eligibility remain in force. Successful repricing is not evidence that physical tail probabilities are accurate.</p>')
    for stage, table in (("volatility_surface","cohort_status"),("local_volatility","cohorts"),("local_vol_mc","cohorts")):
        src = result.sources.get(stage)
        if src:
            parts.append(f'<h3>{esc(STAGES[stage])}</h3>')
            parts.append(_table(src.table(table), ["underlying","valuation_date","route","status","reason","timing_quality","expiries","start_timestamp","end_timestamp","fit_pass","pde_pass","precision_pass","forward_pass","convergence_pass","portfolio_simulation_ready","failed_checks"]))
    parts.append('<h3>Option quotes, fitted smiles and surface construction</h3><p>Each smile compares the IV calculated from saved option quotes with the fitted model at that expiry. The bid/ask IV envelope is shown where inversion was available. The forward is k = log(K/F) = 0. Fitting uses total variance w = IV squared times maturity; the report only displays the existing fit and does not recalibrate it. The saved intraday approximation and original valuation dates still apply.</p>')
    figures("smiles_")
    surface=result.sources.get("volatility_surface")
    if surface:
        parts.append(_table(surface.table("expiry_metrics"),["group_id","expiration","observations","iv_rmse","scaled_price_rmse","inside_spread_fraction","practical_coverage_fraction","market_fit_pass"], ["iv_rmse","inside_spread_fraction","practical_coverage_fraction"]))
        parts.append('<p>IV RMSE is displayed in volatility percentage points (1% means one point of annualized volatility). Strict bid/ask coverage and the configured practical-fit coverage are different checks; the recorded pass/fail flags above are unchanged.</p>')
    figures("volatility_surface_")
    parts.append('<h3>From the implied surface to state-dependent local volatility</h3><p>Implied volatility indexes option strike and maturity. Local volatility is the instantaneous annualized diffusion coefficient evaluated at a future underlying state and time; its plotted coordinate is log(S/F). It is not a direct forecast of a one-day loss. These plots show the supported saved grid. Short-end and wing continuation used by Monte Carlo remain separate assumptions, quantified below.</p>')
    figures("local_volatility_")
    if not any(k in result.sources for k in ("volatility_surface","local_volatility")):
        parts.append('<p>No usable saved surface is available in this execution. The historical analysis above stands independently.</p>')
    physical = result.sources.get("local_vol_real_world")
    if physical:
        parts.append('<h3>Physical-model assumptions and boundary usage</h3>')
        parts.append('<ul>' + ''.join(f'<li>{esc(w)}</li>' for w in physical.summary.get("warnings",[])) + '</ul>')
        parts.append('<p>Wing visits and short-end usage quantify model assumptions, not observed market frequencies. The maximum local volatility is an annualized instantaneous standard deviation; it is not a one-day loss.</p>')
        parts.append(_table(physical.table("boundaries"), percent=["maximum_local_volatility"]))
    parts.append('</section><section id="sources"><h2>Evidence, limitations and next actions</h2><p>Resolve missing currencies or inputs before interpreting results in the reporting currency. Retain diagnostic models as sensitivity cases. Review stress outcomes and boundary exposure; do not select a model solely because its mean return is higher or its loss probability lower.</p>')
    for name, src in result.sources.items():
        # Older statistics/history reports are Markdown, while volatility
        # stages write HTML. Link the verified artifact instead of inventing
        # report.html for every stage. A manifest is a valid final fallback.
        report = src.files.get("report") or src.files.get("report.html") or src.files.get("report.md")
        target = src.folder / (report or "manifest.json")
        label = "detailed source report" if report else "source manifest"
        parts.append(f'<p><strong>{esc(STAGES[name])}</strong>: <a href="{esc(target.as_uri())}">{label}</a> | <code>{esc(src.folder)}</code></p>')
        for warning in src.summary.get("warnings", []):
            parts.append(f'<p class="muted">{esc(warning)}</p>')
    parts.append('<details><summary>Active report policies and reproducibility</summary><pre>' + esc(json.dumps(result.cfg, indent=2)) + '</pre><p>Source manifest hashes, report code hash and dependency versions are recorded in summary.json. comparison.csv contains unrounded terminal metrics. The HTML embeds all its figures; detailed source links require the original folders.</p></details></section></main>')
    css = 'body{margin:0;background:#f2f5f9;color:#1e293b;font:16px/1.6 system-ui,sans-serif}header{background:#142b45;color:white;padding:40px max(24px,calc((100% - 1180px)/2))}header p{color:#b8d7eb}h1{font-size:32px;line-height:1.25}nav{position:sticky;top:0;background:white;padding:14px;display:flex;gap:24px;flex-wrap:wrap;box-shadow:0 2px 8px #ddd;z-index:2}nav a,a{color:#126a8e}main{max-width:1180px;margin:auto;padding:24px}section{background:white;border-radius:12px;padding:28px;margin-bottom:24px;scroll-margin-top:90px}h2{font-size:25px;border-bottom:2px solid #d7e8ef;padding-bottom:12px}h3{margin-top:26px}.table-wrap{overflow:auto}table{border-collapse:collapse;font-size:13px;width:100%}th,td{text-align:left;padding:10px;border-bottom:1px solid #e2e8f0;white-space:nowrap}th{background:#edf4f8}tbody tr:nth-child(even){background:#f8fafc}figure{margin:24px 0}img{width:100%;height:auto}figcaption,.muted{font-size:14px;color:#52657b}details{background:#f8fafc;padding:16px;margin-top:16px}summary{cursor:pointer;font-weight:600}pre,code{white-space:pre-wrap;overflow-wrap:anywhere}li{margin:10px 0}@media print{nav{position:static}section{break-inside:auto}figure{break-inside:avoid}}'
    css += '.equation{text-align:center;background:#f8fafc;padding:20px 12px;margin:18px 0;overflow-x:auto}.equation img{width:auto;max-width:100%;height:auto;vertical-align:middle}'
    return '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Academic project - portfolio analysis</title><style>' + css + '</style></head><body>' + ''.join(parts) + '</body></html>'


class _MarkdownReport(HTMLParser):
    """Translate our own HTML report, so both formats share every metric.

    This is a deliberately small renderer for the tags emitted by _render(),
    not a general web-page converter. Tables keep their formatted values;
    embedded PNGs become links to the SAME saved figures. SVG equations remain
    inline HTML images to retain scientific notation without network access.
    """
    SECTIONS = ("overview", "modeling", "forecast", "vol", "history", "sources")

    def __init__(self, figures, sections, include_figures):
        super().__init__(convert_charrefs=True)
        selected = self.SECTIONS if sections is None else sections
        if isinstance(selected, str) or not isinstance(selected, (list, tuple)) or any(s not in self.SECTIONS for s in selected):
            raise ValueError("Report sections must be a list drawn from: " + ", ".join(self.SECTIONS))
        self.selected = set(selected)
        self.include_figures = include_figures
        self.images = {base64.b64encode(content).decode(): name + ".png" for name, (_, content) in figures.items()}
        self.parts, self.links = [], []
        self.section, self.hidden = None, 0
        self.rows, self.row, self.cell = None, None, None
        self.pre = False

    def visible(self):
        return not self.hidden and (self.section is None or self.section in self.selected)

    def emit(self, value):
        if self.cell is not None:
            self.cell.append(value)
        else:
            self.parts.append(value)

    def handle_starttag(self, tag, attributes):
        attrs = dict(attributes)
        if tag in {"head", "nav"} or (tag == "figure" and not self.include_figures):
            self.hidden += 1
        if tag == "section":
            self.section = attrs.get("id")
        if not self.visible():
            return
        if tag in {"h1", "h2", "h3", "h4"}:
            self.emit("\n\n" + "#" * int(tag[1]) + " ")
        elif tag in {"p", "ul", "ol", "figure"}:
            self.emit("\n\n")
        elif tag == "li":
            self.emit("\n- ")
        elif tag in {"strong", "summary"}:
            self.emit("**")
        elif tag == "em":
            self.emit("*")
        elif tag == "br":
            self.emit(" ")
        elif tag == "pre":
            self.emit("\n\n```text\n"); self.pre = True
        elif tag == "code" and not self.pre:
            self.emit("`")
        elif tag == "a":
            self.links.append(attrs.get("href", "")); self.emit("[")
        elif tag == "table":
            self.rows = []
        elif tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []
        elif tag == "img":
            source, alt = attrs.get("src", ""), attrs.get("alt", "")
            if source.startswith("data:image/png;base64,"):
                name = self.images.get(source.split(",", 1)[1])
                if name and self.include_figures:
                    self.emit("\n\n![" + alt.replace("]", "\\]") + "](" + name + ")\n\n")
            elif source.startswith("data:image/svg+xml;base64,"):
                self.emit('\n\n<img alt="' + html.escape(alt, quote=True) + '" src="' + source + '"/>\n\n')

    def handle_endtag(self, tag):
        if tag in {"head", "nav"} or (tag == "figure" and not self.include_figures):
            self.hidden -= 1
            return
        if tag == "section":
            self.section = None
            return
        if not self.visible():
            return
        if tag in {"td", "th"}:
            self.row.append("".join(self.cell).strip().replace("|", "\\|").replace("\n", " "))
            self.cell = None
        elif tag == "tr":
            self.rows.append(self.row)
        elif tag == "table":
            if self.rows:
                rows = self.rows[:1] + [["---"] * len(self.rows[0])] + self.rows[1:]
                self.emit("\n\n" + "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n\n")
            self.rows = None
        elif tag in {"strong", "summary"}:
            self.emit("**")
        elif tag == "em":
            self.emit("*")
        elif tag == "pre":
            self.emit("\n```\n\n"); self.pre = False
        elif tag == "code" and not self.pre:
            self.emit("`")
        elif tag == "a":
            self.emit("](<" + self.links.pop() + ">)")
        elif tag in {"h1", "h2", "h3", "h4", "p", "ul", "ol", "figcaption", "details", "figure"}:
            self.emit("\n\n")

    def handle_data(self, text):
        if self.visible() and (self.rows is None or self.cell is not None):
            # Ignore HTML indentation between table cells, preserving cell text.
            self.emit(text if self.pre else html.escape(re.sub(r"\s+", " ", text), quote=False))


def _markdown(result, page, sections=None, include_figures=True):
    parser = _MarkdownReport(result.figures, sections, include_figures)
    parser.feed(page)
    parser.close()
    return re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", "".join(parser.parts)).strip() + "\n"
