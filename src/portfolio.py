"""Portfolio input validation only; this module does not value or trade holdings."""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re

from .json_config import _unique_object, _invalid_constant


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
        not math.isclose(sum(allocation.values()), 1, abs_tol=1e-8, rel_tol=0)):
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
    if any(totals[k] > allocation[k] + 1e-8 for k in totals):
        raise ValueError('Known holding weights exceed the target allocation.')
    if not missing and any(not math.isclose(totals[k], allocation[k], abs_tol=1e-8, rel_tol=0) for k in totals):
        raise ValueError('Complete holding weights must match target_allocation and add to one.')
    refs = data['reference_data']
    if not isinstance(refs, list) or any(not isinstance(k, str) or k not in requests for k in refs):
        raise ValueError('reference_data must contain configured request IDs.')
    if len(refs) != len(set(refs)) or seen.intersection(refs):
        raise ValueError('Reference data must be unique and separate from holdings.')
    return PortfolioConfig(data['reporting_currency'], dict(allocation), tuple(dict(h) for h in holdings), tuple(refs), nav)
