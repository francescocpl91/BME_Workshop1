"""Read fixed, manually supplied OIS par quotes and archive them without APIs.

No zero rates, discount factors, forwards, maturities or missing conventions are
inferred. Fixed curves keep their source date for every requested COB.
"""
from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re

from .storage import write_json

REGIONS = {'us': ('USD', 'SOFR'), 'euro': ('EUR', 'ESTR')}


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result: raise ValueError('Duplicate OIS JSON key: '+key)
        result[key] = value
    return result


def _constant(value):
    raise ValueError("Non-finite OIS JSON value: "+value)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def load_ois_config(path):
    """Load/validate a manual input. None means the input is explicitly disabled."""
    document = json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=_object, parse_constant=_constant)
    if not isinstance(document, dict): raise ValueError('OIS input must be a JSON object.')
    if type(document.get('schema_version')) is not int or document.get('schema_version') != 1 or document.get('input_type') != 'manual_ois':
        raise ValueError('OIS input requires schema_version=1 and input_type=manual_ois.')
    if type(document.get('enabled')) is not bool:
        raise ValueError('OIS enabled must be true or false.')
    if not document['enabled']: return None
    if document.get('date_policy') != 'fixed_for_all_cobs':
        raise ValueError('Unsupported OIS date policy.')
    if not isinstance(document.get('source'), dict): raise ValueError('OIS source metadata is required.')
    if not isinstance(document.get('curves'), dict) or set(document['curves']) != set(REGIONS):
        raise ValueError('OIS curves must contain us and euro snapshot lists.')
    for region, (currency, benchmark) in REGIONS.items():
        snapshots = document['curves'][region]
        if not isinstance(snapshots, list) or len(snapshots) != 1:
            raise ValueError('Fixed OIS inputs require exactly one curve per region.')
        for curve in snapshots:
            if not isinstance(curve, dict): raise ValueError('OIS snapshot must be an object.')
            if (curve.get('currency'), curve.get('overnight_index'), curve.get('curve_id')) != (currency, benchmark, currency+'_'+benchmark+'_OIS'):
                raise ValueError('OIS currency/index/id mismatch in '+region)
            if curve.get('quote_type') != 'spot_starting_ois_par_swap_rate' or curve.get('rate_unit') != 'percent_per_annum':
                raise ValueError('OIS inputs must be par swap quotes in percent per annum, not zero rates or decimal rates.')
            stamp = curve.get('as_of_date')
            if not isinstance(stamp, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', stamp):
                raise ValueError('OIS as_of_date must be YYYY-MM-DD.')
            date.fromisoformat(stamp)
            if not isinstance(curve.get('conventions'), dict): raise ValueError('OIS convention metadata is required; unknown fields must remain null.')
            points = curve.get('points')
            if not isinstance(points, list) or not points: raise ValueError('OIS curve requires tenor/rate points.')
            tenors = set()
            for point in points:
                if not isinstance(point, dict): raise ValueError('OIS point must be an object.')
                tenor = point.get('tenor')
                if not isinstance(tenor, str) or not re.fullmatch(r'[1-9][0-9]*[DWMY]', tenor) or tenor in tenors:
                    raise ValueError('Invalid or duplicate OIS tenor: '+str(tenor))
                tenors.add(tenor)
                if not _number(point.get('rate_percent')): raise ValueError('OIS rate_percent must be a finite number; negative rates are allowed.')
    return document


def ois_fingerprint(document):
    # Editing explanatory instructions alone must not invalidate saved quotes.
    data = {k:v for k,v in document.items() if k != 'instructions'}
    return hashlib.sha256(json.dumps(data,sort_keys=True,allow_nan=False).encode()).hexdigest()


def plan_ois_dump(document, cob):
    """Return deterministic files/summary rows. No market data or file calls."""
    fingerprint = ois_fingerprint(document)
    result = []
    for region, (currency, benchmark) in REGIONS.items():
        curve = document['curves'][region][0]
        observed = date.fromisoformat(curve['as_of_date'])
        relation = 'same_as_cob' if observed == cob else 'before_cob' if observed < cob else 'after_cob'
        location = 'ois/'+region
        identifier = currency+'_'+benchmark+'_OIS'
        row = dict(request_id=identifier,ticker=identifier,provider='manual_ois',operation='manual_curve_snapshot',
                   retrieval_method='manual_json',cob=str(cob),folder=location,finished=True,
                   status='response_saved',usage_policy='fixed_for_all_cobs',
                   curve_as_of_date=curve['as_of_date'],
                   observation_date_relation=relation,suitable_for_strict_cob=(relation=='same_as_cob'),
                   quote_count=len(curve['points']),responses_saved=1,
                   available_as_of_dates=[curve['as_of_date']],input_fingerprint=fingerprint)
        row['message'] = 'Fixed par OIS curve saved for this COB with its original observation date; bootstrap/convention verification remains necessary.'
        payload = {'schema_version':1,'cob':str(cob),'usage_policy':'fixed_for_all_cobs',
                   'observation_date_relation':relation,
                   'suitable_for_strict_cob':row['suitable_for_strict_cob'],'input_fingerprint':fingerprint,
                   'source':deepcopy(document['source']),'curve':deepcopy(curve),
                   'discount_factors_calculated':False,'forward_rates_calculated':False,'requires_bootstrap':True}
        result.append((location,payload,row))
    return result


def write_ois_dump(document, cob, folder, progress):
    """Save manual input alongside API results within the already-locked COB run."""
    write_json(Path(folder)/'ois'/'input.json',document)
    rows = []
    for location,payload,row in plan_ois_dump(document,cob):
        write_json(Path(folder)/location/'curve.json',payload)
        write_json(Path(folder)/location/'status.json',row)
        rows.append(row)
        progress('OIS [MANUAL] '+row['ticker']+': '+row['status']+'; as of '+str(row['curve_as_of_date'])+'; '+row['observation_date_relation'])
    return rows


def verify_ois_dump(document, cob, folder, saved_rows):
    """Refuse reuse if the saved manual quotes/status are missing or changed."""
    by_key = {(r['provider'],r['request_id']):r for r in saved_rows}
    try:
        stored = load_ois_config(Path(folder)/'ois'/'input.json')
        if stored is None or ois_fingerprint(stored) != ois_fingerprint(document): raise ValueError()
        for location,payload,row in plan_ois_dump(document,cob):
            if json.loads((Path(folder)/location/'curve.json').read_text(encoding='utf-8-sig')) != payload: raise ValueError()
            if json.loads((Path(folder)/location/'status.json').read_text(encoding='utf-8-sig')) != row: raise ValueError()
            if by_key.get(('manual_ois',row['request_id'])) != row: raise ValueError()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise ValueError('Saved OIS input is missing or inconsistent; no API calls or archive writes made.') from exc
