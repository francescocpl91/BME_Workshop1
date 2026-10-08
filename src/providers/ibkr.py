"""IBKR contract validation and routing to the historical/snapshot adapters."""
from datetime import date, datetime
import json
import math
from types import SimpleNamespace
from .base import MarketDataProvider
from ..models import RetrievalError

CONTRACT_FIELDS = {'symbol', 'secType', 'exchange', 'currency', 'primaryExchange',
                   'tradingClass', 'lastTradeDateOrContractMonth', 'localSymbol', 'strike', 'right', 'multiplier'}

def validate_ib_input(inputs, kind, currency, contract=None):
    required = {'symbol', 'secType', 'exchange', 'currency'}
    if not isinstance(inputs, dict) or not required <= inputs.keys() or set(inputs) - CONTRACT_FIELDS:
        raise ValueError('IBKR input requires symbol, secType, exchange, currency and documented contract fields.')
    for key, value in inputs.items():
        if key == 'strike':
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('IBKR strike must be a positive number.')
        elif not isinstance(value, str) or not value or value != value.strip():
            raise ValueError('IBKR contract identifiers must be non-empty text without surrounding whitespace.')
    if inputs['currency'] != currency:
        raise ValueError('IBKR input.currency must match request currency.')
    if kind == 'option':
        from ..options import validate_contract
        contract = validate_contract(contract, {}, currency)
        expected = {'symbol': contract['underlying'], 'secType': 'OPT', 'currency': 'USD',
                    'tradingClass': contract['root'],
                    'strike': contract['strike'], 'right': 'C' if contract['option_type'] == 'call' else 'P',
                    'multiplier': str(int(contract['multiplier']))}
        if any(inputs.get(k) != v for k, v in expected.items()):
            raise ValueError('IBKR native option fields must match the common SPX contract.')
        expiry = date.fromisoformat(contract['expiration'])
        occ = contract['root'] + expiry.strftime('%y%m%d') + expected['right'] + f"{round(contract['strike'] * 1000):08d}"
        local_symbol = inputs.get('localSymbol')
        if local_symbol is not None and local_symbol != contract['root'].ljust(6) + occ[len(contract['root']):]:
            raise ValueError('IBKR localSymbol must match the common contract with its root padded to six characters.')
        # An explicit OCC localSymbol identifies the contract without guessing a
        # holiday-sensitive last trading date. TWS must still resolve one match.
        native_date = inputs.get('lastTradeDateOrContractMonth')
        if native_date is not None or local_symbol is None:
            try:
                if not native_date or len(native_date) != 8 or not native_date.isdigit(): raise ValueError
                last_trade = datetime.strptime(native_date, '%Y%m%d').date()
            except ValueError:
                raise ValueError('IBKR options require an exact localSymbol or last trading date as YYYYMMDD.') from None
            if not 0 <= (expiry - last_trade).days <= 7:
                raise ValueError('IBKR last trading date must be within the week ending at expiration; resolved expiration is checked separately.')
        if inputs['exchange'] not in {'CBOE', 'SMART'}:
            raise ValueError('Use CBOE or SMART for SPX option requests.')
        return occ
    if kind != 'price' or inputs['secType'] not in {'STK', 'IND'}:
        raise ValueError('IBKR currently supports stocks/ETFs, indices and SPX options.')
    if set(inputs) - required - {'primaryExchange', 'tradingClass'}:
        raise ValueError('Option fields do not belong to a stock/index request.')
    if inputs['secType'] == 'STK' and not inputs.get('primaryExchange'):
        raise ValueError('Set primaryExchange to disambiguate IBKR stock/ETF listings.')
    return inputs['symbol']

def _model_summary(events, requested_mode):
    # IB uses -1 / max-double for unavailable values. Raw callbacks retain these.
    def valid(value, positive=False):
        return (type(value) in (int, float) and math.isfinite(value)
                and (value > 0 if positive else value >= 0) and value < 1e100)
    models = [e for e in events if e['event'] == 'tickOptionComputation'
              and e['tick_type'] in {13, 83}
              and valid(e['implied_volatility'], True)
              and valid(e['option_price']) and valid(e['underlying_price'], True)]
    if not models:
        raise RetrievalError('no_model_data', 'No complete model observation (IV, option price and underlying price) arrived. Partial quotes and IV may still be present in the raw response; inspect its fields and subscription warnings.')
    latest = models[-1]
    modes = [e['market_data_type'] for e in events if e['event'] == 'marketDataType']
    return {'model_observations': len(models), 'model_tick_type': latest['tick_type'],
            'model_data_timing': 'delayed' if latest['tick_type'] == 83 else 'live_or_frozen',
            'requested_market_data_type': requested_mode,
            'actual_market_data_type': modes[-1] if modes else None,
            'model_received_at_utc': latest['received_at_utc'],
            'implied_volatility': latest['implied_volatility'],
            'model_option_price': latest['option_price'],
            'underlying_price': latest['underlying_price']}

def _resolved_contract(details, instrument, inputs, contract=None):
    if len(details) != 1:
        raise RetrievalError('ambiguous_contract', 'IBKR must resolve exactly one contract; inspect returned contract details.')
    resolved = details[0].contract
    # SMART is a routing choice and may differ from the resolved exchange.
    for key, value in inputs.items():
        if key == 'exchange': continue
        if key == 'primaryExchange' and {str(getattr(resolved, key, '')), value} <= {'NASDAQ', 'ISLAND'}: continue
        actual = getattr(resolved, key, None)
        matches = math.isclose(float(actual), value, abs_tol=1e-8) if key == 'strike' else str(actual) == str(value)
        if not matches: raise RetrievalError('contract_mismatch', 'IBKR resolved a different contract: ' + key)
    if contract is not None:
        # IBKR AM-settled SPX uses last trading day, not legal expiration.
        expected_expiry = contract['expiration'].replace('-', '')
        if getattr(details[0], 'realExpirationDate', '') != expected_expiry:
            raise RetrievalError('contract_mismatch', 'IBKR resolved a different or missing option expiration date.')
        expected_symbol = instrument.symbol_for('ibkr', 'option')
        if getattr(resolved, 'localSymbol', '').replace(' ', '') != expected_symbol:
            raise RetrievalError('contract_mismatch', 'IBKR resolved a different OCC option symbol.')
    if not resolved.conId:
        raise RetrievalError('contract_mismatch', 'IBKR returned no resolved contract ID.')
    resolved.exchange = inputs['exchange']
    return resolved

class IBKRProvider(MarketDataProvider):
    name = 'ibkr'
    api_url = 'TWS socket API (local authenticated session; no API key)'

    def check_access(self):
        from .ibkr_snapshots import sdk
        sdk()
        if self.settings.get('IBKR_HOST', '127.0.0.1') not in {'127.0.0.1', 'localhost', '::1'}:
            raise RetrievalError('configuration_error', 'This adapter connects to local TWS only.')
        try:
            port=int(self.settings.get('IBKR_PORT','7496'))
            client=int(self.settings.get('IBKR_CLIENT_ID','81'))
            timeout=float(self.settings.get('IBKR_TIMEOUT','20'))
            if not 1<=port<=65535 or not 1<=client<=2147483647 or not 1<=timeout<=60:raise ValueError
        except (ValueError,TypeError):
            raise RetrievalError('configuration_error','Invalid IBKR port, client ID or timeout.') from None

    def cache_settings(self):
        return {'bar_size':'1 day','what_to_show':'TRADES','use_rth':True,
                'capture_format':'serialized_sdk_historical_bars','adapter':'ib_insync'}

    def fetch(self, instrument, symbol, window):
        raise RetrievalError('unsupported_operation','Use download_cob() for IBKR raw historical data.')

    def option_model_settings(self):
        return {'market_data_type':int(self.settings.get('IBKR_OPTION_MARKET_DATA_TYPE','1')),
                'capture_seconds':float(self.settings.get('IBKR_OPTION_CAPTURE_SECONDS','60')),
                'capture_format':'serialized_sdk_callbacks','adapter':'ib_insync',
                'method':'reqTickersAsync','snapshot':True,'regulatory_snapshot':False,
                'price_meaning':'IB model value; not an executable bid/ask quote',
                'timestamp_meaning':'Callback receipt UTC, not exchange observation time'}

    def iter_option_models(self, requests, progress=lambda _:None):
        from .ibkr_snapshots import iter_snapshots
        yield from iter_snapshots(self,requests,progress)

    def fetch_raw(self, instrument, symbol, window):
        """Reuse the same history adapter for an explicit single-price request."""
        from .ibkr_history import iter_history
        job=SimpleNamespace(request_id=instrument.ticker,instrument=instrument,
                            window=window,contract=None,provider_options={})
        for result in iter_history([job],self.settings,lambda _:None):
            for payload in result['payloads']:
                self.client.capture_response(json.dumps(payload,allow_nan=False).encode(),None,
                    'application/json','ibkr://historical/'+payload.get('method','response'))
            if result['status']!='response_saved':
                raise RetrievalError(result['status'],result['message'])
