"""Explicit SPX/SPXW contract identity, independent of history-provider syntax.

Only European, cash-settled S&P 500 index options are supported in this phase.
Option retrieval is raw-only: model values, quotes and trade bars stay distinct.
"""
from datetime import date
from decimal import Decimal, InvalidOperation
import re

RAW_OPTION_PROVIDERS = {"ibkr"}
OPTION_RESPONSE_TYPES = {"ibkr": "serialized_sdk_option_snapshot"}


def validate_contract(contract, symbols, currency):
    required = {"underlying", "root", "expiration", "strike", "option_type",
                "exercise_style", "settlement", "settlement_time", "multiplier"}
    if not isinstance(contract, dict) or set(contract) != required:
        raise ValueError("index_option requires contract fields: " + ", ".join(sorted(required)))
    if contract["underlying"] != "SPX" or contract["root"] not in {"SPX", "SPXW"}:
        raise ValueError("This adapter supports SPX/SPXW index options only; SPY ETF options are different contracts.")
    if contract["exercise_style"] != "european" or contract["settlement"] != "cash":
        raise ValueError("SPX/SPXW contracts require european exercise and cash settlement.")
    expected_settlement = "AM" if contract["root"] == "SPX" else "PM"
    if contract["settlement_time"] != expected_settlement:
        raise ValueError("SPX uses AM settlement; SPXW uses PM settlement.")
    if currency != "USD" or type(contract["multiplier"]) not in (int, float) or contract["multiplier"] != 100:
        raise ValueError("SPX/SPXW contracts require USD and a multiplier of 100 USD per index point.")
    if contract["option_type"] not in {"call", "put"}:
        raise ValueError("contract.option_type must be call or put.")
    if not isinstance(contract["expiration"], str) or not re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}", contract["expiration"]):
        raise ValueError("contract.expiration must be YYYY-MM-DD (2000-2099).")
    expiry = date.fromisoformat(contract["expiration"])
    try:
        if type(contract["strike"]) not in (int, float):
            raise ValueError("contract.strike must be a positive JSON number.")
        strike = Decimal(str(contract["strike"])) * 1000
        if not strike.is_finite() or not 0 < strike <= 99999999 or strike != strike.to_integral_value():
            raise ValueError("contract.strike must fit OCC strike encoding (at most three decimal places).")
    except InvalidOperation:
        raise ValueError("Invalid contract.strike.") from None
    occ = contract["root"] + expiry.strftime("%y%m%d") + ("C" if contract["option_type"] == "call" else "P") + f"{int(strike):08d}"
    for provider, symbol in symbols.items():
        if symbol is None:
            continue
        if provider not in RAW_OPTION_PROVIDERS:
            raise ValueError("index_option history is not implemented for " + provider)
        expected = occ
        if symbol != expected:
            raise ValueError("Option identifier does not match contract root, expiry, side and strike for " + provider)
    return dict(contract)
