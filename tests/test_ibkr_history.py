"""Historical adapter isolation and request-shape tests using a fake TWS."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone, date
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ib_insync import BarData
from test_ibkr_snapshots import Event
from test_cob import fixture
from src.config import parse_requests
from src.providers.ibkr_history import iter_history


class HistoricalIB:
    def __init__(self,bar_day=6):
        self.errorEvent=Event();self.connections=0;self.active=0;self.peak=0
        self.calls=[];self.bar_day=bar_day;self.next_id=0
        self.client=SimpleNamespace(reqHistoricalData=lambda *args:None)
        self.loop=asyncio.new_event_loop()
    def connect(self,*args,**kwargs):self.connections+=1
    def disconnect(self):self.loop.close()
    def run(self,coro):return self.loop.run_until_complete(coro)
    def reqMktData(self,*args):raise AssertionError('Historical adapter must never request live data')
    async def reqContractDetailsAsync(self,contract):
        resolved=deepcopy(contract);resolved.conId=int(contract.strike)+1
        return [SimpleNamespace(contract=resolved,realExpirationDate='20261218')]
    async def reqHistoricalDataAsync(self,contract,**kwargs):
        self.next_id+=1;self.client.reqHistoricalData(self.next_id,contract)
        self.calls.append(kwargs);self.active+=1;self.peak=max(self.peak,self.active)
        await asyncio.sleep(.12)
        self.active-=1
        return [BarData(date=datetime(2026,10,self.bar_day,19,59,tzinfo=timezone.utc),open=10,high=11,low=9,close=10,volume=1,average=10,barCount=1)]


class HistoricalTests(unittest.TestCase):
    def test_one_connection_bounded_concurrency_and_explicit_historical_cutoff(self):
        config=fixture(3)
        for r in config['requests']:
            r['timeframe']={'start':'2026-10-06','end':'2026-10-06'}
            r['provider_options']['IBKR_HISTORY_DATA_TYPES']='["TRADES"]'
        jobs=parse_requests(config);ib=HistoricalIB()
        with patch('ib_insync.IB',return_value=ib):
            results=list(iter_history(jobs,{'IBKR_HISTORY_MAX_CONCURRENT':'2','IBKR_HISTORY_REQUEST_INTERVAL':'.1'},lambda x:None))
        self.assertEqual(ib.connections,1);self.assertEqual(len(ib.calls),3)
        self.assertLessEqual(ib.peak,2);self.assertEqual(ib.peak,2)
        self.assertTrue(all(r['status']=='response_saved' for r in results))
        for call in ib.calls:
            self.assertEqual(call['endDateTime'],'20261006-20:00:00')
            self.assertEqual(call['durationStr'],'1 D');self.assertEqual(call['barSizeSetting'],'1 min')
            self.assertFalse(call['keepUpToDate'])
        self.assertFalse(results[0]['info']['historical_greeks_available'])

    def test_previous_day_bars_do_not_count_as_requested_cob(self):
        config=fixture()
        config['requests'][0]['timeframe']={'start':'2026-10-06','end':'2026-10-06'}
        config['requests'][0]['provider_options']['IBKR_HISTORY_DATA_TYPES']='["TRADES"]'
        ib=HistoricalIB(bar_day=5)
        with patch('ib_insync.IB',return_value=ib):
            result=list(iter_history(parse_requests(config),{},lambda x:None))[0]
        self.assertEqual(result['status'],'no_data')
        self.assertEqual(result['info']['channels']['TRADES']['bars_on_cob_before_cutoff'],0)
        self.assertEqual(result['info']['channels']['TRADES']['bars_returned'],1)

    def test_spx_and_ief_fetch_daily_chunks_back_to_head_timestamp(self):
        from pathlib import Path
        import json,tempfile
        from src.json_config import load_provider_configs
        doc=json.loads((Path(__file__).resolve().parents[1]/'config/ibkr.json').read_text())
        doc['requests']=[r for r in doc['requests'] if not r.get('contract')]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json';path.write_text(json.dumps(doc))
            jobs=parse_requests(load_provider_configs(path,cob_date=date(2026,10,7),history_start=date(2000,1,3)))
        class PriceIB(HistoricalIB):
            async def reqContractDetailsAsync(self,contract):
                contract.conId=123
                return [SimpleNamespace(contract=deepcopy(contract))]
            async def reqHeadTimeStampAsync(self,*args,**kwargs):
                return datetime(2024,1,2,tzinfo=timezone.utc)
        ib=PriceIB()
        with patch('ib_insync.IB',return_value=ib):
            results=list(iter_history(jobs,{'IBKR_HISTORY_REQUEST_INTERVAL':'.1'},lambda _:None))
        self.assertEqual(ib.connections,1)
        self.assertEqual({r['request'].instrument.ticker for r in results},{'SPX','IEF','SPY'})
        self.assertEqual(len(ib.calls),9)
        self.assertTrue(all(c['barSizeSetting']=='1 day' and c['whatToShow']=='TRADES' and not c['keepUpToDate'] for c in ib.calls))
        self.assertTrue(all(c['endDateTime']<='20261008-00:00:00' for c in ib.calls))
        self.assertTrue(all(r['info']['history_lower_bound_reached'] for r in results))

    def test_failed_earliest_date_lookup_falls_back_to_bars_and_retains_error(self):
        from pathlib import Path
        import json,tempfile
        from src.json_config import load_provider_configs
        from ib_insync import RequestError
        doc=json.loads((Path(__file__).resolve().parents[1]/'config/ibkr.json').read_text())
        doc['requests']=[r for r in doc['requests'] if r['ticker']=='IEF']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input.json';path.write_text(json.dumps(doc))
            jobs=parse_requests(load_provider_configs(path,cob_date=date(2026,10,7),history_start=date(2000,1,3)))
        class NoHeadIB(HistoricalIB):
            async def reqContractDetailsAsync(self,contract):
                contract.conId=123
                return [SimpleNamespace(contract=deepcopy(contract))]
            async def reqHeadTimeStampAsync(self,*args,**kwargs):
                raise RequestError(4,162,'Historical Market Data Service error message:Query failed')
            async def reqHistoricalDataAsync(self,contract,**kwargs):
                if len(self.calls)>=2:
                    self.calls.append(kwargs)
                    return []
                return await super().reqHistoricalDataAsync(contract,**kwargs)
        ib=NoHeadIB()
        with patch('ib_insync.IB',return_value=ib):
            result=list(iter_history(jobs,{'IBKR_HISTORY_REQUEST_INTERVAL':'.1'},lambda _:None))[0]
        self.assertEqual(result['status'],'response_saved')
        self.assertEqual(len(ib.calls),3)
        self.assertEqual(result['info']['bars_returned'],2)
        self.assertTrue(result['info']['earliest_date_lookup_failed'])
        self.assertFalse(result['info']['history_lower_bound_reached'])
        self.assertEqual(result['info']['history_stop_reason'],'first_empty_year_without_head_timestamp')
        failure=next(p for p in result['payloads'] if p['method']=='reqHeadTimeStamp')
        self.assertFalse(failure['completed']);self.assertEqual(failure['error_code'],162)

    def test_earliest_lookup_can_precede_fallback_but_explicit_windows_stay_bounded(self):
        from src.models import Instrument, DateWindow
        inputs = dict(symbol='IEF',secType='STK',exchange='SMART',currency='USD',primaryExchange='NASDAQ')
        instrument = Instrument('IEF', 'price', {'ibkr':'IEF'}, 'USD', provider_inputs={'ibkr':inputs})
        job = SimpleNamespace(request_id='IEF', instrument=instrument, contract=None,
                              window=DateWindow(date(2000,1,3),date(2001,1,3)), provider_options={})
        class PriceIB(HistoricalIB):
            async def reqContractDetailsAsync(self,contract):
                contract.conId=123
                return [SimpleNamespace(contract=deepcopy(contract))]
            async def reqHeadTimeStampAsync(self,*args,**kwargs):
                return datetime(1999,1,4,tzinfo=timezone.utc)
        for earliest, expected in [(True,'1999-01-04'),(False,'2000-01-03')]:
            ib=PriceIB()
            with self.subTest(earliest=earliest), patch('ib_insync.IB',return_value=ib):
                result=list(iter_history([job],{'IBKR_HISTORY_REQUEST_INTERVAL':'.1'},lambda _:None,earliest))[0]
            self.assertEqual(result['status'],'response_saved')
            self.assertEqual(result['info']['effective_history_start'],expected)
