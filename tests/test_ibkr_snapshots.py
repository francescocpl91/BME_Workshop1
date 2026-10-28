"""Offline regression tests for the active one-connection snapshot adapter."""
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from src import MarketDataDownloader
from src.config import parse_requests
from src.json_config import load_provider_configs
from src.providers.ibkr import IBKRProvider
from src.providers.ibkr_snapshots import iter_snapshots
from src.models import RetrievalError

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "config"


def config(provider, count=7):
    assert provider == 'ibkr'
    doc = json.loads((FIXTURES/'ibkr.json').read_text())
    doc['requests'] = [r for r in doc['requests'] if r.get('contract')][:count]
    for row in doc['requests']:
        row['operation']='option_model_snapshot';row['frequency']='snapshot';row.pop('timeframe',None)
        row['parameters']={'market_data_type':'live','capture_seconds':1}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'input.json';path.write_text(json.dumps(doc))
        return load_provider_configs(path)


class Event:
    def __init__(self): self.callbacks=[]
    def __iadd__(self, callback): self.callbacks.append(callback);return self
    def emit(self,*args):
        for callback in self.callbacks:callback(*args)


class FakeIB:
    def __init__(self, fail_strike=None, disconnected=False, partial_warning=False, underlying=7800.):
        self.errorEvent=Event();self.connections=0;self.connected=False;self.active=0;self.peak=0
        self.fail_strike=fail_strike;self.disconnected=disconnected;self.next_id=0
        self.requests=[];self.cancelled=[];self.tickers={}
        self.partial_warning=partial_warning;self.ended_by_error=False;self.underlying=underlying
        self.client=SimpleNamespace(reqMktData=self.request,cancelMktData=self.cancelled.append)
        self.wrapper=SimpleNamespace(marketDataType=lambda *a:None,tickOptionComputation=lambda *a:None,
                                     tickSnapshotEnd=lambda *a:None,_endReq=lambda *a:None,endTicker=lambda *a:None,
                                     error=self.error)
    def error(self,*args):self.ended_by_error=True
    def connect(self,*args,**kwargs):
        self.connections+=1
        if self.disconnected:raise ConnectionError('offline')
        self.connected=True
    def disconnect(self):self.connected=False
    def isConnected(self):return self.connected
    def reqMarketDataType(self, mode):self.mode=mode
    def request(self, req_id, contract, *args):self.requests.append((req_id,contract,args))
    async def reqContractDetailsAsync(self, contract):
        contract.conId=int(contract.strike)+1000
        return [SimpleNamespace(contract=deepcopy(contract),realExpirationDate='20261016')]
    async def qualifyContractsAsync(self, contract):
        await self.reqContractDetailsAsync(contract);return [contract]
    async def reqTickersAsync(self,contract,regulatorySnapshot=False):
        self.next_id+=1;ident=self.next_id
        self.client.reqMktData(ident,contract,'',True,regulatorySnapshot,[])
        self.active+=1;self.peak=max(self.active,self.peak)
        ticker=SimpleNamespace(modelGreeks=None);self.tickers[id(contract)]=ticker
        try:
            if self.partial_warning:
                self.wrapper.error(ident,10090,"Partial subscription; other ticks remain active.","")
                if self.ended_by_error:return [ticker]
            await asyncio.sleep(.005)
            self.wrapper.marketDataType(ident,self.mode)
            if contract.strike==self.fail_strike:raise asyncio.TimeoutError()
            self.wrapper.tickOptionComputation(ident,13,0,.2,.5,contract.strike/10,0,.01,.2,-.1,self.underlying)
            self.wrapper.tickSnapshotEnd(ident)
            return [ticker]
        finally:self.active-=1
    def ticker(self, contract):return self.tickers.get(id(contract))
    def run(self, coro):return asyncio.run(coro)


class SnapshotTests(unittest.TestCase):
    def test_seven_contracts_one_connection_five_concurrent_independent_results(self):
        jobs=parse_requests(config('ibkr'));ib=FakeIB()
        provider=IBKRProvider({'IBKR_OPTION_MAX_CONCURRENT':'5'})
        with patch('src.providers.ibkr_snapshots.sdk',return_value=(lambda:ib,SimpleNamespace)):
            results=list(iter_snapshots(provider,jobs,lambda _:None))
        self.assertEqual((ib.connections,ib.peak,len(ib.requests)),(1,5,7))
        self.assertFalse(ib.connected)
        for result in results:
            self.assertIsNone(result['error'])
            self.assertEqual(result['info']['model_option_price'],result['request'].contract['strike']/10)
            body=json.loads(result['body']);self.assertTrue(body['snapshot']);self.assertFalse(body['regulatory_snapshot'])
        self.assertTrue(all(args[1:3]==(True,False) for _,_,args in ib.requests))
    def test_partial_subscription_warning_does_not_end_snapshot_before_quotes(self):
        jobs=parse_requests(config('ibkr',1));ib=FakeIB(partial_warning=True)
        with patch('src.providers.ibkr_snapshots.sdk',return_value=(lambda:ib,SimpleNamespace)):
            results=list(iter_snapshots(IBKRProvider({}),jobs,lambda _:None))
        self.assertFalse(ib.ended_by_error)
        self.assertIsNone(results[0]['error'])
        events=json.loads(results[0]['body'])['events']
        self.assertTrue(any(e.get('code')==10090 for e in events))
        self.assertTrue(any(e['event']=='tickSnapshotEnd' for e in events))
        self.assertGreater(results[0]['info']['model_option_price'],0)

    def test_missing_underlying_keeps_iv_as_partial_in_raw_summary(self):
        conf=config('ibkr',1);conf['settings']={}
        ib=FakeIB(partial_warning=True,underlying=None)
        with tempfile.TemporaryDirectory() as tmp,patch('src.providers.ibkr_snapshots.sdk',return_value=(lambda:ib,SimpleNamespace)):
            result=MarketDataDownloader.from_config(conf).download_raw(tmp)
            row=result.summary.iloc[0]
            self.assertEqual(row['status'],'partial_model_data')
            self.assertEqual(row['implied_volatility'],.2)
            self.assertGreater(row['model_option_price'],0)
            self.assertIsNone(row['underlying_price'])
            self.assertEqual(row['missing_fields'],['underlying_price'])
            self.assertTrue(row['snapshot_completed'])

    def test_raw_run_uses_snapshot_dispatch_even_for_one_contract(self):
        conf=config('ibkr',1);conf['settings']={'IBKR_OPTION_MAX_CONCURRENT':'1'}
        ib=FakeIB()
        with tempfile.TemporaryDirectory() as tmp,patch('src.providers.ibkr_snapshots.sdk',return_value=(lambda:ib,SimpleNamespace)):
            result=MarketDataDownloader.from_config(conf).download_raw(tmp)
            self.assertEqual(result.summary.status.tolist(),['response_saved'])
            response=next(result.folder.rglob('response_001.json'))
            self.assertTrue(json.loads(response.read_text())['snapshot'])
            self.assertEqual(ib.connections,1)

    def test_timeout_preserves_other_contracts_and_cancels_request(self):
        jobs=parse_requests(config('ibkr',3));ib=FakeIB(fail_strike=jobs[0].contract['strike'])
        with patch('src.providers.ibkr_snapshots.sdk',return_value=(lambda:ib,SimpleNamespace)):
            results=list(iter_snapshots(IBKRProvider({}),jobs,lambda _:None))
        self.assertEqual([r['error'].code if r['error'] else 'ok' for r in results],['snapshot_timeout','ok','ok'])
        self.assertEqual(len(ib.cancelled),1)
    def test_offline_tws_returns_a_failure_for_each_input(self):
        jobs=parse_requests(config('ibkr',3));ib=FakeIB(disconnected=True)
        with patch('src.providers.ibkr_snapshots.sdk',return_value=(lambda:ib,SimpleNamespace)):
            results=list(iter_snapshots(IBKRProvider({}),jobs,lambda _:None))
        self.assertEqual([r['error'].code for r in results],['network_error']*3)
        self.assertFalse(ib.connected)
