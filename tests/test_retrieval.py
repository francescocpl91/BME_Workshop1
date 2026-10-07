"""Offline contract tests: synthetic fixtures are never production market data."""
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch, Mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pandas as pd
from market_data.models import DateWindow, Instrument, PriceSeries, normalize, RetrievalError
from market_data.config import load_instruments, split_inputs
from market_data.providers.alpha_vantage import AlphaVantageProvider
from market_data.providers.marketdata import MarketDataAppProvider
from market_data.providers.massive import MassiveProvider
from market_data.providers.fred import FredProvider
from market_data.providers.ecb import ECBProvider
from market_data.providers.yahoo import YahooProvider
from market_data.runner import retrieve
from market_data.storage import SeriesCache

WINDOW = DateWindow(date(2024, 1, 2), date(2024, 1, 3))
ASSET = Instrument('AAPL', currency='USD')
class FakeClient:
    min_interval = 0
    last_request = 0
    def __init__(self, *responses): self.responses, self.calls = iter(responses), []
    def get_json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)
    def get(self, url, **kwargs): return self.get_json(url, **kwargs)

class Tests(unittest.TestCase):
    def test_calendar_lookback(self):
        self.assertEqual(DateWindow.parse('1y', date(2024,2,29)).start, date(2023,2,28))
        self.assertEqual(DateWindow.parse('1mo', date(2024,3,31)).start, date(2024,2,29))
        with self.assertRaises(ValueError): DateWindow.parse('0y')

    def test_normalization_and_conflicts(self):
        data = pd.DataFrame({'date':['2024-01-03','2024-01-02','2023-01-01'], 'close':[2,1,3]})
        out = normalize(PriceSeries(data), WINDOW, 'price')
        self.assertEqual(out.data.close.tolist(), [1,2])
        with self.assertRaises(RetrievalError):
            normalize(PriceSeries(pd.DataFrame({'date':['2024-01-02']*2,'close':[1,2]})),WINDOW,'price')

    def test_routing(self):
        items = load_instruments(split_inputs('SPY spy FRED:DGS10'), None)
        self.assertEqual(len(items), 2)
        self.assertIsNone(items[1].symbol_for('yahoo','price'))
        self.assertIsNone(items[0].symbol_for('fred','observation'))
        self.assertEqual(items[1].symbol_for('fred','observation'), 'DGS10')

    def test_alpha(self):
        client = FakeClient({'Time Series (Daily)':{'2024-01-02':{'1. open':'1','2. high':'2','3. low':'1','4. close':'2','5. volume':'50'}}})
        p = AlphaVantageProvider({'ALPHA_VANTAGE_API_KEY':'test'}, client)
        result = normalize(p.fetch(ASSET,'AAPL',WINDOW),WINDOW,'price')
        self.assertEqual(result.data.volume.iloc[0],50)
        self.assertEqual(result.metadata['close_basis'],'unadjusted')
        self.assertNotIn('adj_close',result.data)

    def test_alpha_entitlement(self):
        p = AlphaVantageProvider({'ALPHA_VANTAGE_API_KEY':'test'}, FakeClient({'Information':'premium endpoint'}))
        with self.assertRaises(RetrievalError) as ctx: p.fetch(ASSET,'AAPL',WINDOW)
        self.assertEqual(ctx.exception.code,'entitlement_error')

    def test_marketdata(self):
        t = int(pd.Timestamp('2024-01-02 09:30',tz='America/New_York').timestamp())
        client = FakeClient({'s':'ok','t':[t],'o':[1],'h':[2],'l':[1],'c':[2],'v':[3]})
        p = MarketDataAppProvider({'MARKETDATA_API_TOKEN':'test'}, client)
        result = normalize(p.fetch(ASSET,'AAPL',WINDOW),WINDOW,'price')
        self.assertEqual(str(result.data.date.iloc[0].date()),'2024-01-02')
        self.assertEqual(client.calls[0][1]['params']['to'],'2024-01-03')

    def test_massive_pagination(self):
        def bar(day): return {'t':int(pd.Timestamp(day+' 00:00',tz='America/New_York').timestamp()*1000),'o':1,'h':2,'l':1,'c':2,'v':3}
        client = FakeClient({'status':'OK','results':[bar('2024-01-02')],'next_url':'https://api.massive.com/next'}, {'status':'OK','results':[bar('2024-01-03')]})
        p = MassiveProvider({'MASSIVE_API_KEY':'test'},client)
        result = normalize(p.fetch(ASSET,'AAPL',WINDOW),WINDOW,'price')
        self.assertEqual(len(result.data),2)
        self.assertEqual(result.metadata['pages'],2)

    def test_massive_rejects_foreign_pagination(self):
        p = MassiveProvider({'MASSIVE_API_KEY':'secret'}, FakeClient({'status':'OK','next_url':'https://example.com/steal'}))
        with self.assertRaises(RetrievalError): p.fetch(ASSET,'AAPL',WINDOW)

    def test_fred_missing_is_not_zero(self):
        client = FakeClient({'seriess':[{'units':'Percent','frequency':'Daily'}]}, {'count':2,'observations':[{'date':'2024-01-02','value':'4.1'},{'date':'2024-01-03','value':'.'}]})
        p = FredProvider({'FRED_API_KEY':'test'},client)
        result = normalize(p.fetch(Instrument('US10Y','observation'),'DGS10',WINDOW),WINDOW,'observation')
        self.assertTrue(pd.isna(result.data.value.iloc[1]))
        self.assertEqual(result.metadata['missing_values'],1)

    def test_ecb_csv(self):
        client = FakeClient('TIME_PERIOD,OBS_VALUE,FREQ,UNIT,UNIT_MULT\n2024-01-02,2.5,B,PCPA,0\n')
        p = ECBProvider({},client)
        out = normalize(p.fetch(Instrument('EUR10Y','observation'),'YC/B.TEST',WINDOW),WINDOW,'observation')
        self.assertEqual(out.data.value.iloc[0],2.5)
        self.assertEqual(out.metadata['unit'],'PCPA')

    def test_yahoo_end_inclusive(self):
        frame = pd.DataFrame({'Open':[1],'High':[2],'Low':[1],'Close':[2],'Adj Close':[1.9],'Volume':[3]},index=pd.DatetimeIndex(['2024-01-02'],tz='America/New_York',name='Date'))
        obj = Mock(history_metadata={'currency':'USD'})
        obj.history.return_value=frame
        with patch('yfinance.Ticker',return_value=obj):
            out = normalize(YahooProvider({},FakeClient()).fetch(ASSET,'AAPL',WINDOW),WINDOW,'price')
        self.assertEqual(obj.history.call_args.kwargs['end'],'2024-01-04')
        self.assertEqual(out.data.adj_close.iloc[0],1.9)

    def test_failure_isolation_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            good=AlphaVantageProvider({'ALPHA_VANTAGE_API_KEY':'test'},FakeClient({'Time Series (Daily)':{'2024-01-02':{'4. close':'2'}}}))
            bad=MassiveProvider({})
            cache=SeriesCache(root/'cache')
            folder, summary=retrieve([ASSET],[good,bad],WINDOW,root/'runs',cache,progress=lambda _:None)
            self.assertEqual(summary.status.tolist(),['ok','missing_key'])
            self.assertTrue((folder/'prices_wide.csv').exists())
            _, again=retrieve([ASSET],[good],WINDOW,root/'runs',cache,progress=lambda _:None)
            self.assertTrue(again.cache_hit.iloc[0])
            self.assertNotIn('ALPHA_VANTAGE_API_KEY',(folder/'manifest.json').read_text())

if __name__ == '__main__': unittest.main()
