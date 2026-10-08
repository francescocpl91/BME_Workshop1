"""Index history uses the index endpoint without stock adjustments."""
import unittest
from datetime import date
from src.providers.massive import MassiveProvider
from src.models import Instrument, DateWindow

class Client:
    def __init__(self, payloads): self.payloads=iter(payloads); self.calls=[]
    def get_json(self,url,**kwargs):
        self.calls.append((url,kwargs)); return next(self.payloads)

class IndexRoutingTests(unittest.TestCase):
    def test_massive_index_uses_index_identifier_without_stock_adjustments(self):
        client=Client([{'status':'OK','ticker':'I:SPX','results':[{'t':1,'c':7000}]}])
        MassiveProvider({'MASSIVE_API_KEY':'fixture'},client).fetch_raw(Instrument('SPX'), 'I:SPX', DateWindow(date(2026,10,1),date(2026,10,7)))
        self.assertIn('/v2/aggs/ticker/I%3ASPX/', client.calls[0][0])
        self.assertNotIn('adjusted', client.calls[0][1]['params'])
