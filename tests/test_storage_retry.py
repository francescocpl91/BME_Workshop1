import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from src.storage import write_json

class StorageRetryTests(unittest.TestCase):
    def test_transient_lock_retried_with_atomic_contents(self):
        original=Path.replace
        calls=[]
        def flaky(source,target):
            calls.append(source)
            if len(calls)==1:raise PermissionError('temporary lock')
            return original(source,target)
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'status.json';p.write_text('{"old":true}')
            with patch.object(Path,'replace',flaky),patch('src.storage.time.sleep'):
                write_json(p,{'new':True})
            self.assertEqual(json.loads(p.read_text()),{'new':True})
            self.assertEqual(len(calls),2)
    def test_permanent_denial_preserves_previous_file_and_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'status.json';p.write_text('{"old":true}')
            with patch.object(Path,'replace',side_effect=PermissionError('denied')) as replace,patch('src.storage.time.sleep'):
                with self.assertRaises(PermissionError):write_json(p,{'new':True})
            self.assertEqual(json.loads(p.read_text()),{'old':True})
            self.assertEqual(replace.call_count,7)
