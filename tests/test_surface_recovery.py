"""Interrupted derived-output writes recover without changing saved market data."""
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import hashlib,json,unittest

from src.volatility_surface import SurfaceResult, _rename_directory
from src.storage import write_json
from test_svi_surface import settings


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);cfg=settings();cfg['plots']=False
        self.result=SurfaceResult(self.root,'unit',dict(identity=dict(config=cfg),
            summary=dict(skip_reason='no_option_data')),dict(cohort_status=[]),{})
        self.target=self.root/'volatility_surface/unit'

    def signatures(self):
        return {p.name:(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
                for p in self.target.iterdir() if p.is_file()}

    def test_render_failure_leaves_no_final_directory_and_retry_succeeds(self):
        with patch('src.volatility_surface.render_report',side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):self.result.save()
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.glob('.p-*')),[])
        self.assertEqual(self.result.save(),self.target)
        before=self.signatures();self.assertEqual(self.result.save(),self.target)
        self.assertEqual(before,self.signatures())

    def test_manifest_write_failure_is_also_retryable(self):
        def fail_manifest(path,value):
            if path.name=='manifest.json':raise OSError('interrupted finalization')
            return write_json(path,value)
        with patch('src.volatility_surface.write_json',side_effect=fail_manifest):
            with self.assertRaises(OSError):self.result.save()
        self.assertFalse(self.target.exists())
        self.assertEqual(self.result.save(),self.target)

    def test_old_partial_is_preserved_and_recovery_is_recorded(self):
        self.target.mkdir(parents=True);(self.target/'report.md').write_text('earlier partial output')
        self.result.save()
        saved=json.loads((self.target/'manifest.json').read_text())
        backup=self.target.parent/saved['recovered_incomplete_output']
        self.assertEqual((backup/'report.md').read_text(),'earlier partial output')
        self.assertTrue(saved['complete'])
        self.assertNotEqual((self.target/'report.md').read_text(),'earlier partial output')

    def test_failed_publication_restores_old_partial_output(self):
        self.target.mkdir(parents=True);(self.target/'partial.csv').write_text('keep me')
        original=Path.rename
        def interrupted(path,target):
            if path.name=='output':raise OSError('publication interrupted')
            return original(path,target)
        with patch.object(Path,'rename',interrupted):
            with self.assertRaises(OSError):self.result.save()
        self.assertEqual((self.target/'partial.csv').read_text(),'keep me')
        self.assertEqual(self.result.save(),self.target)

    def test_transient_preservation_and_publication_locks_retry(self):
        self.target.mkdir(parents=True);(self.target/'partial.csv').write_text('keep me')
        original=Path.rename;attempts={}
        def locked(path,target):
            key=path.name;attempts[key]=attempts.get(key,0)+1
            if attempts[key] <= 2:raise PermissionError('temporary scanner lock')
            return original(path,target)
        with patch.object(Path,'rename',locked),patch('src.volatility_surface.time.sleep'):
            self.assertEqual(self.result.save(),self.target)
        self.assertEqual(attempts['unit'],3);self.assertEqual(attempts['output'],3)
        saved=json.loads((self.target/'manifest.json').read_text())
        self.assertEqual((self.target.parent/saved['recovered_incomplete_output']/'partial.csv').read_text(),'keep me')

    def test_persistent_publication_lock_restores_partial_with_retry(self):
        self.target.mkdir(parents=True);(self.target/'partial.csv').write_text('keep me')
        original=Path.rename;attempts={}
        def locked(path,target):
            key='restore' if path.name.startswith('.i-') else path.name
            attempts[key]=attempts.get(key,0)+1
            if key=='output' or (key=='restore' and attempts[key]<=2):raise PermissionError('locked')
            return original(path,target)
        with patch.object(Path,'rename',locked),patch('src.volatility_surface.time.sleep'):
            with self.assertRaises(PermissionError):self.result.save()
        self.assertEqual(attempts['output'],7);self.assertEqual(attempts['restore'],3)
        self.assertEqual((self.target/'partial.csv').read_text(),'keep me')

    def test_retry_never_overwrites_destination_created_during_lock(self):
        source=self.root/'source';destination=self.root/'destination';source.mkdir()
        def competing(path,target):
            target.mkdir();(target/'other').write_text('keep me')
            raise PermissionError('destination occupied')
        with patch.object(Path,'rename',competing),patch('src.volatility_surface.time.sleep') as sleep:
            with self.assertRaises(PermissionError):_rename_directory(source,destination)
            sleep.assert_not_called()
        self.assertTrue(source.exists());self.assertEqual((destination/'other').read_text(),'keep me')

    def test_completed_corruption_is_not_overwritten_or_quarantined(self):
        self.result.save();(self.target/'parameters.json').write_text('{} tampered')
        before=self.signatures()
        with self.assertRaisesRegex(ValueError,'checksum'):self.result.save()
        self.assertEqual(before,self.signatures())
        self.assertEqual(list(self.target.parent.glob('.i-*')),[])

    def test_incomplete_manifest_must_have_matching_identity(self):
        self.target.mkdir(parents=True)
        (self.target/'manifest.json').write_text(json.dumps(dict(complete=False,identity={})))
        with self.assertRaisesRegex(ValueError,'identity_mismatch'):self.result.save()
        write_json(self.target/'manifest.json',dict(complete=False,identity=deepcopy(self.result.manifest['identity'])))
        self.assertEqual(self.result.save(),self.target)

    def test_abandoned_private_staging_does_not_block_retry(self):
        abandoned=self.target.parent/'.unit.pending-abandoned';abandoned.mkdir(parents=True)
        (abandoned/'unfinished').write_text('retained')
        self.assertEqual(self.result.save(),self.target)
        self.assertEqual((abandoned/'unfinished').read_text(),'retained')

    def test_long_windows_project_path_can_publish(self):
        # Match a deep OneDrive project: the JSON writer adds its own UUID.
        self.result.folder=self.root/('x'*(150-len(str(self.root))-1))
        target=self.result.folder/'volatility_surface/unit'
        target.mkdir(parents=True)
        name='derivative_refit_domains.csv'
        (target/name).write_text('preserved')
        self.assertEqual(self.result.save(),target)
        manifest=json.loads((target/'manifest.json').read_text())
        backup=target.parent/manifest['recovered_incomplete_output']
        self.assertEqual((backup/name).read_text(),'preserved')

    def test_malformed_manifest_shapes_fail_closed_without_overwriting(self):
        self.result.save()
        path=self.target/'manifest.json'
        original=json.loads(path.read_text(encoding='utf-8'))
        for malformed in ([], None, {**original,'files':[]}, {**original,'files':None}):
            with self.subTest(manifest=malformed):
                path.write_text(json.dumps(malformed),encoding='utf-8')
                before=path.read_bytes()
                with self.assertRaisesRegex(ValueError,'surface_existing_output_inconsistent'):
                    self.result.save()
                self.assertEqual(path.read_bytes(),before)

    def test_output_id_cannot_escape_surface_directory(self):
        for value in ('..','../escape','C:\\escape',''):
            self.result.run_id=value
            with self.assertRaises(ValueError):self.result.save()


if __name__=='__main__':unittest.main()
