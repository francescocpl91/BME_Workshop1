"""Browser report safety, content parity and immutable export checks."""
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
from unittest.mock import patch

from test_surface_quality import practical_policy
from test_volatility_surface import fixture
from src.surface_reporting import render_html, _inline, _cell
from src.surface_reporting import render_report
from src.volatility_surface import fit_calibrated


class Parse(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.hrefs=[];self.tags=[]
    def handle_starttag(self,tag,attrs):
        self.tags.append(tag);attrs=dict(attrs)
        if 'id' in attrs:self.ids.append(attrs['id'])
        if 'href' in attrs:self.hrefs.append(attrs['href'])


class HtmlTests(unittest.TestCase):
    def test_nested_decision_formatting_and_status_labels(self):
        self.assertEqual(_inline('**Decision: `fitted_with_limitations`.**'),
            '<strong>Decision: <code>fitted_with_limitations</code>.</strong>')
        self.assertEqual(_cell('not_assessable_from_single_capture'),'Not assessable from a single capture')
        self.assertIn('badge pass',_cell('PASSED'))

    def test_safe_local_renderer_handles_tables_lists_links_and_unicode(self):
        result=SimpleNamespace(manifest={'identity':{'config':{'plots':False}}},models={})
        text='# Test €STR\n\n## Outcome\n\n<script>alert(1)</script>\n\n| Check | Result |\n|---|---|\n| A \\| B | PASS |\n\n- **First**\n  continued\n- `Second`\n\n[local](report.md) [bad](javascript:alert)\n\n## Outcome\n'
        html=render_html(text,result);parser=Parse();parser.feed(html)
        self.assertNotIn('script',parser.tags)
        self.assertIn('&lt;script&gt;',html)
        self.assertIn('€STR',html)
        self.assertIn('A | B',html)
        self.assertIn('table',parser.tags);self.assertIn('li',parser.tags)
        self.assertEqual(len(parser.ids),len(set(parser.ids)))
        self.assertTrue(all(not x.startswith('javascript:') for x in parser.hrefs))
        self.assertTrue(all(x[1:] in parser.ids for x in parser.hrefs if x.startswith('#')))

    def test_html_is_integrated_checksummed_and_source_data_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=practical_policy();cfg['fit']['model']='ssvi';cfg['plots']=False
            cfg['stability'].update(timing_role='diagnostic',maturity_holdout_scope='supported_domain')
            path=root/'policy.json';path.write_text(json.dumps(cfg))
            cal=fixture(root);before=deepcopy(cal.tables)
            with patch('socket.socket.connect',side_effect=AssertionError('Offline')):
                result=fit_calibrated(cal,path);target=result.save()
            self.assertEqual(before,cal.tables)
            html=(target/'report.html').read_text(encoding='utf-8')
            parser=Parse();parser.feed(html)
            self.assertIn('Outcome',html);self.assertIn('Unsupported holdout',html)
            self.assertIn('report_html',json.loads((target/'manifest.json').read_text())['files'])
            self.assertEqual(target,result.save())
            self.assertTrue(all(x[1:] in parser.ids for x in parser.hrefs if x.startswith('#')))
            (target/'report.html').write_text(html+'modified',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'checksum'):result.save()

    def test_report_separates_supported_pass_timing_and_unsupported_wings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);cfg=practical_policy();cfg['fit']['model']='ssvi'
            cfg['stability'].update(timing_role='diagnostic',maturity_holdout_scope='supported_domain')
            path=root/'policy.json';path.write_text(json.dumps(cfg))
            result=fit_calibrated(fixture(root),path)
            cohort=result.tables['cohort_status'][0];cohort.update(stability_status='passed',stability_scope='supported_domain',
                timing_status='not_assessable_from_single_capture',unsupported_holdout_observations=2)
            text=' '.join(render_report(result).split())
            self.assertIn('Timing evidence',text)
            self.assertIn('Supported conversion domain',text)
            self.assertIn('not_assessable_from_single_capture',text)
            self.assertIn('unsupported',text.lower())
            self.assertFalse(cohort['local_volatility_implemented'])


if __name__=='__main__':unittest.main()
