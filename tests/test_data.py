import copy
import importlib
import json
import math
import struct
import zlib
from pathlib import Path
import tempfile
import unittest

class SamplesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        def chunk(kind, data):
            return (struct.pack('!I', len(data)) + kind + data
                    + struct.pack('!I', zlib.crc32(kind + data)))
        png = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 24, 16, 8, 2, 0, 0, 0))
               + chunk(b'IDAT', zlib.compress((b'\0' + b'\0' * 72) * 16)) + chunk(b'IEND', b''))
        (self.root / 'image.png').write_bytes(png)
        self.sample = dict(sample_id='p1', image_id=1, image_path='image.png',
                           query='the red cup', boxes=[[1, 2, 10, 12], [12, 2, 23, 15]],
                           candidate_ids=[7, 9], groundable=True,
                           gt_box=[1, 2, 10, 12], target_index=0, split='train')

    def load(self, rows):
        path = self.root / 'samples.jsonl'
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        try:
            loader = importlib.import_module('common.data').load_samples
        except (ImportError, AttributeError):
            self.fail('load_samples is not implemented')
        return loader(path)

    def test_loads_positive_and_negative_without_reordering(self):
        negative = dict(self.sample, sample_id='n1', groundable=False,
                        gt_box=None, target_index=None)
        result = self.load([self.sample, negative])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['candidate_ids'], [7, 9])
        self.assertEqual(result[0]['image_path'], 'image.png')
        self.assertIsNone(result[1]['target_index'])

    def test_rejects_invalid_boxes_with_sample_context(self):
        for box in ([1, 2, 25, 12], [10, 2, 1, 12], [1, 2, math.nan, 12],
                    [True, 2, 10, 12], [1, 2, 10]):
            with self.subTest(box=box):
                row = dict(self.sample, boxes=[box, self.sample['boxes'][1]])
                with self.assertRaisesRegex(ValueError, 'p1'):
                    self.load([row])

    def test_rejects_inconsistent_labels_and_candidates(self):
        variants = [dict(target_index=-1), dict(target_index=2),
                    dict(target_index=True), dict(target_index=1),
                    dict(candidate_ids=[7]), dict(candidate_ids=[7, 7]),
                    dict(groundable=False), dict(groundable=1), dict(gt_box=None)]
        for changes in variants:
            with self.subTest(changes=changes):
                with self.assertRaises(ValueError):
                    self.load([dict(self.sample, **changes)])

    def test_rejects_missing_image_field_and_duplicate_sample(self):
        with self.assertRaises(ValueError):
            self.load([dict(self.sample, image_path='missing.png')])
        row = copy.deepcopy(self.sample)
        del row['query']
        with self.assertRaises(ValueError):
            self.load([row])
        with self.assertRaises(ValueError):
            self.load([self.sample, self.sample])

    def test_preserves_groundable_proposal_miss_and_empty_candidates(self):
        miss = dict(self.sample, target_index=None, reason='proposal_miss')
        empty = dict(self.sample, sample_id='empty', boxes=[], candidate_ids=[],
                     target_index=None, reason='no_candidates')
        result = self.load([miss, empty])
        self.assertTrue(result[0]['groundable'])
        self.assertEqual(result[1]['boxes'], [])
        with self.assertRaises(ValueError):
            self.load([dict(miss, reason=None)])

    def test_gt_candidates_require_exact_target_but_proposals_allow_overlap(self):
        row = dict(self.sample, boxes=[[1, 2, 10, 12], [2, 2, 11, 12]],
                   target_index=1, source={'candidates': 'COCO2014_instances'})
        with self.assertRaises(ValueError):
            self.load([row])
        row['source']['candidates'] = 'detector'
        self.assertEqual(self.load([row])[0]['target_index'], 1)


if __name__ == '__main__':
    unittest.main()
