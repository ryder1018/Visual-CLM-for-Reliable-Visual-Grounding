import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest


class ReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.samples = self.root / 'samples.jsonl'
        self.samples.write_text('{"sample_id":"p1"}\n{"sample_id":"n1"}\n')
        self.review = dict(samples_sha256=hashlib.sha256(self.samples.read_bytes()).hexdigest(),
                           reviewer='', reviewed_at=None,
                           samples=[dict(sample_id='p1', status='pending', notes=''),
                                    dict(sample_id='n1', status='pending', notes='')])

    def check(self):
        path = self.root / 'review.json'
        path.write_text(json.dumps(self.review))
        try:
            check = importlib.import_module('common.scripts.review_c1').check_review
        except (ImportError, AttributeError):
            self.fail('check_review is not implemented')
        return check(self.samples, path)

    def test_pending_is_reported_without_claiming_approval(self):
        self.assertEqual(self.check(), {'pending': 2, 'approved': 0, 'rejected': 0})

    def test_approval_requires_reviewer_timestamp_and_matching_dataset(self):
        for row in self.review['samples']:
            row['status'] = 'approved'
        with self.assertRaises(ValueError):
            self.check()
        self.review.update(reviewer='test reviewer', reviewed_at='2026-10-04T08:00:00+00:00')
        self.assertEqual(self.check(), {'pending': 0, 'approved': 2, 'rejected': 0})
        self.review['samples_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            self.check()

    def test_rejects_missing_duplicate_and_unknown_review_entries(self):
        variants = [[dict(sample_id='p1', status='pending', notes='')],
                    [dict(sample_id='p1', status='pending', notes='')] * 2,
                    [dict(sample_id='p1', status='approved', notes=''),
                     dict(sample_id='n1', status='unknown', notes='')]]
        for rows in variants:
            with self.subTest(rows=rows):
                self.review['samples'] = rows
                with self.assertRaises(ValueError):
                    self.check()

    def test_approved_review_rejects_non_string_reviewer(self):
        self.review['reviewed_at'] = '2026-10-04T08:00:00+00:00'
        for row in self.review['samples']:
            row['status'] = 'approved'
        for reviewer in (None, 123, True):
            with self.subTest(reviewer=reviewer):
                self.review['reviewer'] = reviewer
                with self.assertRaises(ValueError):
                    self.check()


if __name__ == '__main__':
    unittest.main()
