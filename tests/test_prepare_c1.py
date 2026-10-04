import importlib
import unittest


class PrepareTest(unittest.TestCase):
    def setUp(self):
        self.refs = [
            dict(image_id=1, split='train', no_target=False, ann_id=[20], ref_id=1,
                 sentences=[dict(sent_id=11, raw='the red cup')]),
            dict(image_id=1, split='train', no_target=True, ann_id=[-1], ref_id=2,
                 sentences=[dict(sent_id=12, raw='the blue cup')]),
        ]
        self.images = {1: dict(id=1, width=100, height=80, file_name='image.jpg')}
        self.anns = {1: [dict(id=20, image_id=1, category_id=47,
                              bbox=[10, 20, 30, 40], iscrowd=0),
                        dict(id=10, image_id=1, category_id=84,
                             bbox=[50, 10, 20, 30], iscrowd=0)]}

    def select(self, refs=None, selection=None):
        try:
            select = importlib.import_module('common.scripts.prepare_c1').select_pairs
        except (ImportError, AttributeError):
            self.fail('select_pairs is not implemented')
        return select(self.refs if refs is None else refs, self.images, self.anns,
                      count=1, seed=42, selection=selection)

    def test_converts_boxes_and_keeps_distractors_in_annotation_order(self):
        positive, negative = self.select()
        self.assertEqual(positive['candidate_ids'], [10, 20])
        self.assertEqual(positive['boxes'], [[50, 10, 70, 40], [10, 20, 40, 60]])
        self.assertEqual(positive['gt_box'], [10, 20, 40, 60])
        self.assertEqual(positive['target_index'], 1)
        self.assertEqual(positive['query'], 'the red cup')
        self.assertEqual(negative['query'], 'the blue cup')
        self.assertFalse(negative['groundable'])
        self.assertIsNone(negative['target_index'])
        self.assertIsNone(negative['gt_box'])
        self.assertEqual(negative['candidate_ids'], positive['candidate_ids'])
        self.assertEqual(self.select(), [positive, negative])

    def test_excludes_multi_target_and_non_train_references(self):
        variants = [dict(ann_id=[10, 20]), dict(split='val'), dict(ann_id=[999])]
        for changes in variants:
            with self.subTest(changes=changes):
                refs = [dict(self.refs[0], **changes), self.refs[1]]
                with self.assertRaises(ValueError):
                    self.select(refs)

    def test_curation_preserves_raw_sentence_id_and_excludes_noisy_images(self):
        self.refs[0]['sentences'].append(dict(sent_id=13, raw='cup on the left'))
        rows = self.select(selection={'sentence_overrides': {'1': 13}})
        self.assertEqual(rows[0]['query'], 'cup on the left')
        self.assertEqual(rows[0]['source']['sent_id'], 13)
        with self.assertRaises(ValueError):
            self.select(selection={'excluded_image_ids': [1]})


if __name__ == '__main__':
    unittest.main()
