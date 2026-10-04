"""Read sample JSONL; image paths are relative to the JSONL directory."""

import json
import math
from pathlib import Path

from PIL import Image


FIELDS = {'sample_id', 'image_id', 'image_path', 'query', 'boxes',
          'candidate_ids', 'groundable', 'gt_box', 'target_index', 'split'}


def _box_valid(box, width, height):
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) for v in box)
            and 0 <= box[0] < box[2] <= width
            and 0 <= box[1] < box[3] <= height)


def _iou(a, b):
    intersection = (max(0, min(a[2], b[2]) - max(a[0], b[0]))
                    * max(0, min(a[3], b[3]) - max(a[1], b[1])))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def _validate(sample, root):
    if not isinstance(sample, dict) or not FIELDS <= sample.keys():
        raise ValueError('missing required fields or sample is not an object')
    for key in ('sample_id', 'image_path', 'query', 'split'):
        if not isinstance(sample[key], str) or not sample[key].strip():
            raise ValueError(f'{key} must be a nonempty string')
    if type(sample['image_id']) not in (str, int) or sample['image_id'] == '':
        raise ValueError('image_id must be a nonempty string or integer')
    if Path(sample['image_path']).is_absolute():
        raise ValueError('image_path must be relative to the JSONL directory')
    with Image.open(root / sample['image_path']) as image:
        image.load()
        width, height = image.size
    boxes, ids = sample['boxes'], sample['candidate_ids']
    if not isinstance(boxes, list) or not isinstance(ids, list) or len(boxes) != len(ids):
        raise ValueError('boxes and candidate_ids must be equal-length lists')
    if any(type(i) not in (str, int) or i == '' for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('candidate_ids must be unique string/integer IDs')
    if not all(_box_valid(box, width, height) for box in boxes):
        raise ValueError('boxes must be finite, nondegenerate pixel xyxy within the image')
    if type(sample['groundable']) is not bool:
        raise ValueError('groundable must be a boolean')
    gt, target = sample['gt_box'], sample['target_index']
    if not sample['groundable']:
        if gt is not None or target is not None:
            raise ValueError('negative samples require null gt_box and target_index')
    elif not _box_valid(gt, width, height):
        raise ValueError('groundable samples require a valid gt_box')
    elif target is None:
        expected = 'proposal_miss' if boxes else 'no_candidates'
        if sample.get('reason') != expected:
            raise ValueError(f'missing target_index requires reason={expected}')
    elif type(target) is not int or not 0 <= target < len(boxes):
        raise ValueError('target_index must be a valid zero-based integer')
    elif _iou(gt, boxes[target]) < 0.5:
        raise ValueError('target_index candidate must overlap gt_box with IoU >= 0.5')
    elif (sample.get('source', {}).get('candidates') == 'COCO2014_instances'
          and gt != boxes[target]):
        raise ValueError('COCO GT candidate must exactly match gt_box')


def load_samples(path) -> list[dict]:
    """Validate every sample; raise ValueError with file, line and sample ID.

    Return dictionaries without modifying candidate order or relative paths.
    Groundable proposal misses remain positive; their reason is required.
    Human review status is tracked separately and is not inferred here.
    """
    path = Path(path)
    samples, seen = [], set()
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            sample = None
            try:
                sample = json.loads(line)
                _validate(sample, path.parent)
                if sample['sample_id'] in seen:
                    raise ValueError('duplicate sample_id')
                seen.add(sample['sample_id'])
            except (ValueError, OSError, TypeError, KeyError) as error:
                identity = sample.get('sample_id', '?') if isinstance(sample, dict) else '?'
                raise ValueError(f'{path}:{line_number} sample {identity}: {error}') from error
            samples.append(sample)
    return samples
