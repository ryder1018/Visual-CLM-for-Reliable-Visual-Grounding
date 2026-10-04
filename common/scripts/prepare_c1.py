"""Build the query/region debug subset from official gRefCOCO and COCO."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import zipfile


REVISION = '81eede59b3ac070049f597d023c0ff08d1fb80e9'
HF_BASE = f'https://huggingface.co/datasets/FudanCVL/gRefCOCO/resolve/{REVISION}'
COCO_BASE = 'https://s3.amazonaws.com/images.cocodataset.org'


def select_pairs(refs, images, annotations, count=20, seed=42, selection=None):
    """Return paired single-target/no-target queries with ALL same-image boxes.

    Only train references are eligible. Prefer small, non-crowd candidate sets
    for manual debugging; these sampling filters are not an evaluation protocol.
    """
    selection = selection or {}
    excluded = set(selection.get('excluded_image_ids', []))
    overrides = selection.get('sentence_overrides', {})
    grouped = defaultdict(lambda: {'positive': [], 'negative': []})
    for ref in refs:
        if ref['split'] != 'train' or not ref['sentences'] or ref['image_id'] in excluded:
            continue
        if ref['no_target']:
            grouped[ref['image_id']]['negative'].append(ref)
        elif len(ref['ann_id']) == 1 and ref['ann_id'][0] >= 0:
            grouped[ref['image_id']]['positive'].append(ref)
    image_ids = sorted(grouped)
    random.Random(seed).shuffle(image_ids)
    result = []
    for image_id in image_ids:
        group, image = grouped[image_id], images.get(image_id)
        anns = sorted(annotations.get(image_id, []), key=lambda ann: ann['id'])
        if not image or not group['negative'] or not 2 <= len(anns) <= 15:
            continue
        boxes = [[a['bbox'][0], a['bbox'][1], a['bbox'][0] + a['bbox'][2],
                  a['bbox'][1] + a['bbox'][3]] for a in anns]
        if any(a.get('iscrowd', 0) or not (0 <= b[0] < b[2] <= image['width']
               and 0 <= b[1] < b[3] <= image['height']) for a, b in zip(anns, boxes)):
            continue
        ids = [a['id'] for a in anns]
        positives = [r for r in group['positive'] if r['ann_id'][0] in ids]
        if not positives:
            continue
        positive = min(positives, key=lambda r: r['ref_id'])
        negative = min(group['negative'], key=lambda r: r['ref_id'])
        for ref, groundable in ((positive, True), (negative, False)):
            sentence = min(ref['sentences'], key=lambda s: s['sent_id'])
            if str(ref['ref_id']) in overrides:
                chosen = overrides[str(ref['ref_id'])]
                options = [s for s in ref['sentences'] if s['sent_id'] == chosen]
                if not options:
                    raise ValueError(f"ref {ref['ref_id']} has no sentence {chosen}")
                sentence = options[0]
            target = ids.index(ref['ann_id'][0]) if groundable else None
            result.append(dict(
                sample_id=f"grefcoco_train_ref{ref['ref_id']}_sent{sentence['sent_id']}",
                image_id=image_id, image_path=f"images/{image['file_name']}",
                query=sentence['raw'], boxes=boxes, candidate_ids=ids,
                groundable=groundable, gt_box=boxes[target] if groundable else None,
                target_index=target, split='train',
                source=dict(dataset='gRefCOCO', revision=REVISION, ref_id=ref['ref_id'],
                            sent_id=sentence['sent_id'], candidates='COCO2014_instances')))
        if len(result) == count * 2:
            return result
    raise ValueError(f'only {len(result) // 2} eligible image pairs, need {count}')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download(url, path):
    """Download to a partial file; failures never become a completed file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_name(path.name + '.part')
        subprocess.run(['curl', '-fLsS', '-C', '-', '--connect-timeout', '30',
                        '--max-time', '900', url, '-o', str(part)], check=True)
        part.replace(path)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    import ijson
    from PIL import Image
    from common.data import load_samples
    from common.scripts.review_c1 import render_review

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw'))
    parser.add_argument('--out', type=Path, default=Path('data/debug'))
    parser.add_argument('--count', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--selection', type=Path, default=Path('data/debug/selection.json'))
    args = parser.parse_args()
    if args.count < 1:
        parser.error('--count must be positive')
    if any((args.out / f).exists() for f in ('samples.jsonl', 'review.json', 'manifest.json')):
        parser.error('output already exists; use a new --out directory to preserve reviews')
    paths = [args.raw / 'grefcoco/grefs(unc).json', args.raw / 'grefcoco/instances.json',
             args.raw / 'coco/annotations/instances_train2014.json']
    download(f'{HF_BASE}/grefs%28unc%29.json', paths[0])
    download(f'{HF_BASE}/instances.json', paths[1])
    if not paths[2].exists():
        archive = args.raw / 'coco/annotations_trainval2014.zip'
        download(f'{COCO_BASE}/annotations/annotations_trainval2014.zip', archive)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extract('annotations/instances_train2014.json', args.raw / 'coco')
    refs = json.loads(paths[0].read_text(encoding='utf-8'))
    wanted = {r['image_id'] for r in refs if r['split'] == 'train'}
    print('Reading COCO annotations with a streaming parser...', flush=True)
    images = {}
    with paths[2].open('rb') as stream:
        for image in ijson.items(stream, 'images.item', use_float=True):
            if image['id'] in wanted:
                images[image['id']] = image
            if len(images) == len(wanted):
                break
    annotations = defaultdict(list)
    with paths[2].open('rb') as stream:
        for ann in ijson.items(stream, 'annotations.item', use_float=True):
            if ann['image_id'] in wanted:
                annotations[ann['image_id']].append({k: ann[k] for k in
                    ('id', 'image_id', 'category_id', 'bbox', 'iscrowd')})
    selection = json.loads(args.selection.read_text(encoding='utf-8'))
    samples = select_pairs(refs, images, annotations, args.count, args.seed, selection)
    del refs, annotations
    assets = []
    for sample in samples[::2]:
        image = images[sample['image_id']]
        url = f"{COCO_BASE}/train2014/{image['file_name']}"
        path = args.out / sample['image_path']
        download(url, path)
        with Image.open(path) as actual:
            actual.load()
            if actual.size != (image['width'], image['height']):
                raise ValueError(f'COCO image dimensions disagree: {path}')
        assets.append(dict(image_id=sample['image_id'], path=sample['image_path'],
                           url=url, sha256=sha256(path), width=image['width'], height=image['height']))
        print(f"Downloaded {image['file_name']}", flush=True)
    sample_path = args.out / 'samples.jsonl'
    sample_path.write_text(''.join(json.dumps(s, ensure_ascii=False) + '\n' for s in samples), encoding='utf-8')
    load_samples(sample_path)
    dataset_hash = sha256(sample_path)
    urls = [f'{HF_BASE}/grefs%28unc%29.json', f'{HF_BASE}/instances.json',
            f'{COCO_BASE}/annotations/annotations_trainval2014.zip']
    manifest = dict(dataset='gRefCOCO_C1_debug', created_at=datetime.now(timezone.utc).isoformat(),
        source_split='train', seed=args.seed, count_per_label=args.count, human_review='pending',
        samples_sha256=dataset_hash, images=assets,
        sources=[dict(path=str(p), url=u, sha256=sha256(p)) for p, u in zip(paths, urls)],
        selection=dict(candidate_count=[2, 15], exclude_crowd=True, exclude_multi_target=True,
                       candidate_order='annotation_id', bbox_conversion='xywh_to_xyxy_no_clipping',
                       curation=selection, curation_path=str(args.selection),
                       curation_sha256=sha256(args.selection)),
        cross_dataset_image_disjoint='not_checked; debug only')
    write_json(args.out / 'manifest.json', manifest)
    review = dict(samples_sha256=dataset_hash, reviewer='', reviewed_at=None,
                  samples=[dict(sample_id=s['sample_id'], status='pending', notes='') for s in samples])
    write_json(args.out / 'review.json', review)
    render_review(sample_path, args.out / 'review.html')
    print(f'Validated {len(samples)} samples; human review still pending.', flush=True)


if __name__ == '__main__':
    main()
