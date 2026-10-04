"""Generate an offline review page and check the saved human review record."""

import argparse
from collections import Counter
from datetime import datetime
import html
import json
from pathlib import Path

from common.scripts.prepare_c1 import sha256


def check_review(sample_path, review_path):
    samples = [json.loads(line) for line in Path(sample_path).read_text(encoding='utf-8').splitlines()]
    review = json.loads(Path(review_path).read_text(encoding='utf-8'))
    if review.get('samples_sha256') != sha256(sample_path):
        raise ValueError('review does not match samples.jsonl SHA256')
    entries = review.get('samples', [])
    expected = [s['sample_id'] for s in samples]
    actual = [s['sample_id'] for s in entries]
    if len(set(actual)) != len(actual) or sorted(actual) != sorted(expected):
        raise ValueError('review must cover each sample exactly once')
    counts = Counter(s['status'] for s in entries)
    if set(counts) - {'pending', 'approved', 'rejected'}:
        raise ValueError('unknown review status')
    if counts['approved'] or counts['rejected']:
        reviewer = review.get('reviewer')
        if not isinstance(reviewer, str) or not reviewer.strip() or not review.get('reviewed_at'):
            raise ValueError('reviewed samples require reviewer and reviewed_at')
        if datetime.fromisoformat(review['reviewed_at']).tzinfo is None:
            raise ValueError('reviewed_at must include a timezone')
    return {status: counts[status] for status in ('pending', 'approved', 'rejected')}


def render_review(sample_path, output_path):
    from PIL import Image
    from common.data import load_samples

    sample_path, output_path = Path(sample_path), Path(output_path)
    samples = load_samples(sample_path)
    cards = []
    for pair_number, (positive, negative) in enumerate(zip(samples[::2], samples[1::2]), 1):
        with Image.open(sample_path.parent / positive['image_path']) as image:
            width, height = image.size
        marks = []
        for index, box in enumerate(positive['boxes']):
            x1, y1, x2, y2 = box
            color = '#ff3154' if index == positive['target_index'] else '#00b5ff'
            marks.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
                         f'fill="none" stroke="{color}" stroke-width="3"/>'
                         f'<text x="{x1+3}" y="{max(y1+18, 18)}" fill="white" stroke="black" '
                         f'stroke-width="3" paint-order="stroke" font-size="18">{index}</text>')
        source = html.escape(positive['image_path'], quote=True)
        rows = []
        for sample, label, criterion in (
            (positive, '正例', '描述有唯一目標，紅框正確'),
            (negative, '負例', '整張圖片沒有符合描述的目標')):
            identity = html.escape(sample['sample_id'], quote=True)
            query = html.escape(sample['query'])
            rows.append(f'<div class="query"><strong>{label}：{query}</strong>'
                        f'<p>{criterion}</p><code>{identity}</code><br>'
                        f'<select data-id="{identity}"><option value="pending">待審核</option>'
                        f'<option value="approved">通過</option><option value="rejected">不通過／有歧義</option>'
                        f'</select><input data-note="{identity}" placeholder="問題或備註" aria-label="備註"></div>')
        cards.append(f'<article><h2>{pair_number:02d} · image {positive["image_id"]} · '
                     f'{len(positive["boxes"])} candidates · target index {positive["target_index"]}</h2>'
                     f'<div class="pictures"><figure><img src="{source}" alt="原圖"><figcaption>原圖：核對負例</figcaption>'
                     f'</figure><figure><div class="overlay"><img src="{source}" alt="候選框">'
                     f'<svg viewBox="0 0 {width} {height}">{"".join(marks)}</svg></div>'
                     f'<figcaption>候選框：紅色為正例目標，數字為 0-based index</figcaption></figure></div>'
                     f'<div class="queries">{"".join(rows)}</div></article>')
    review_path = sample_path.parent / 'review.json'
    review = json.loads(review_path.read_text(encoding='utf-8'))
    data = json.dumps(review, ensure_ascii=False).replace('<', '\\u003c')
    positives = sum(sample['groundable'] for sample in samples)
    document = (TEMPLATE.replace('__CARDS__', '\n'.join(cards)).replace('__DATA__', data)
                .replace('__POSITIVES__', str(positives))
                .replace('__NEGATIVES__', str(len(samples) - positives))
                .replace('__REVIEW_PATH__', html.escape(str(review_path))))
    output_path.write_text(document, encoding='utf-8')


TEMPLATE = r'''<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>C1 圖片與 query 人工審核</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f3f5f8;color:#172233}
header,main{max-width:1250px;margin:auto;padding:24px}header{background:#fff}
article{background:white;border-radius:12px;padding:20px;margin:20px 0}h2{font-size:19px}
.pictures,.queries{display:grid;grid-template-columns:1fr 1fr;gap:20px}figure{margin:0}
img{display:block;width:100%;height:auto}.overlay{position:relative}.overlay svg{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}
.query{padding:14px;background:#f4f7fc;border-radius:8px}.query p{margin:6px 0}
code{font-size:12px;overflow-wrap:anywhere}select,input,button{font:inherit;padding:8px;margin:8px 5px 4px 0}
input[data-note]{width:90%}button{cursor:pointer}#status{font-weight:600}.hide-boxes svg{visibility:hidden}
@media(max-width:760px){.pictures,.queries{grid-template-columns:1fr}header,main{padding:12px}}
</style>
<header><h1>C1：__POSITIVES__ 正例＋__NEGATIVES__ 負例人工審核</h1>
<p>逐張確認正例目標唯一、目標框正確，以及負例描述在整張圖中不存在。看不清楚、有歧義或標註不符請選「不通過」並填原因。</p>
<p>選擇結果後請匯出 JSON，將下載檔案放回 <code>__REVIEW_PATH__</code>。本頁不會直接修改專案檔案；離開前請匯出保存。</p>
<label>審核者 <input id="reviewer" placeholder="姓名或代號"></label>
<button id="export">匯出審核 JSON</button>
<label>匯入先前審核 <input id="import" type="file" accept=".json,application/json"></label>
<button id="toggle">顯示／隱藏候選框</button><p id="status"></p></header>
<main>__CARDS__</main>
<script>
const initial=__DATA__;
function progress(){const counts={pending:0,approved:0,rejected:0};document.querySelectorAll('select[data-id]').forEach(s=>counts[s.value]++);document.querySelector('#status').textContent=`待審核 ${counts.pending} · 通過 ${counts.approved} · 不通過 ${counts.rejected}`;}
function apply(record){
 if(record.samples_sha256!==initial.samples_sha256)throw Error('此審核 JSON 對應不同資料版本');
 const expected=initial.samples.map(s=>s.sample_id).sort();
 const actual=record.samples.map(s=>s.sample_id).sort();
 if(JSON.stringify(expected)!==JSON.stringify(actual))throw Error('審核樣本 IDs 不一致');
 const entries=new Map(record.samples.map(s=>[s.sample_id,s]));
 for(const entry of record.samples)if(!['pending','approved','rejected'].includes(entry.status))throw Error('未知審核狀態');
 document.querySelectorAll('select[data-id]').forEach(s=>s.value=entries.get(s.dataset.id).status);
 document.querySelectorAll('input[data-note]').forEach(i=>i.value=entries.get(i.dataset.note).notes||'');
 document.querySelector('#reviewer').value=record.reviewer||'';progress();
}
apply(initial);document.querySelectorAll('select[data-id]').forEach(s=>s.addEventListener('change',progress));
document.querySelector('#toggle').onclick=()=>document.body.classList.toggle('hide-boxes');
document.querySelector('#import').onchange=async event=>{try{if(event.target.files[0])apply(JSON.parse(await event.target.files[0].text()));}catch(error){alert(error.message);}};
document.querySelector('#export').onclick=()=>{
 const reviewer=document.querySelector('#reviewer').value.trim();
 if(!reviewer){alert('請填寫審核者姓名或代號');return;}
 const notes=new Map(Array.from(document.querySelectorAll('input[data-note]'),i=>[i.dataset.note,i.value]));
 const record={samples_sha256:initial.samples_sha256,reviewer,reviewed_at:new Date().toISOString(),samples:Array.from(document.querySelectorAll('select[data-id]'),s=>({sample_id:s.dataset.id,status:s.value,notes:notes.get(s.dataset.id)}))};
 const url=URL.createObjectURL(new Blob([JSON.stringify(record,null,2)+'\n'],{type:'application/json'}));
 const link=document.createElement('a');link.href=url;link.download='review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
};
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, default=Path('data/debug/samples.jsonl'))
    parser.add_argument('--review', type=Path, default=Path('data/debug/review.json'))
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.render:
        render_review(args.samples, args.samples.parent / 'review.html')
    counts = check_review(args.samples, args.review)
    print(json.dumps(counts))
    return int(bool(counts['pending'] or counts['rejected']))


if __name__ == '__main__':
    raise SystemExit(main())
