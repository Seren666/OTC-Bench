"""Verify downloaded provenance, full label counts and the delivered report."""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import html
import json
import math
import fitz

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    design = json.loads((ROOT / 'design/frozen_design.json').read_text())
    report = json.loads((ROOT / 'analysis/report.json').read_text())
    provenance = json.loads((ROOT / 'evidence/payload_verification_all.json').read_text())
    assert report['status'] == 'COMPLETE' and provenance['status'] == 'PASS'
    assert report['frozen_design_sha256'] == provenance['frozen_design_sha256'] == sha(ROOT / 'design/frozen_design.json')
    assert len(design['models']) == 10 and len(design['arms']) == 6
    assert design['targets'] == 500 and design['requests_per_arm_model'] == 3000
    assert len(provenance['checks']) == 60
    assert sum(x['primary'] for x in provenance['checks']) == 180000
    assert {(x['model'], x['arm']) for x in provenance['checks']} == {(m, a) for m in design['models'] for a in design['arms']}
    for item in provenance['checks']:
        assert item['primary'] == 3000
        for name, digest in item['files_sha256'].items():
            assert sha(ROOT / 'results' / item['model'] / item['arm'] / name) == digest
    for name, digest in report['source_hashes'].items():
        assert sha(ROOT / name) == digest
    total = Counter()
    targets = Counter()
    nlabels = 0
    with (ROOT / 'analysis/labels.csv').open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[k] for k in ['model', 'arm', 'rule', 'budget', 'condition'])
            total[key] += 1
            targets[key] += row['label'] == 'target'
            nlabels += 1
    assert nlabels == 720000
    lookup = {}
    for row in report['metrics']:
        if row['endpoint'] in {'target_O', 'target_G', 'target_T', 'target_N-near'}:
            condition = row['endpoint'][7:]
            key = tuple(row[k] for k in ['model', 'arm', 'rule', 'budget']) + (condition,)
            assert total[key] == (500 if condition in ['O', 'G'] else 1000)
            assert math.isclose(row['estimate'], 100 * targets[key] / total[key], abs_tol=1e-10)
            lookup[key] = row['estimate']
    assert len(lookup) == 960
    all_metrics = {tuple(row[k] for k in ['model', 'arm', 'rule', 'budget', 'endpoint']): row['estimate']
                   for row in report['metrics']}
    for row in report['summary']:
        assert row['n_prompts'] == 5
        model, rule, budget, endpoint = (row[k] for k in ['model', 'rule', 'budget', 'endpoint'])
        values = [all_metrics[model, arm, rule, budget, endpoint] for arm in ['P1', 'P2', 'P3', 'P4', 'P5']]
        mean = sum(values) / 5
        sd = math.sqrt(sum((value - mean)**2 for value in values) / 4)
        assert math.isclose(row['mean'], mean, abs_tol=1e-10)
        assert math.isclose(row['sample_sd'], sd, abs_tol=1e-10)
        assert row['original'] == all_metrics[model, 'original', rule, budget, endpoint]
        for arm, value in zip(['P1', 'P2', 'P3', 'P4', 'P5'], values):
            assert row[arm] == value
    document = (ROOT / 'Prompt_comparison.html').read_text(encoding='utf-8')
    assert document.count('<section>') == 6 and document.count('<table>') == 24
    assert document.count('class="mode active"') == 6
    for prompt in design['prompts'].values():
        assert html.escape(prompt) in document
    expected_cells = Counter()
    for model in design['models']:
        for arm in design['arms']:
            for rule in ['W', 'A']:
                for budget in ['primary32', 'completed128']:
                    for condition in ['O', 'G', 'T', 'N-near']:
                        value = lookup[model, arm, rule, budget, condition]
                        if arm == 'original':
                            text = f'<td>{value:.2f}</td>'
                        else:
                            delta = value - lookup[model, 'original', rule, budget, condition]
                            color = '#087f43' if delta > 1e-9 else '#bf3037' if delta < -1e-9 else '#737b83'
                            text = f'<td>{value:.2f} <span class="delta" style="color:{color}">({delta:+.2f})</span></td>'
                        expected_cells[text] += 1
    for cell, count in expected_cells.items():
        assert document.count(cell) == count, cell
    pdf = fitz.open(ROOT / 'Prompt_comparison.pdf')
    assert len(pdf) == 7
    for i, page in enumerate(pdf):
        text = page.get_text()
        assert ('Original' if i == 0 else 'Prompts 1-5: Mean' if i == 6 else f'Prompt {i}') in text
        for model in design['display_order']:
            assert design['display_names'][model] in text
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    x0, y0, x1, y1 = span['bbox']
                    assert x0 >= 25 and y0 >= 20 and x1 <= page.rect.width - 25 and y1 <= page.rect.height - 20
    output = {
        'status': 'PASS', 'models': 10, 'prompts': 6,
        'primary_outputs': sum(x['primary'] for x in provenance['checks']),
        'length_retries': sum(x['length_retries'] for x in provenance['checks']),
        'still_length128': sum(x['still_length128'] for x in provenance['checks']),
        'labels': nlabels, 'verified_rate_cells': len(lookup),
        'verified_prompt_mean_sd_cells': len(report['summary']),
        'html_tables': 24, 'pdf_pages': 7,
        'report_sha256': {name: sha(ROOT / name) for name in ['Prompt_comparison.html', 'Prompt_comparison.pdf', 'analysis/report.json']},
    }
    (ROOT / 'evidence/final_verification.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
