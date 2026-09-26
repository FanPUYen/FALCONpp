"""Check every data cell of the six current manuscript tables against CSV exports.

The manuscript is an independent comparison target, never an input to the
numerical computation. This intentionally supports the current table schema;
unknown tables or structural changes fail instead of being silently skipped.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

# CSV row names and corresponding fragments of the original LaTeX row labels.
SCHEMA = [
    ('tab:intro-sampling', [
        ('q', 'Modulus', 'design_input'),
        ('gamma', 'Trapdoor-quality', 'design_input'),
        ('sigma_sig', 'Signature standard deviation', 'design_input'),
        ('FALCONws correction iterations', r'\FalconWS', 'recomputed'),
        ('FALCON++ total trials upper estimate', r'\FalconPP', 'recomputed')]),
    ('tab:intro-comparison', [
        ('FALCON / FALCON+', 'FALCON /', 'literature_input'),
        ('FALCONws', r'\FalconWS', 'literature_input'),
        ('FALCON++', r'\FalconPP', 'recomputed')]),
    ('tab:falconpp-parameters', [
        ('Target security lambda', 'Target security', 'design_input'),
        ('q', 'Modulus', 'design_input'),
        ('gamma', 'Secret-width', 'design_input'),
        ('sigma_sig', 'Signature standard deviation', 'design_input'),
        ('beta', 'Signature norm cap', 'design_input'),
        ('Sampler-loss budget', 'Sampler-loss budget', 'design_input'),
        ('Salt wire length / entropy', 'Salt wire length', 'design_input')]),
    ('tab:key-size-comparison', [
        ('FALCON / FALCON+', r'\FalconPlus{}-512', 'literature_input'),
        ('FALCONws', r'\FalconWS{}-512', 'literature_input'),
        ('FALCON++', r'\FalconPP{}-512', 'own_sizes'),
        ('FALCON / FALCON+', r'\FalconPlus{}-1024', 'literature_input'),
        ('FALCONws', r'\FalconWS{}-1024', 'literature_input'),
        ('FALCON++', r'\FalconPP{}-1024', 'own_sizes')]),
    ('tab:profile-certificates', [
        ('Moment order k', 'Moment order', 'design_input'),
        ('Global Gamma_hat (display only)', 'Global', 'design_input'),
        ('r_infinity upper estimate', r'r_\infty', 'recomputed'),
        ('p_corr lower estimate', r'p_{\rm corr}', 'recomputed'),
        ('p_norm lower estimate', r'p_{\rm norm}', 'recomputed'),
        ('FALCON++ total trials', 'Our sampler total trials', 'recomputed'),
        ('FALCONws correction trials', 'Sampler trials', 'recomputed')]),
    ('tab:section6-attack-margins', [
        ('Key recovery BKZ block size', 'Key recovery: BKZ block size', 'recomputed'),
        ('Key recovery C/Q', 'Key recovery: cost', 'recomputed'),
        ('Forgery BKZ block size', 'Forgery: BKZ block size', 'recomputed'),
        ('Forgery C/Q', 'Forgery: cost', 'recomputed'),
        ('chi-BDD BKZ block size', 'BKZ block size', 'recomputed'),
        ('chi-BDD classical reduction exponent', 'Classical reduction', 'recomputed'),
        ('chi-BDD log2 sufficient samples', 'Samples', 'recomputed')]),
]


def canonical_cell(text):
    text = re.sub(r'\\textbf\{([^{}]*)\}', r'\1', text.strip())
    text = text.replace('$', '')
    text = re.sub(r'\\times10\^\{(-?\d+)\}', r'e\1', text)
    text = re.sub(r'1-e\^\{(-?[\d.]+)\}', r'1-exp(\1)', text)
    return re.sub(r'\s+', '', text)


def manuscript_tables(text):
    # Strip LaTeX comments without discarding escaped percent signs.
    uncommented = re.sub(r'(?<!\\)%[^\n]*', '', text)
    blocks = re.findall(r'\\begin\{table\*?\}.*?\\end\{table\*?\}', uncommented, re.S)
    if len(blocks) != len(SCHEMA):
        raise ValueError(f'Expected exactly six active table environments; found {len(blocks)}')
    tables = []
    for block, (expected_label, schema) in zip(blocks, SCHEMA):
        label = re.search(r'\\label\{([^}]+)\}', block)
        if label is None or label.group(1) != expected_label:
            raise ValueError(f'Unexpected table label/order; expected {expected_label}')
        if r'\midrule' not in block or r'\bottomrule' not in block:
            raise ValueError(f'Unsupported table structure: {expected_label}')
        body = block.split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0]
        body = body.replace(r'\midrule', '')
        rows = [[cell.strip() for cell in row.split('&')]
                for row in re.split(r'\\\\', body) if '&' in row]
        if len(rows) != len(schema):
            raise ValueError(f'Row count changed in {expected_label}: {len(rows)}')
        for row, (_, fragment, _) in zip(rows, schema):
            if fragment not in row[0]:
                raise ValueError(f'Row identity changed in {expected_label}: {row[0]}')
        tables.append(rows)
    return tables


def verify_tables(paper, table_dir):
    text = paper.read_text(encoding='utf-8')
    tables = manuscript_tables(text)
    report = {
        'scope': 'Every displayed data cell in the six active main-text tables; exact display precision after converting LaTeX math notation. Design choices and literature inputs are identified separately from recomputed values. This is not a measured-wire-size claim.',
        'table_source_sha256': hashlib.sha256(paper.read_bytes()).hexdigest(),
        'status': 'pass', 'tables_checked': 0, 'cells_checked': 0,
        'provenance_counts': {}, 'tables': [], 'mismatches': [],
    }
    for number, (paper_rows, (label, schema)) in enumerate(zip(tables, SCHEMA), 1):
        path = table_dir/f'table{number}.csv'
        with path.open(newline='', encoding='utf-8') as f:
            csv_document = list(csv.reader(f))
        expected_header = (
            ['Scheme', '512 pk (B)', '512 sig (B)', '512 C/Q', '1024 pk (B)', '1024 sig (B)', '1024 C/Q'] if number == 2
            else ['Scheme', 'n', 'q', 'pk (B)', 'sig (B)'] if number == 4
            else ['Quantity', 'FALCON++-512', 'FALCON++-1024'])
        if not csv_document or csv_document[0] != expected_header:
            raise ValueError(f'CSV column identities changed: {path.name}')
        generated_rows = csv_document[1:]
        if len(generated_rows) != len(schema):
            raise ValueError(f'CSV row count changed: {path.name}')
        checks = []
        for row_number, (expected, actual, (row_label, _, origin)) in enumerate(zip(paper_rows, generated_rows, schema), 1):
            if not actual or actual[0] != row_label:
                raise ValueError(f'CSV row identity changed: {path.name}, row {row_number}')
            expected_cells = expected[1:]
            actual_cells = actual[1:]
            # Table 4's CSV has an explicit n column; the manuscript places n
            # in the scheme name. Validate it before matching q/pk/sig.
            if number == 4:
                degree = re.search(r'-(512|1024)', expected[0]).group(1)
                if not actual_cells or actual_cells[0] != degree:
                    raise ValueError(f'CSV degree changed: {path.name}, row {row_number}')
                actual_cells = actual_cells[1:]
            if len(actual_cells) != len(expected_cells):
                raise ValueError(f'CSV column count changed: {path.name}, row {row_number}')
            for column, (paper_value, generated_value) in enumerate(zip(expected_cells, actual_cells), 1):
                source = ('design_input' if column == 1 else 'recomputed') if origin == 'own_sizes' else origin
                check = {'row': row_label, 'column': column, 'provenance': source,
                         'paper_latex': paper_value, 'generated': generated_value,
                         'match': canonical_cell(paper_value) == canonical_cell(generated_value)}
                checks.append(check)
                report['cells_checked'] += 1
                report['provenance_counts'][source] = report['provenance_counts'].get(source, 0)+1
                if not check['match']:
                    report['mismatches'].append({'table': number, **check})
        report['tables'].append({'table': number, 'label': label, 'status': 'pass' if all(c['match'] for c in checks) else 'fail',
                                 'csv_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                 'cells_checked': len(checks), 'cells': checks})
        report['tables_checked'] += 1
    if report['mismatches']:
        report['status'] = 'fail'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper', type=Path, default=ROOT/'data/paper_tables.tex')
    parser.add_argument('--tables-dir', type=Path, default=ROOT/'results/tables')
    parser.add_argument('--report', type=Path, default=ROOT/'results/table_verification.json')
    args = parser.parse_args()
    try:
        report = verify_tables(args.paper, args.tables_dir)
    except (ValueError, OSError) as error:
        report = {'status': 'fail', 'error': str(error)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    for table in report.get('tables', []):
        print(f'Table {table["table"]}: {table["status"].upper()} ({table["cells_checked"]} data cells)')
    if report['status'] != 'pass':
        print(json.dumps(report.get('mismatches', report.get('error')), indent=2))
        raise SystemExit(1)
    print(f'PASS: all {report["tables_checked"]} tables, {report["cells_checked"]} displayed data cells match the paper table source.')
    print('Provenance:', json.dumps(report['provenance_counts']))


if __name__ == '__main__':
    main()
