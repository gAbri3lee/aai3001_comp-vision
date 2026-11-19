"""Report paired preregistered runs without selecting a favourable seed."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(summaries):
    if sorted(s['seed'] for s in summaries) != [0, 1, 2]:
        raise ValueError('Require each registered seed exactly once')
    if any(s['test_used'] for s in summaries):
        raise ValueError('Study must not use the test set')
    summaries = sorted(summaries, key=lambda s: s['seed'])
    names = list(summaries[0]['baseline']['per_class'])
    result = {'seeds': [0, 1, 2], 'test_used': False,
              'baseline_ap50': [s['baseline']['map50'] for s in summaries], 'arms': {}}
    for arm in ['control', 'reviewed', 'reviewed_scene_balanced']:
        metrics = [s['arms'][arm]['metrics'] for s in summaries]
        aps = [m['map50'] for m in metrics]
        paired = [m['map50'] - s['arms']['control']['metrics']['map50']
                  for m, s in zip(metrics, summaries)]
        baseline_delta = [m['map50'] - s['baseline']['map50']
                          for m, s in zip(metrics, summaries)]
        per_class = {}
        for name in names:
            per_class[name] = {key: statistics.mean(m['per_class'][name][key] for m in metrics)
                               for key in ['ap50', 'precision', 'recall']}
        # Both comparisons matter: beating a degraded control is insufficient.
        gain_pass = statistics.mean(paired) >= .02 and statistics.mean(baseline_delta) >= .02
        class_pass = all(
            per_class[name][key] >= statistics.mean(s['baseline']['per_class'][name][key] for s in summaries) - tolerance
            and per_class[name][key] >= statistics.mean(s['arms']['control']['metrics']['per_class'][name][key] for s in summaries) - tolerance
            for name in names for key, tolerance in [('recall', .03), ('precision', .02)])
        result['arms'][arm] = {'ap50_by_seed': aps, 'ap50_mean': statistics.mean(aps),
            'ap50_sample_sd': statistics.stdev(aps), 'paired_delta_vs_control': paired,
            'mean_delta_vs_baseline': statistics.mean(baseline_delta),
            'mean_delta_vs_control': statistics.mean(paired), 'per_class_mean': per_class,
            'accuracy_gate_passed': gain_pass and class_pass,
            'external_and_runtime_gate_completed': False,
            'automatically_deployed': False}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('summaries', type=Path, nargs=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plans, summaries = [], []
    for path in args.summaries:
        summary = json.loads(path.read_text())
        plan_text = (path.parent / 'registered-plan.json').read_text()
        # Registered Windows and Linux runs used different line endings.
        encodings = (plan_text, plan_text.replace('\n', '\r\n'))
        if summary['plan_sha256'] not in {hashlib.sha256(text.encode()).hexdigest() for text in encodings}:
            raise ValueError(f'Summary does not belong to the registered plan beside {path}')
        summaries.append(summary)
        plans.append(json.loads(plan_text))
    normalized = [{k: v for k, v in p.items() if k not in {'implementation_revision', 'study_script_sha256'}} for p in plans]
    if any(p != normalized[0] for p in normalized[1:]):
        raise ValueError('Unmatched data or training protocol across seeds')
    result = summarize(summaries)
    result['plan_hashes'] = [s['plan_sha256'] for s in sorted(summaries, key=lambda s: s['seed'])]
    result['orchestration_revision_only_may_differ'] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({arm: {'ap50_mean': value['ap50_mean'], 'accuracy_gate_passed': value['accuracy_gate_passed']}
                      for arm, value in result['arms'].items()}, indent=2))
