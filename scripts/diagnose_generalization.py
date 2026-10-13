"""Descriptive failure analysis of saved OOD predictions; never changes a policy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from melody_rl.data import read_rows
from melody_rl.music import parse_score, pitch_name


def describe(rows):
    candidates = [(row, candidate) for row in rows for candidate in row['candidates']]
    valid = [(r, c) for r, c in candidates if c['metrics']['valid']]
    counts = Counter(len(parse_score(c['text'])) for _, c in valid)
    prefixes = Counter(tuple(pitch_name(p) for bar in parse_score(c['text']) for p in bar)[:2] for _, c in valid)
    return {
        'tasks': len(rows), 'completions': len(candidates),
        'successful_completions': sum(c['metrics']['all_satisfied'] for _, c in candidates),
        'invalid_completions': len(candidates)-len(valid),
        'truncated_completions': sum(c['truncated'] for _, c in candidates),
        'valid_completions_by_actual_bar_count': {str(k):v for k,v in sorted(counts.items())},
        'wrong_bar_count_among_valid': sum(len(parse_score(c['text'])) != r['task']['bars'] for r,c in valid),
        'top_two_note_prefixes_among_valid': [{'prefix':list(k),'count':v} for k,v in prefixes.most_common(5)],
        'constraint_failures_among_valid': {
            key: sum(not c['metrics']['checks'][key] for _,c in valid)
            for key in ('bars','meter','key','range','final','leap')},
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result={'note':'Descriptive post-evaluation analysis, not a new confirmatory hypothesis or model-selection step. Failure counts can overlap.', 'splits':{}}
    for split in ('test_composition','test_length'):
        result['splits'][split]={}
        for policy in ('sft','grpo','diverse'):
            stages={}
            for mode in ('greedy','sampled'):
                rows=read_rows(args.run/f"{split.replace('_','-')}-{policy}-{mode}/predictions.jsonl")
                stages[mode]=describe(rows)
                stages[mode]['by_meter'] = {str(beats)+'/4': describe([r for r in rows if r['task']['beats']==beats]) for beats in (3,4) if any(r['task']['beats']==beats for r in rows)}
                if split=='test_length':
                    stages[mode]['by_key_meter_familiarity']={
                        'known_during_training':describe([r for r in rows if not (r['task']['key']=='G' and r['task']['beats']==4)]),
                        'withheld_G_major_4_4':describe([r for r in rows if r['task']['key']=='G' and r['task']['beats']==4])}
            result['splits'][split][policy]=stages
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(args.output.resolve())


if __name__=='__main__':
    main()
