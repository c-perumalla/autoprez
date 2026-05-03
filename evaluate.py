#!/usr/bin/env python3
"""
Autoprez Evaluation Script

Compares correct_answers.txt against test_answers.txt and produces a score.

Scoring rules:
  - Line 1 is excluded (it's the initial state at t=0, not a real transition).
  - For each subsequent line:
      early by ≤1s:  0 points (perfect)
      early by >1s: -1.0 points
      late (any):   -1.5 points
      missing:      -1.5 points

Usage:
  python evaluate.py correct_answers.txt test_answers.txt [--json]
"""

import sys
import argparse
import json


def parse_answers(filepath):
    """Parse an answers file into a dict of {line_num: timestamp}."""
    answers = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                line_num = int(parts[0])
                timestamp = float(parts[1])
                answers[line_num] = timestamp
    return answers


def evaluate(correct_path, test_path):
    """
    Evaluate test answers against correct answers.
    
    Returns:
        dict with keys: total_score, max_score, per_line, summary
    """
    correct = parse_answers(correct_path)
    test = parse_answers(test_path)
    
    per_line = []
    total_score = 0.0
    
    # Get all line numbers except line 1 (excluded from scoring)
    all_lines = sorted([ln for ln in correct.keys() if ln != 1])
    
    for line_num in all_lines:
        expected = correct[line_num]
        
        if line_num not in test:
            # Line is missing from test answers
            penalty = -1.5
            total_score += penalty
            per_line.append({
                'line': line_num,
                'expected': expected,
                'actual': None,
                'delta': None,
                'penalty': penalty,
                'reason': 'missing'
            })
            continue
        
        actual = test[line_num]
        delta = actual - expected  # positive = late, negative = early
        
        if delta > 0:
            # Late — any amount
            penalty = -1.5
            reason = 'late'
        elif delta < 0 and abs(delta) > 1.0:
            # Early by more than 1 second
            penalty = -1.0
            reason = 'too_early'
        else:
            # Within [-1.0, 0.0] or exactly on time
            penalty = 0.0
            reason = 'perfect'
        
        total_score += penalty
        per_line.append({
            'line': line_num,
            'expected': round(expected, 2),
            'actual': round(actual, 2),
            'delta': round(delta, 2),
            'penalty': penalty,
            'reason': reason
        })
    
    return {
        'total_score': total_score,
        'max_score': 0.0,
        'lines_scored': len(all_lines),
        'lines_perfect': sum(1 for p in per_line if p['penalty'] == 0),
        'lines_early': sum(1 for p in per_line if p['reason'] == 'too_early'),
        'lines_late': sum(1 for p in per_line if p['reason'] == 'late'),
        'lines_missing': sum(1 for p in per_line if p['reason'] == 'missing'),
        'per_line': per_line
    }


def print_report(result):
    """Print a human-readable evaluation report."""
    print("=" * 60)
    print("AUTOPREZ EVALUATION REPORT")
    print("=" * 60)
    print()
    
    print(f"{'Line':<6} {'Expected':>10} {'Actual':>10} {'Delta':>8} {'Penalty':>8} {'Status'}")
    print("-" * 60)
    
    for p in result['per_line']:
        expected = f"{p['expected']:.1f}s"
        actual = f"{p['actual']:.1f}s" if p['actual'] is not None else "MISSING"
        delta = f"{p['delta']:+.2f}s" if p['delta'] is not None else "—"
        penalty = f"{p['penalty']:+.1f}" if p['penalty'] != 0 else "0.0"
        
        status_icons = {
            'perfect': '✅',
            'too_early': '⏪ early',
            'late': '⏩ late',
            'missing': '❌ miss'
        }
        status = status_icons.get(p['reason'], p['reason'])
        
        print(f"  {p['line']:<4} {expected:>10} {actual:>10} {delta:>8} {penalty:>8}  {status}")
    
    print("-" * 60)
    print(f"\n  Total Score: {result['total_score']:.1f} / {result['max_score']:.1f}")
    print(f"  Perfect: {result['lines_perfect']}/{result['lines_scored']}  |  "
          f"Early: {result['lines_early']}  |  Late: {result['lines_late']}  |  "
          f"Missing: {result['lines_missing']}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Evaluate autoprez matcher accuracy')
    parser.add_argument('correct_answers', type=str, help='Path to correct answers file')
    parser.add_argument('test_answers', type=str, help='Path to test answers file')
    parser.add_argument('--json', action='store_true', help='Output results as JSON')
    args = parser.parse_args()
    
    result = evaluate(args.correct_answers, args.test_answers)
    
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_report(result)
    
    # Always print the score line for easy parsing by the optimizer
    print(f"SCORE: {result['total_score']:.1f}")
    
    return result['total_score']


if __name__ == '__main__':
    score = main()
    sys.exit(0 if score == 0 else 1)
