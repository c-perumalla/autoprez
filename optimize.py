#!/usr/bin/env python3
"""
Autoprez Optimization Agent — Moonshine Voice v2

Autonomously tunes matcher parameters to achieve the best possible
lyric transition timing across all test songs.

Strategy: Greedy hill-climbing with random restarts.
  1. Score all songs with baseline parameters
  2. Perturb one parameter at a time
  3. Keep changes that improve total score
  4. After single-param sweep, try random combinations
  5. Stop when score = 0 or max iterations reached

Tunable parameters (v2 — sequential word matching):
  - min_match_pct: % of expected words that must match in sequence
  - trigger_words: how many trailing words form the matching target (0=all)
  - min_time_on_line: minimum seconds before allowing transition
  - word_tolerance: max Levenshtein distance for word similarity
  - update_interval: how often Moonshine updates its transcript (seconds)

Usage:
  python optimize.py [--max-iterations 30] [--base-dir .]
"""

import os
import sys
import json
import time
import copy
import random
import subprocess
import argparse
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("WARNING: psutil not installed. Memory checks disabled. Install with: pip install psutil")

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False
    print("WARNING: soundfile not installed. Cannot determine audio duration. Install with: pip install soundfile")


# =============================================
# CONFIGURATION
# =============================================

# Songs to test (basename without extension)
SONG_NAMES = ['amazing_grace', 'be_thou_my_vision', 'how_great_thou_art']

# Default / baseline parameters
BASELINE_PARAMS = {
    'min_match_pct': 70,
    'trigger_words': 0,       # 0 = use entire line
    'min_time_on_line': 4.0,
    'word_tolerance': 2,
    'update_interval': 0.5,
}

# Search space for each parameter
SEARCH_SPACE = {
    'min_match_pct': [50, 60, 70, 80],
    'trigger_words': [0, 3, 4, 5],
    'min_time_on_line': [2.0, 3.0, 4.0, 5.0],
    'word_tolerance': [1, 2, 3],
    'update_interval': [0.2, 0.3, 0.5],
}

# Resource limits
MAX_MEMORY_PERCENT = 80
COOLDOWN_SECONDS = 2
TIMEOUT_MULTIPLIER = 5  # timeout = song_duration * this


# =============================================
# HELPERS
# =============================================

def get_audio_duration(wav_path):
    """Get duration of a WAV file in seconds."""
    if HAS_SOUNDFILE:
        info = sf.info(wav_path)
        return info.duration
    else:
        return 180.0


def check_memory():
    """Return True if memory usage is below threshold."""
    if not HAS_PSUTIL:
        return True
    mem = psutil.virtual_memory()
    if mem.percent > MAX_MEMORY_PERCENT:
        print(f"  ⚠️  Memory usage at {mem.percent:.1f}% (threshold: {MAX_MEMORY_PERCENT}%). Skipping run.")
        return False
    return True


def build_matcher_command(song_file, audio_file, output_file, log_file, params, python_exe, iteration):
    """Build the command line for running matcher in test mode."""
    song_base = os.path.splitext(os.path.basename(song_file))[0]
    transcript_file = os.path.join(os.path.dirname(log_file), f'{song_base}_transcript_run{iteration}.txt')

    cmd = [
        python_exe, 'matcher.py', song_file,
        '--test-mode',
        '--audio-file', audio_file,
        '--output-file', output_file,
        '--log-file', log_file,
        '--transcript-file', transcript_file,
        '--min-match-pct', str(params['min_match_pct']),
        '--trigger-words', str(params['trigger_words']),
        '--min-time-on-line', str(params['min_time_on_line']),
        '--word-tolerance', str(params['word_tolerance']),
        '--update-interval', str(params['update_interval']),
    ]
    return cmd


def run_matcher(song_name, params, iteration, base_dir, python_exe):
    """
    Run matcher for a single song and return the score.
    
    Returns:
        (score, per_line_details) or (None, None) if failed
    """
    song_file = os.path.join(base_dir, f'{song_name}.txt')
    audio_file = os.path.join(base_dir, f'{song_name}.wav')
    correct_file = os.path.join(base_dir, f'{song_name}_correct_answers.txt')
    output_file = os.path.join(base_dir, f'test_answers_{song_name}.txt')
    log_dir = os.path.join(base_dir, 'logs_v2')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{song_name}_run{iteration}.log')
    
    for f, desc in [(song_file, 'lyrics'), (audio_file, 'audio'), (correct_file, 'correct answers')]:
        if not os.path.exists(f):
            print(f"  ❌ Missing {desc} file: {f}")
            return None, None
    
    duration = get_audio_duration(audio_file)
    timeout = max(duration * TIMEOUT_MULTIPLIER, 60)
    
    cmd = build_matcher_command(song_file, audio_file, output_file, log_file, params, python_exe, iteration)
    
    print(f"  🎵 Running matcher for {song_name}...")
    print(f"     Params: match_pct={params['min_match_pct']}%, "
          f"trigger_words={params['trigger_words']}, "
          f"min_time={params['min_time_on_line']}s, "
          f"word_tol={params['word_tolerance']}, "
          f"update={params['update_interval']}s")
    
    try:
        if sys.platform == 'darwin':
            cmd = ['nice', '-n', '10'] + cmd
        
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=base_dir
        )
        run_time = time.time() - start_time
        
        if result.returncode != 0:
            print(f"  ❌ Matcher failed for {song_name}: {result.stderr[:200]}")
            return None, None
        
        print(f"     Completed in {run_time:.1f}s")
        
    except subprocess.TimeoutExpired:
        print(f"  ⏰ Matcher timed out for {song_name} (timeout={timeout:.0f}s)")
        return None, None
    except Exception as e:
        print(f"  ❌ Error running matcher for {song_name}: {e}")
        return None, None
    
    if not os.path.exists(output_file):
        print(f"  ❌ No output file produced for {song_name}")
        return None, None
    
    try:
        eval_cmd = [python_exe, 'evaluate.py', correct_file, output_file, '--json']
        eval_result = subprocess.run(
            eval_cmd,
            capture_output=True,
            text=True,
            timeout=10,
            cwd=base_dir
        )
        
        output_lines = eval_result.stdout.strip().split('\n')
        json_lines = []
        for line in output_lines:
            if line.startswith('SCORE:'):
                break
            json_lines.append(line)
        
        eval_data = json.loads('\n'.join(json_lines))
        score = eval_data['total_score']
        per_line = eval_data.get('per_line', [])
        
        print(f"     Score: {score:.1f} ({eval_data['lines_perfect']}/{eval_data['lines_scored']} perfect)")
        return score, per_line
        
    except Exception as e:
        print(f"  ❌ Error running evaluation for {song_name}: {e}")
        return None, None


def evaluate_params(params, iteration, base_dir, python_exe, songs=None):
    """Run all songs with given params and return total score + details."""
    if songs is None:
        songs = SONG_NAMES
    
    total_score = 0.0
    per_song_scores = {}
    per_song_details = {}
    
    for song_name in songs:
        if not check_memory():
            print("  ⚠️  Aborting iteration due to high memory usage")
            return None, None, None
        
        score, details = run_matcher(song_name, params, iteration, base_dir, python_exe)
        
        if score is None:
            print(f"  ⚠️  Failed to score {song_name}, treating as worst case")
            score = -100.0
            details = []
        
        total_score += score
        per_song_scores[song_name] = score
        per_song_details[song_name] = details
        
        time.sleep(COOLDOWN_SECONDS)
    
    return total_score, per_song_scores, per_song_details


def save_log(log_data, base_dir):
    """Save optimization log to disk."""
    log_path = os.path.join(base_dir, 'optimization_log_v2.json')
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def save_best_params(best, base_dir):
    """Save best parameters to disk."""
    best_path = os.path.join(base_dir, 'best_params_v2.json')
    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2)


# =============================================
# OPTIMIZATION STRATEGIES
# =============================================

def single_param_sweep(current_params, current_score, iteration_start, max_iterations, 
                       base_dir, python_exe, log_data, best_info):
    """Try changing one parameter at a time."""
    best_params = copy.deepcopy(current_params)
    best_score = current_score
    iteration = iteration_start
    
    param_order = list(SEARCH_SPACE.keys())
    random.shuffle(param_order)
    
    for param_name in param_order:
        if iteration >= max_iterations or best_score == 0:
            break
            
        values = SEARCH_SPACE[param_name]
        current_val = best_params[param_name]
        
        for val in values:
            if val == current_val or iteration >= max_iterations or best_score == 0:
                continue
            
            iteration += 1
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}: Trying {param_name} = {val} (was {current_val})")
            print(f"{'='*60}")
            
            test_params = copy.deepcopy(best_params)
            test_params[param_name] = val
            
            total_score, per_song_scores, per_song_details = evaluate_params(
                test_params, iteration, base_dir, python_exe
            )
            
            if total_score is None:
                print(f"  ⚠️  Iteration {iteration} failed, skipping")
                continue
            
            entry = {
                'iteration': iteration,
                'params': copy.deepcopy(test_params),
                'scores': per_song_scores,
                'per_line_details': per_song_details,
                'total_score': total_score,
                'changed_param': param_name,
                'changed_from': current_val,
                'changed_to': val,
                'improved': total_score > best_score,
                'timestamp': datetime.now().isoformat()
            }
            log_data.append(entry)
            save_log(log_data, base_dir)
            
            if total_score > best_score:
                print(f"\n  ✅ IMPROVED! {best_score:.1f} -> {total_score:.1f} ({param_name}: {current_val} -> {val})")
                best_score = total_score
                best_params = copy.deepcopy(test_params)
                current_val = val
                
                best_info.update({
                    'params': copy.deepcopy(best_params),
                    'total_score': best_score,
                    'per_song_scores': per_song_scores,
                    'iteration': iteration,
                    'timestamp': datetime.now().isoformat()
                })
                save_best_params(best_info, base_dir)
            else:
                print(f"  ❌ No improvement ({total_score:.1f} vs best {best_score:.1f})")
    
    return best_params, best_score, iteration


def random_combination_search(current_params, current_score, iteration_start, max_iterations,
                               base_dir, python_exe, log_data, best_info, n_combos=5):
    """Try random parameter combinations."""
    best_params = copy.deepcopy(current_params)
    best_score = current_score
    iteration = iteration_start
    
    for combo in range(n_combos):
        if iteration >= max_iterations or best_score == 0:
            break
        
        iteration += 1
        
        test_params = copy.deepcopy(best_params)
        n_changes = random.randint(2, 3)
        params_to_change = random.sample(list(SEARCH_SPACE.keys()), n_changes)
        
        changes = {}
        for param_name in params_to_change:
            new_val = random.choice(SEARCH_SPACE[param_name])
            changes[param_name] = (test_params[param_name], new_val)
            test_params[param_name] = new_val
        
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration}: Random combo - {changes}")
        print(f"{'='*60}")
        
        total_score, per_song_scores, per_song_details = evaluate_params(
            test_params, iteration, base_dir, python_exe
        )
        
        if total_score is None:
            continue
        
        entry = {
            'iteration': iteration,
            'params': copy.deepcopy(test_params),
            'scores': per_song_scores,
            'per_line_details': per_song_details,
            'total_score': total_score,
            'strategy': 'random_combo',
            'changes': {k: {'from': v[0], 'to': v[1]} for k, v in changes.items()},
            'improved': total_score > best_score,
            'timestamp': datetime.now().isoformat()
        }
        log_data.append(entry)
        save_log(log_data, base_dir)
        
        if total_score > best_score:
            print(f"\n  ✅ IMPROVED! {best_score:.1f} -> {total_score:.1f}")
            best_score = total_score
            best_params = copy.deepcopy(test_params)
            
            best_info.update({
                'params': copy.deepcopy(best_params),
                'total_score': best_score,
                'per_song_scores': per_song_scores,
                'iteration': iteration,
                'timestamp': datetime.now().isoformat()
            })
            save_best_params(best_info, base_dir)
        else:
            print(f"  ❌ No improvement ({total_score:.1f} vs best {best_score:.1f})")
    
    return best_params, best_score, iteration


# =============================================
# MAIN
# =============================================

def main():
    parser = argparse.ArgumentParser(description='Autoprez Optimization Agent (Moonshine v2)')
    parser.add_argument('--max-iterations', type=int, default=30, help='Maximum optimization iterations')
    parser.add_argument('--base-dir', type=str, default='.', help='Base directory containing songs and matcher')
    parser.add_argument('--python', type=str, default=sys.executable, help='Python executable to use')
    parser.add_argument('--resume', action='store_true', help='Resume from previous best_params.json')
    args = parser.parse_args()
    
    base_dir = os.path.abspath(args.base_dir)
    python_exe = args.python
    
    print("=" * 60)
    print("🔧 AUTOPREZ OPTIMIZATION AGENT (Moonshine v2)")
    print("=" * 60)
    print(f"  Base dir: {base_dir}")
    print(f"  Python: {python_exe}")
    print(f"  Max iterations: {args.max_iterations}")
    print(f"  Songs: {SONG_NAMES}")
    print(f"  Search space:")
    for param, values in SEARCH_SPACE.items():
        print(f"    {param}: {values}")
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        print(f"  Memory: {mem.percent:.1f}% used ({mem.available / 1e9:.1f}GB available)")
    print()
    
    missing = []
    for song in SONG_NAMES:
        for ext, desc in [('.txt', 'lyrics'), ('.wav', 'audio'), ('_correct_answers.txt', 'answers')]:
            fpath = os.path.join(base_dir, f'{song}{ext}')
            if not os.path.exists(fpath):
                missing.append(f"{song}{ext} ({desc})")
    
    if missing:
        print("❌ Missing required files:")
        for m in missing:
            print(f"   - {m}")
        sys.exit(1)
    
    # Initialize or resume
    log_data = []
    if args.resume and os.path.exists(os.path.join(base_dir, 'best_params_v2.json')):
        with open(os.path.join(base_dir, 'best_params_v2.json'), 'r') as f:
            best_info = json.load(f)
        saved_keys = set(best_info.get('params', {}).keys())
        expected_keys = set(BASELINE_PARAMS.keys())
        if saved_keys == expected_keys:
            current_params = best_info['params']
            print(f"📂 Resuming from best_params_v2.json (score: {best_info['total_score']:.1f})")
        else:
            print(f"⚠️  Saved params have different keys, starting fresh")
            current_params = copy.deepcopy(BASELINE_PARAMS)
            best_info = {}
        
        log_path = os.path.join(base_dir, 'optimization_log_v2.json')
        if os.path.exists(log_path):
            try:
                with open(log_path, 'r') as f:
                    log_data = json.load(f)
            except Exception:
                log_data = []
    else:
        current_params = copy.deepcopy(BASELINE_PARAMS)
        best_info = {}
    
    # Phase 1: Baseline
    print(f"\n{'='*60}")
    print("BASELINE EVALUATION")
    print(f"{'='*60}")
    
    total_score, per_song_scores, per_song_details = evaluate_params(
        current_params, 0, base_dir, python_exe
    )
    
    if total_score is None:
        print("❌ Baseline evaluation failed completely.")
        sys.exit(1)
    
    print(f"\n📊 Baseline total score: {total_score:.1f}")
    for song, score in per_song_scores.items():
        print(f"   {song}: {score:.1f}")
    
    baseline_entry = {
        'iteration': 0,
        'params': copy.deepcopy(current_params),
        'scores': per_song_scores,
        'per_line_details': per_song_details,
        'total_score': total_score,
        'strategy': 'baseline',
        'timestamp': datetime.now().isoformat()
    }
    log_data.append(baseline_entry)
    save_log(log_data, base_dir)
    
    best_score = total_score
    best_params = copy.deepcopy(current_params)
    
    if not best_info or total_score > best_info.get('total_score', -999):
        best_info = {
            'params': copy.deepcopy(best_params),
            'total_score': best_score,
            'per_song_scores': per_song_scores,
            'iteration': 0,
            'timestamp': datetime.now().isoformat()
        }
    save_best_params(best_info, base_dir)
    
    if best_score == 0:
        print("\n🎉 Perfect score achieved with baseline parameters!")
        return
    
    # Phase 2: Single parameter sweep
    print(f"\n{'='*60}")
    print("PHASE 2: Single Parameter Sweep")
    print(f"{'='*60}")
    
    best_params, best_score, iteration = single_param_sweep(
        best_params, best_score, 0, args.max_iterations,
        base_dir, python_exe, log_data, best_info
    )
    
    if best_score == 0:
        print("\n🎉 Perfect score achieved!")
        print_final_report(best_info, base_dir)
        return
    
    # Phase 3: Random combination search
    remaining_iterations = args.max_iterations - iteration
    if remaining_iterations > 0:
        print(f"\n{'='*60}")
        print(f"PHASE 3: Random Combination Search ({remaining_iterations} iterations remaining)")
        print(f"{'='*60}")
        
        best_params, best_score, iteration = random_combination_search(
            best_params, best_score, iteration, args.max_iterations,
            base_dir, python_exe, log_data, best_info,
            n_combos=remaining_iterations
        )
    
    print_final_report(best_info, base_dir)


def print_final_report(best_info, base_dir):
    """Print final optimization report."""
    print(f"\n{'='*60}")
    print("🏆 OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"\n  Best total score: {best_info['total_score']:.1f}")
    print(f"  Found at iteration: {best_info['iteration']}")
    print(f"\n  Best parameters:")
    for k, v in best_info['params'].items():
        default = BASELINE_PARAMS.get(k)
        changed = " ← CHANGED" if v != default else ""
        print(f"    {k}: {v}{changed}")
    
    if 'per_song_scores' in best_info:
        print(f"\n  Per-song scores:")
        for song, score in best_info['per_song_scores'].items():
            print(f"    {song}: {score:.1f}")
    
    print(f"\n  Results saved to:")
    print(f"    {os.path.join(base_dir, 'best_params_v2.json')}")
    print(f"    {os.path.join(base_dir, 'optimization_log_v2.json')}")
    print()


if __name__ == '__main__':
    main()
