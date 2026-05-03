"""
Word Error Rate (WER) Calculation Module

Computes WER using standard edit distance between reference and hypothesis word sequences.
WER = (Substitutions + Insertions + Deletions) / Reference Word Count
"""

import re


def normalize_text(text):
    """Normalize text for WER comparison: lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", '', text)  # Keep apostrophes for contractions
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(reference, hypothesis):
    """
    Compute Word Error Rate between reference and hypothesis.
    
    Args:
        reference: The expected/correct text
        hypothesis: The transcribed/predicted text
    
    Returns:
        dict with keys: wer, substitutions, insertions, deletions, 
                        ref_words, hyp_words, ref_count, hyp_count
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    if not ref_words:
        return {
            'wer': 0.0 if not hyp_words else 1.0,
            'substitutions': 0,
            'insertions': len(hyp_words),
            'deletions': 0,
            'ref_words': ref_words,
            'hyp_words': hyp_words,
            'ref_count': 0,
            'hyp_count': len(hyp_words),
        }
    
    n = len(ref_words)
    m = len(hyp_words)
    
    # DP table for edit distance
    d = [[0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        d[i][0] = i  # deletions
    for j in range(m + 1):
        d[0][j] = j  # insertions
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )
    
    # Backtrace to count S, I, D
    i, j = n, m
    substitutions = 0
    insertions = 0
    deletions = 0
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i - 1] == hyp_words[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif j > 0 and d[i][j] == d[i][j - 1] + 1:
            insertions += 1
            j -= 1
        else:
            deletions += 1
            i -= 1
    
    wer = (substitutions + insertions + deletions) / n
    
    return {
        'wer': round(wer, 4),
        'substitutions': substitutions,
        'insertions': insertions,
        'deletions': deletions,
        'ref_words': ref_words,
        'hyp_words': hyp_words,
        'ref_count': n,
        'hyp_count': m,
    }


def format_wer_report(per_line_results):
    """
    Format a human-readable WER report.
    
    Args:
        per_line_results: list of dicts, each with:
            'line_num', 'expected', 'heard', 'wer_result'
    
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("WORD ERROR RATE REPORT")
    lines.append("=" * 60)
    
    total_ref = 0
    total_errors = 0
    
    for r in per_line_results:
        wer = r['wer_result']
        total_ref += wer['ref_count']
        total_errors += wer['substitutions'] + wer['insertions'] + wer['deletions']
        
        lines.append(f"\nLine {r['line_num']:>2}: \"{r['expected']}\"")
        lines.append(f"  Heard: \"{r['heard']}\"")
        lines.append(f"  WER: {wer['wer']*100:.1f}% "
                     f"(S={wer['substitutions']} I={wer['insertions']} D={wer['deletions']})")
    
    overall = total_errors / total_ref if total_ref > 0 else 0
    lines.append(f"\n{'='*60}")
    lines.append(f"Overall WER: {overall*100:.1f}% "
                 f"({total_errors} errors / {total_ref} reference words)")
    lines.append("")
    
    return "\n".join(lines)
