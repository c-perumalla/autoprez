"""
Autoprez Matcher — Moonshine Voice Edition (v2)

Uses Moonshine's streaming transcription API with event-driven callbacks.

Matching strategy (hybrid):
  - Accumulate completed Moonshine phrases + current in-progress line
  - Use sequential ordered word matching on the CURRENT lyric line's trigger words
  - Enforce a minimum time-on-line before allowing any transition
  - Track per-line transcripts and compute WER

Key insight from testing: Moonshine's transcript lags behind the audio by
several seconds. The matcher must be patient — use min_time_on_line to prevent
premature triggers from words that belong to the previous line still being
transcribed.
"""
import time
import sys
import os
import argparse
import urllib.request
import json
import re
import threading

from moonshine_voice.transcriber import Transcriber, TranscriptEventListener
from moonshine_voice.utils import load_wav_file
from moonshine_voice import get_model_for_language
from rapidfuzz import fuzz
from wer import compute_wer, format_wer_report, normalize_text


def word_similar(expected, actual, tolerance=2):
    """
    Check if two words are similar enough to count as a match.
    Handles Moonshine's common singing errors.
    """
    if expected == actual:
        return True

    # Prefix match (handles "believ" -> "believe", "dangerou" -> "dangerous")
    min_len = min(len(expected), len(actual))
    if min_len >= 3:
        if expected.startswith(actual[:min_len]) or actual.startswith(expected[:min_len]):
            return True

    # Quick exit: if length difference is too big, skip Levenshtein
    if abs(len(expected) - len(actual)) > tolerance:
        return False

    n, m = len(expected), len(actual)
    if n == 0 or m == 0:
        return False

    max_dist = tolerance if len(expected) <= 5 else tolerance + 1

    prev = list(range(m + 1))
    curr = [0] * (m + 1)

    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            if expected[i - 1] == actual[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[m] <= max_dist


def sequential_word_match(expected_words, transcript_words, tolerance=2):
    """
    Count how many expected words appear in order in the transcript.
    Returns (match_count, match_ratio).
    """
    if not expected_words:
        return 0, 0.0

    matched = 0
    search_start = 0

    for word in expected_words:
        for j in range(search_start, len(transcript_words)):
            if word_similar(word, transcript_words[j], tolerance):
                matched += 1
                search_start = j + 1
                break

    return matched, matched / len(expected_words)


# Map string names to fuzz functions (kept for hybrid approach)
FUZZ_METHODS = {
    'partial_ratio': fuzz.partial_ratio,
    'ratio': fuzz.ratio,
    'token_sort_ratio': fuzz.token_sort_ratio,
    'token_set_ratio': fuzz.token_set_ratio,
}


class LyricMatcher(TranscriptEventListener):
    """Listens to Moonshine transcript events and triggers slide transitions."""

    def __init__(self, lyrics, min_match_pct=60, trigger_words=4,
                 min_time_on_line=4.0, word_tolerance=2,
                 output_file=None, test_mode=False):
        self.lyrics = lyrics
        self.current_line_idx = 0
        self.start_time = time.time()
        self.last_transition_time = time.time()
        self.min_match_pct = min_match_pct
        self.trigger_words = trigger_words  # 0 = use entire line
        self.min_time_on_line = min_time_on_line
        self.word_tolerance = word_tolerance
        self.output_file = output_file
        self.test_mode = test_mode
        self.transition_log = []  # [(line_num, timestamp)]
        self.finished = False
        self.lock = threading.Lock()

        # Accumulated transcript since last transition
        self.completed_phrases = []  # completed Moonshine lines
        self.current_line_text = ""  # in-progress Moonshine line

        # In test mode, track audio position in seconds (not wall-clock)
        self.audio_position = 0.0
        self.last_transition_audio_pos = 0.0

        # WER tracking: per-line transcripts
        self.per_line_transcripts = {}

        # Log the first line at t=0
        self.transition_log.append((1, 0.0))

        self._print_current_target()

    def _print_current_target(self):
        if self.current_line_idx < len(self.lyrics):
            print(f"\n[TARGET] -> {self.lyrics[self.current_line_idx]}")

    def _get_expected_words(self):
        """Get the expected words for matching (respecting trigger_words setting)."""
        expected = self.lyrics[self.current_line_idx]
        words = normalize_text(expected).split()
        if self.trigger_words > 0 and len(words) > self.trigger_words:
            return words[-self.trigger_words:]
        return words

    def _get_accumulated_text(self):
        """Get the full accumulated text (completed phrases + current in-progress)."""
        parts = self.completed_phrases + ([self.current_line_text] if self.current_line_text else [])
        return " ".join(parts)

    def _check_match(self):
        """Check if the accumulated transcript matches the expected line using both methods."""
        with self.lock:
            if self.current_line_idx >= len(self.lyrics) or self.finished:
                return

            # Check min time on line
            if self.test_mode:
                time_on_line = self.audio_position - self.last_transition_audio_pos
            else:
                time_on_line = time.time() - self.last_transition_time

            if time_on_line < self.min_time_on_line:
                return

            accumulated = self._get_accumulated_text()
            if not accumulated.strip():
                return

            expected_words = self._get_expected_words()
            transcript_words = normalize_text(accumulated).split()

            # Method 1: Sequential ordered word matching
            matched, match_ratio = sequential_word_match(
                expected_words, transcript_words, self.word_tolerance
            )
            seq_pct = match_ratio * 100

            # Method 2: Fuzzy partial_ratio on the trigger phrase as backup
            trigger_phrase = " ".join(expected_words)
            fuzzy_score = fuzz.partial_ratio(trigger_phrase.lower(), accumulated.lower())

            # Trigger if EITHER method passes the threshold
            # Sequential matching is preferred (more precise), fuzzy is fallback
            triggered = False
            method_used = ""

            if seq_pct >= self.min_match_pct:
                triggered = True
                method_used = f"seq={seq_pct:.0f}%({matched}/{len(expected_words)})"
            elif fuzzy_score >= max(self.min_match_pct + 10, 80):
                # Fuzzy needs a higher bar to avoid false positives
                triggered = True
                method_used = f"fuzzy={fuzzy_score:.0f}%"

            if triggered:
                if self.test_mode:
                    elapsed = self.audio_position
                else:
                    elapsed = time.time() - self.start_time

                print(f"\n---- TRANSITION! {method_used} | Time: {time_on_line:.1f}s | t={elapsed:.1f}s ----")

                # Store transcript for WER
                self.per_line_transcripts[self.current_line_idx] = accumulated.strip()

                self.current_line_idx += 1
                self.last_transition_time = time.time()
                self.last_transition_audio_pos = self.audio_position

                # Reset buffers
                self.completed_phrases = []
                self.current_line_text = ""

                # Log the transition
                next_line_num = self.current_line_idx + 1  # 1-indexed
                self.transition_log.append((next_line_num, round(elapsed, 2)))

                next_line_text = ""
                if self.current_line_idx < len(self.lyrics):
                    next_line_text = self.lyrics[self.current_line_idx]
                    self._print_current_target()
                else:
                    print("Finished all lyrics!")
                    next_line_text = "END_OF_SONG"
                    self.finished = True

                # Notify the server (skip in test mode)
                if not self.test_mode:
                    try:
                        req = urllib.request.Request("http://127.0.0.1:9000/transition")
                        req.add_header('Content-Type', 'application/json; charset=utf-8')
                        payload = json.dumps({
                            "action": "transition_slide",
                            "next_line": next_line_text,
                            "line_index": self.current_line_idx,
                            "latency_metric": f"{time_on_line:.2f}s"
                        }).encode('utf-8')
                        req.add_header('Content-Length', str(len(payload)))
                        urllib.request.urlopen(req, payload)
                    except Exception as e:
                        print(f"  (server not running: {e})")

    def _send_transcript_update(self):
        if not self.test_mode:
            try:
                req = urllib.request.Request("http://127.0.0.1:9000/transition")
                req.add_header('Content-Type', 'application/json; charset=utf-8')
                payload = json.dumps({
                    "action": "transcript_update",
                    "text": self._get_accumulated_text()
                }).encode('utf-8')
                req.add_header('Content-Length', str(len(payload)))
                urllib.request.urlopen(req, payload)
            except Exception:
                pass

    def force_transition(self):
        """Force a slide transition bypassing all matching logic and time limits."""
        with self.lock:
            if self.current_line_idx >= len(self.lyrics) or self.finished:
                return
                
            if self.test_mode:
                elapsed = self.audio_position
            else:
                elapsed = time.time() - self.start_time

            print(f"\n---- MANUAL TRANSITION! | t={elapsed:.1f}s ----")

            # Store whatever transcript we had for WER
            accumulated = self._get_accumulated_text()
            self.per_line_transcripts[self.current_line_idx] = accumulated.strip()

            self.current_line_idx += 1
            self.last_transition_time = time.time()
            self.last_transition_audio_pos = self.audio_position

            # Reset buffers so we don't carry old text to the next line
            self.completed_phrases = []
            self.current_line_text = ""

            # Log the transition
            next_line_num = self.current_line_idx + 1  # 1-indexed
            self.transition_log.append((next_line_num, round(elapsed, 2)))

            next_line_text = ""
            if self.current_line_idx < len(self.lyrics):
                next_line_text = self.lyrics[self.current_line_idx]
                self._print_current_target()
            else:
                print("Finished all lyrics!")
                next_line_text = "END_OF_SONG"
                self.finished = True

            # Notify the server
            if not self.test_mode:
                try:
                    req = urllib.request.Request("http://127.0.0.1:9000/transition")
                    req.add_header('Content-Type', 'application/json; charset=utf-8')
                    payload = json.dumps({
                        "action": "transition_slide",
                        "next_line": next_line_text,
                        "line_index": self.current_line_idx,
                        "latency_metric": "manual"
                    }).encode('utf-8')
                    req.add_header('Content-Length', str(len(payload)))
                    urllib.request.urlopen(req, payload)
                except Exception as e:
                    print(f"  (server not running: {e})")

    def force_revert(self):
        """Force revert to the previous line."""
        with self.lock:
            if self.current_line_idx > 0:
                self.current_line_idx -= 1
                self.finished = False

                if self.test_mode:
                    elapsed = self.audio_position
                else:
                    elapsed = time.time() - self.start_time

                print(f"\n---- MANUAL REVERT! | t={elapsed:.1f}s ----")

                self.last_transition_time = time.time()
                self.last_transition_audio_pos = self.audio_position

                # Reset buffers
                self.completed_phrases = []
                self.current_line_text = ""

                # Remove last entry in log
                if self.transition_log and len(self.transition_log) > 1:
                    self.transition_log.pop()

                next_line_text = self.lyrics[self.current_line_idx]
                self._print_current_target()

                # Notify the server
                if not self.test_mode:
                    try:
                        req = urllib.request.Request("http://127.0.0.1:9000/transition")
                        req.add_header('Content-Type', 'application/json; charset=utf-8')
                        payload = json.dumps({
                            "action": "transition_slide",
                            "next_line": next_line_text,
                            "line_index": self.current_line_idx,
                            "latency_metric": "revert"
                        }).encode('utf-8')
                        req.add_header('Content-Length', str(len(payload)))
                        urllib.request.urlopen(req, payload)
                    except Exception as e:
                        print(f"  (server not running: {e})")

    def save_results(self):
        """Write transition log to the output file."""
        if self.output_file:
            with open(self.output_file, 'w') as f:
                for line_num, timestamp in self.transition_log:
                    f.write(f"{line_num} {timestamp}\n")
            print(f"\nResults written to {self.output_file}")

    def save_transcripts(self, transcript_file):
        """Save expected vs heard lyrics to a file for WER analysis."""
        per_line_results = []

        for idx in range(len(self.lyrics)):
            expected = self.lyrics[idx]
            heard = self.per_line_transcripts.get(idx, "(not reached)")
            wer_result = compute_wer(expected, heard) if idx in self.per_line_transcripts else None

            per_line_results.append({
                'line_num': idx + 1,
                'expected': expected,
                'heard': heard,
                'wer_result': wer_result,
            })

        # Write transcript comparison file
        with open(transcript_file, 'w') as f:
            f.write("LINE | EXPECTED | HEARD | WER\n")
            f.write("-" * 80 + "\n")
            for r in per_line_results:
                wer_str = f"{r['wer_result']['wer']*100:.1f}%" if r['wer_result'] else "N/A"
                f.write(f"{r['line_num']:>3} | {r['expected']}\n")
                f.write(f"    | {r['heard']}\n")
                f.write(f"    | WER: {wer_str}\n")
                f.write("\n")

        print(f"Transcript comparison saved to {transcript_file}")

        # Print WER report
        scored = [r for r in per_line_results if r['wer_result'] is not None]
        if scored:
            report = format_wer_report(scored)
            print(report)

        return per_line_results

    # --- Moonshine TranscriptEventListener callbacks ---

    def on_line_started(self, event):
        pass

    def on_line_text_changed(self, event):
        text = event.line.text.strip()
        if text:
            self.current_line_text = text
            sys.stdout.write(f"\r[LIVE] {text}                              ")
            sys.stdout.flush()
            self._send_transcript_update()
            self._check_match()

    def on_line_completed(self, event):
        text = event.line.text.strip()
        if text:
            self.completed_phrases.append(text)
            self.current_line_text = ""
            sys.stdout.write(f"\r[DONE] {text}                              \n")
            sys.stdout.flush()
            self._send_transcript_update()
            self._check_match()


def listen_to_stdin(matcher):
    """Background thread to listen for manual commands from the server."""
    for line in sys.stdin:
        cmd = line.strip()
        if cmd == "NEXT":
            matcher.force_transition()
        elif cmd == "PREV":
            matcher.force_revert()

def main():
    parser = argparse.ArgumentParser(description='Autoprez — Moonshine Voice v2')
    parser.add_argument('song_file', type=str, nargs='?', default='amazing_grace.txt',
                        help='Path to lyrics txt file')

    # Test mode flags
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode with a wav file instead of live mic')
    parser.add_argument('--audio-file', type=str, default=None,
                        help='Path to wav file (required for test mode)')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to write transition timestamps')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Path to write detailed log output')
    parser.add_argument('--transcript-file', type=str, default=None,
                        help='Path to write expected vs heard transcript comparison')

    # Tunable parameters
    parser.add_argument('--min-match-pct', type=int, default=60,
                        help='Minimum %% of expected words that must match in sequence (0-100)')
    parser.add_argument('--trigger-words', type=int, default=4,
                        help='Number of trailing words to require (0=entire line)')
    parser.add_argument('--min-time-on-line', type=float, default=4.0,
                        help='Minimum seconds before allowing a transition')
    parser.add_argument('--word-tolerance', type=int, default=2,
                        help='Max Levenshtein distance for word similarity')
    parser.add_argument('--update-interval', type=float, default=0.5,
                        help='Moonshine update interval in seconds')
    parser.add_argument('--mic-device', type=int, default=None,
                        help='Mic device index')

    args = parser.parse_args()

    # Redirect stdout to log file if specified
    original_stdout = sys.stdout
    if args.log_file:
        sys.stdout = open(args.log_file, 'w')

    # Load lyrics
    try:
        with open(args.song_file, "r") as f:
            lyrics = [line.strip() for line in f.readlines() if line.strip()]
    except Exception as e:
        print(f"Could not load {args.song_file}: {e}")
        return

    if not lyrics:
        print("No lyrics found!")
        return

    print(f"Loaded {len(lyrics)} lyric lines from {args.song_file}")
    print(f"Params: min_match_pct={args.min_match_pct}%, trigger_words={args.trigger_words}, "
          f"min_time={args.min_time_on_line}s, word_tol={args.word_tolerance}, "
          f"update_interval={args.update_interval}s")

    # Create the matcher listener
    matcher = LyricMatcher(
        lyrics,
        min_match_pct=args.min_match_pct,
        trigger_words=args.trigger_words,
        min_time_on_line=args.min_time_on_line,
        word_tolerance=args.word_tolerance,
        output_file=args.output_file,
        test_mode=args.test_mode,
    )

    # Get model path
    model_path, model_arch = get_model_for_language(wanted_language="en")

    if args.test_mode:
        # ---- TEST MODE: Feed wav file through Transcriber ----
        if not args.audio_file:
            print("ERROR: --audio-file is required in test mode")
            sys.exit(1)

        print(f"Test mode: processing {args.audio_file}")

        transcriber = Transcriber(
            model_path=model_path,
            model_arch=model_arch,
            update_interval=args.update_interval,
        )
        transcriber.add_listener(matcher)
        transcriber.start()

        # Load and stream the wav file in chunks
        audio_data, sample_rate = load_wav_file(args.audio_file)
        chunk_duration = 0.1  # 100ms chunks
        chunk_size = int(chunk_duration * sample_rate)

        matcher.start_time = time.time()
        matcher.last_transition_time = time.time()

        for i in range(0, len(audio_data), chunk_size):
            if matcher.finished:
                break
            chunk = audio_data[i: i + chunk_size]
            matcher.audio_position = (i + len(chunk)) / sample_rate
            transcriber.add_audio(chunk, sample_rate)

        transcriber.stop()
        transcriber.close()

        # Save results
        matcher.save_results()

        # Save transcript comparison
        song_base = os.path.splitext(os.path.basename(args.song_file))[0]
        transcript_file = args.transcript_file or f"transcript_{song_base}.txt"
        matcher.save_transcripts(transcript_file)

    else:
        # ---- LIVE MODE: Use MicTranscriber ----
        from moonshine_voice.mic_transcriber import MicTranscriber

        print("Starting Moonshine MicTranscriber (streaming)...")
        mic_kwargs = {
            "model_path": model_path,
            "model_arch": model_arch,
            "update_interval": args.update_interval,
        }
        if args.mic_device is not None:
            mic_kwargs["device_index"] = args.mic_device
            
        mic_transcriber = MicTranscriber(**mic_kwargs)
        mic_transcriber.add_listener(matcher)

        print("Listening... Press Ctrl+C to stop.\n")

        # Start stdin listener thread
        stdin_thread = threading.Thread(target=listen_to_stdin, args=(matcher,), daemon=True)
        stdin_thread.start()

        try:
            mic_transcriber.start()
            while not matcher.finished:
                time.sleep(0.1)
            print("\nSong complete!")
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            mic_transcriber.stop()
            mic_transcriber.close()
            matcher.save_results()

            song_base = os.path.splitext(os.path.basename(args.song_file))[0]
            transcript_file = args.transcript_file or f"transcript_{song_base}.txt"
            matcher.save_transcripts(transcript_file)


if __name__ == "__main__":
    main()
