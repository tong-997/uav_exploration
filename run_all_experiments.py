#!/usr/bin/env python3
"""
第3.6节综合实验 — 一键运行
============================
python run_all_experiments.py              # 运行全部6个实验
python run_all_experiments.py --exp 5      # 只运行实验5
python run_all_experiments.py --exp 1 6    # 运行实验1和6
"""
import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description='Section 3.6 Comprehensive Experiments')
    parser.add_argument('--exp', type=str, nargs='+', default=None,
                        help='Experiment IDs to run (1-6, 4b). Default: all.')
    args = parser.parse_args()

    def _parse_exp_id(s):
        try:
            return int(s)
        except ValueError:
            return s.lower()

    exp_list = [_parse_exp_id(e) for e in args.exp] if args.exp else [1, 2, 3, 4, 5, 6]

    runners = {
        1: ('Exp1: UAV Count & Speed', 'experiments.exp1_param_variation', 'run_exp1'),
        2: ('Exp2: Strategy Comparison', 'experiments.exp2_strategy_comparison', 'run_exp2'),
        3: ('Exp3: Depth Input Modes', 'experiments.exp3_depth_modes', 'run_exp3'),
        4: ('Exp4: Target Detection', 'experiments.exp4_target_detection', 'run_exp4'),
        5: ('Exp5: Safety Analysis', 'experiments.exp5_safety_analysis', 'run_exp5'),
        6: ('Exp6: Multi-Seed Robustness', 'experiments.exp6_robustness', 'run_exp6'),
        '4b': ('Exp4B: Tracking & Fusion', 'experiments.exp4b_tracking_fusion', 'run_exp4b'),
    }

    print('=' * 60)
    print('  Section 3.6 Comprehensive Experiments')
    print(f'  Running: {exp_list}')
    print('=' * 60)

    total_start = time.time()

    for exp_num in sorted(exp_list, key=lambda x: (0, x) if isinstance(x, int) else (1, x)):
        if exp_num not in runners:
            print(f'\n  [SKIP] Unknown experiment: {exp_num}')
            continue

        title, module_name, func_name = runners[exp_num]
        print(f'\n{"=" * 60}')
        print(f'  {title}')
        print(f'{"=" * 60}')

        t0 = time.time()
        try:
            module = __import__(module_name, fromlist=[func_name])
            run_func = getattr(module, func_name)
            run_func()
            elapsed = time.time() - t0
            print(f'  [{title}] Done in {elapsed:.1f}s')
        except Exception as e:
            elapsed = time.time() - t0
            print(f'  [{title}] FAILED after {elapsed:.1f}s: {e}')
            import traceback
            traceback.print_exc()

    total_elapsed = time.time() - total_start
    print(f'\n{"=" * 60}')
    print(f'  All experiments completed in {total_elapsed:.1f}s')
    print(f'  Output → output/exp*/')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
