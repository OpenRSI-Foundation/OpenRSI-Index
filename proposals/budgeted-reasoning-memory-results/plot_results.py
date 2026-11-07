"""Recreate the figures from the adjacent public CSV and JSON (no service calls).

Requires Python 3, matplotlib and numpy. Run: python plot_results.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
rows = list(csv.DictReader((ROOT / 'evaluations.csv').open()))
summary = json.loads((ROOT / 'summary.json').read_text())
original = [r for r in rows if r['phase'] == 'original']
supplemental = [r for r in rows if r['phase'] == 'supplemental']
assert len(original) == 23 and len(supplemental) == 1
assert sum(r['status'] == 'scored' for r in original) == 22
assert summary['result_at_24_hours'] is None
x = np.array([int(r['attempt']) for r in original])

def values(key):
    return np.array([100 * float(r[key]) if r[key] else np.nan for r in original])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white', 'axes.titleweight': 'bold'})
BLUE, GREEN, ORANGE, GRAY = '#2466A6', '#167D57', '#C06012', '#697481'

def frame(title):
    fig, (ax, side) = plt.subplots(1, 2, figsize=(12.5, 5.8),
                                  gridspec_kw={'width_ratios': [5.1, 1.3]})
    fig.subplots_adjust(left=.075, right=.98, bottom=.23, top=.75, wspace=.22)
    fig.suptitle(title, x=.075, y=.96, ha='left', fontsize=18, fontweight='bold')
    fig.text(.075, .89, 'Memory run 2026-11-02 · stopped after 12 h 00 m 13 s of a planned 24 h',
             fontsize=12, color='#334155')
    fig.text(.075, .84, '22 scored + 1 unscored original attempts · no completed 24-hour result',
             fontsize=11, color=GRAY)
    for a in (ax, side):
        a.set_ylim(50, 85)
        a.set_yticks(np.arange(50, 86, 5))
        a.grid(axis='y', color='#E3E8EF', linewidth=.8)
        a.set_axisbelow(True)
    ax.set_xlim(.5, 23.7)
    ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])
    ax.set_xlabel('Original Judge attempt (submission order, not elapsed time)', labelpad=12)
    ax.set_ylabel('Accuracy (%)')
    ax.axvspan(20.7, 21.3, color='#F2D8C5', alpha=.65, zorder=0)
    ax.text(21, 51, 'Unscored', rotation=90, ha='center', va='bottom',
            fontsize=9, color='#9C4710')
    side.set_xlim(0, 1)
    side.set_xticks([.5], ['Reevaluate\nattempt 21'])
    side.set_title('Separate follow-up', fontsize=11, pad=14)
    side.tick_params(axis='y', labelleft=False)
    fig.text(.075, .08, 'Each scored attempt: the same 200 questions at each memory limit (400 generations).',
             fontsize=10, color=GRAY)
    fig.text(.075, .04, 'Unscored attempts are gaps, not zeros. The follow-up is not a resumed session or a 24-hour endpoint.',
             fontsize=10, color=GRAY)
    return fig, ax, side

fig, ax, side = frame('Mean accuracy by research iteration')
mean = values('mean_accuracy')
best = values('best_so_far')
ax.axhline(68.5, color=GRAY, linestyle='--', linewidth=1.3, label='Baseline 68.50%')
ax.plot(x, mean, color=BLUE, marker='o', markersize=4.5, linewidth=1.6,
        label='Attempt mean')
ax.step(x, best, color=GREEN, where='post', linewidth=2.3, label='Best so far')
ax.scatter([23], [78.75], color=GREEN, marker='*', s=140, zorder=5)
ax.annotate('Best 78.75%', (23, 78.75), xytext=(-10, 17), textcoords='offset points',
            ha='right', color=GREEN, fontweight='bold')
ax.annotate('54.00%', (9, 54), xytext=(0, -17), textcoords='offset points',
            ha='center', color=BLUE, fontsize=9)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.015), ncol=3, frameon=False,
          borderaxespad=0, fontsize=10)
side.axhline(68.5, color=GRAY, linestyle='--', linewidth=1.3)
side.scatter([.5], [76.75], color=ORANGE, marker='D', s=70, zorder=5)
side.annotate('76.75%', (.5, 76.75), xytext=(0, 14), textcoords='offset points',
              ha='center', color=ORANGE, fontweight='bold')
fig.savefig(ROOT / 'performance-by-iteration.png', dpi=160)
plt.close(fig)

fig, ax, side = frame('Accuracy at each memory-token limit')
for key, label, color, marker in [('accuracy_512', '512-token memory', BLUE, 'o'),
                                   ('accuracy_2048', '2,048-token memory', ORANGE, 's')]:
    ax.plot(x, values(key), color=color, marker=marker, markersize=4,
            linewidth=1.5, label=label)
    val = 100 * float(supplemental[0][key])
    side.scatter([.5], [val], color=color, marker=marker, s=65, zorder=5)
    side.annotate(f'{val:.2f}%', (.5, val), xytext=(0, 13 if key.endswith('2048') else -23),
                  textcoords='offset points', ha='center', color=color, fontweight='bold')
ax.annotate('Best-mean candidate\n77.00% / 80.50%', (23, 80.5), xytext=(-8, 12),
            textcoords='offset points', ha='right', fontsize=10)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.015), ncol=2, frameon=False,
          borderaxespad=0, fontsize=10)
fig.savefig(ROOT / 'performance-by-memory-budget.png', dpi=160)
plt.close(fig)
print('Created both figures from evaluations.csv and summary.json')
