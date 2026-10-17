"""Regenerate every figure and table in the paper, then the summary numbers.

    python src/run_all.py

Figures land in outputs/figures/ (fig01_... to fig14_...), tables in
outputs/tables/, and every number quoted in the text in
outputs/summary_numbers.json. Takes about a minute.
"""

import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SCRIPTS = [
    'fig01_03_distributions.py',
    'fig04_06_scale.py',
    'fig07_skew.py',
    'fig08_09_tail.py',
    'fig10_14_empirical.py',
    'summary_numbers.py',
]

t0 = time.time()
for name in SCRIPTS:
    print(f'\n=== {name}')
    runpy.run_path(os.path.join(HERE, name), run_name='__main__')
print(f'\nAll done in {time.time() - t0:.0f} s.')
