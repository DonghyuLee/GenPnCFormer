from __future__ import annotations
from typing import List
from .data_types import Case

def generate_cases(min_cells:int=4, max_cells:int=7, include_double:bool=True) -> List[Case]:
    """
    Rules to reproduce the user's manual list pattern:
      - always include no-defect case per cells
      - single defect positions: 2..(cells-1)
      - double defects (if enabled): choose (i,j) in 2..(cells-1), i<j, and gap >= 2 (non-adjacent)
    """
    cases: List[Case] = []
    for cells in range(min_cells, max_cells + 1):
        cases.append(Case(cells=cells, defects=()))
        # single-defect
        for d in range(2, cells):
            cases.append(Case(cells=cells, defects=(d,)))
        # double-defect
        if include_double:
            pos = list(range(2, cells))
            for a in range(len(pos)):
                for b in range(a + 1, len(pos)):
                    i, j = pos[a], pos[b]
                    if (j - i) >= 2:
                        cases.append(Case(cells=cells, defects=(i, j)))
    cases.sort(key=lambda c: (c.cells, c.defects))
    return cases