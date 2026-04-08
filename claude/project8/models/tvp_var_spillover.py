"""
Backwards-compatible import path.

The Streamlit app now calls R via `Rscript` through `project8/tvp_var_spillover.py`
to avoid rpy2 + DLL issues on Windows.
"""

from tvp_var_spillover import get_tvp_var_spillover  # re-export