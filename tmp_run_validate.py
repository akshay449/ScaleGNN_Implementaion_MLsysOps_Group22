"""
Helper script: loads the notebook JSON, concatenates all code cells, executes them in a single process,
then calls `validate_design_notebook()` and prints results. This avoids relying on the Jupyter kernel.
"""
import json
import sys
import traceback
from pathlib import Path

NB_PATH = Path(__file__).parent / 'source_combined_notebook.ipynb'
if not NB_PATH.exists():
    print('Notebook not found at', NB_PATH)
    sys.exit(1)

nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
code_cells = [cell for cell in nb.get('cells', []) if cell.get('cell_type') == 'code']
combined = []
for cell in code_cells:
    src = cell.get('source', [])
    # join list of lines
    if isinstance(src, list):
        combined.append(''.join(src))
    else:
        combined.append(str(src))

full_code = '\n\n# ---- notebook cell boundary ----\n\n'.join(combined)

# Run the combined code in a single namespace
ns = {'__name__': '__main__'}
try:
    exec(full_code, ns)
except Exception as e:
    print('Error while executing notebook code cells:')
    traceback.print_exc()
    sys.exit(2)

# Now call validate_design_notebook if present
if 'validate_design_notebook' in ns:
    try:
        print('\nCalling validate_design_notebook()...')
        res = ns['validate_design_notebook'](device='auto')
        print('\nvalidate_design_notebook returned:')
        print(res)
    except Exception:
        print('Error while running validate_design_notebook():')
        traceback.print_exc()
        sys.exit(3)
else:
    print('validate_design_notebook not found in the executed notebook namespace')
    sys.exit(4)
