"""Shared: copy the v2 data pipeline (cells 1-41) with a location-independent ROOT and a local results/ folder."""
import json
SRC = "/Users/imtiaz/Documents/po1/progress_stage2/orginalProblemStatement_outage_extended_noaa_newgridv2.ipynb"
md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)}
code = lambda s: {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}
src = lambda c: "".join(c["source"])

def data_cells():
    nb = json.load(open(SRC))
    out = []
    for c in nb["cells"][1:42]:
        c = json.loads(json.dumps(c))
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
            s = src(c)
            if s.startswith("ROOT = ("):
                s = s.replace('ROOT = (\n    os.getcwd()\n    if os.path.isdir(os.path.join(os.getcwd(), "Data"))\n    else os.path.dirname(os.getcwd())\n)',
                              '# project folder = the first parent folder that contains Data/\nROOT = os.getcwd()\nwhile not os.path.isdir(os.path.join(ROOT, "Data")) and os.path.dirname(ROOT) != ROOT:\n    ROOT = os.path.dirname(ROOT)')
                s = s.replace('OUT_DIR = os.path.join(\n    ROOT, "progress_stage2", "results_outage_extended_noaa_nws_newgrid"\n)  # CSV / PNG outputs',
                              'OUT_DIR = os.path.join(os.getcwd(), "results")  # CSV / PNG outputs, next to this notebook')
                s = s.replace('TEST_START = pd.Timestamp("2022-01-01")  # test period = 2022-2024',
                              'TEST_START = pd.Timestamp("2022-01-01")  # used only for the summary printout in section 6')
                assert "while not os.path.isdir" in s and 'os.getcwd(), "results"' in s
                c["source"] = s.splitlines(keepends=True)
        else:
            c.pop("outputs", None)
        out.append(c)
    return out

def write(cells, dst, template_meta):
    nb = {"cells": cells, "metadata": template_meta, "nbformat": 4, "nbformat_minor": 5}
    nb["metadata"]["kernelspec"] = {"display_name": "tf_cpu", "language": "python", "name": "python3"}
    json.dump(nb, open(dst, "w"), indent=1, ensure_ascii=False)
    open(dst, "a").write("\n")
    print("wrote", dst, len(cells), "cells")

def meta():
    return json.load(open(SRC))["metadata"]
