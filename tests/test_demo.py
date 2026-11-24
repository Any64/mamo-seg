"""Prueba mínima del notebook de demostración (estructura; la ejecución completa necesita los pesos)."""
import ast
import sys
from pathlib import Path

import nbformat

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))


def test_demo_notebook_structure():
    from build_demo import build
    nb = build()
    nbformat.validate(nb)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    tree = ast.parse(code.replace("display(", "print("))      # el código es Python válido
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "mostrar_caso" in funcs
    assert "train_unet" not in code and "run_experiment" not in code   # no reentrena
    assert "unet_best.pt" in code and "shape_analysis" in code and "plot_confusion" in code


def test_demo_notebook_is_up_to_date():
    from build_demo import OUT, build
    saved = nbformat.read(OUT, as_version=4)
    assert [c.source for c in saved.cells] == [c.source for c in build().cells]
