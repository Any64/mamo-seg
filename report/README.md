# Informe (LaTeX)

- `main.tex`: estructura del informe con las secciones, las tablas de resultados finales y las figuras ya
  enlazadas. Los párrafos marcados con `% TODO` quedan por redactar.
- `refs.bib`: referencias (BibTeX).
- `figures/`: copia de las figuras finales de `results/figures/` (si se regeneran, volver a copiarlas).

Compilar (requiere una distribución de LaTeX, p. ej. MiKTeX o TeX Live):

```
cd report
latexmk -pdf main.tex        # o: pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Los archivos auxiliares (`*.aux`, `*.log`, `*.bbl`...) están en `.gitignore`.
