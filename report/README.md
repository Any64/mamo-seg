# Informe (LaTeX)

**Informe final: [`main.pdf`](main.pdf)**, ya compilado a partir de `main.tex` y versionado junto al código.

- `main.tex`: fuente del informe final.
- `refs.bib`: referencias (BibTeX, estilo `plainnat`).
- `figures/`: copia de las figuras finales de `results/figures/`. Si se regeneran, hay que volver a copiarlas y
  recompilar.

## Recompilar

Requiere una distribución de LaTeX, por ejemplo MiKTeX o TeX Live, con `pdflatex` y `bibtex`.

```
cd report
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Se necesitan dos pasadas finales de `pdflatex` para resolver las citas y las referencias cruzadas. También sirve
`latexmk -pdf main.tex`, que hace lo mismo.

Los archivos auxiliares (`*.aux`, `*.log`, `*.bbl`...) están en `.gitignore`.
