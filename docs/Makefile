# Build LaTeX + PDF from the markdown numerics doc.
#
# Requirements:
#   - pandoc >= 2.x
#   - xelatex (texlive-xetex) + texlive-latex-recommended/extra
#   - DejaVu Serif/Sans/Sans Mono fonts (fontconfig discoverable)
#
# Usage:
#   make            # build purkinje_numerics.pdf
#   make tex        # build purkinje_numerics.tex only
#   make clean      # remove generated artifacts

SRC := purkinje_numerics.md
TEX := purkinje_numerics.tex
PDF := purkinje_numerics.pdf
HDR := purkinje_numerics_header.tex

PANDOC ?= pandoc
XELATEX ?= xelatex

PANDOC_FLAGS := \
  --from markdown+tex_math_dollars+raw_tex \
  --to latex \
  --standalone \
  --pdf-engine=xelatex \
  --include-in-header=$(HDR) \
  --variable=geometry:margin=2.5cm \
  --variable=fontsize:11pt \
  --variable=documentclass:article \
  --variable=colorlinks:true \
  --variable=linkcolor:blue \
  --variable=urlcolor:blue \
  --variable=toccolor:black \
  --variable=title:"Purkinje + PVJ + Regional Ionic Numerics" \
  --variable=author:"cardio-dance project" \
  --variable=date:"2026-05" \
  --toc \
  --number-sections

.PHONY: all tex pdf clean

all: pdf

tex: $(TEX)

pdf: $(PDF)

$(TEX): $(SRC) $(HDR)
	$(PANDOC) $(SRC) $(PANDOC_FLAGS) -o $(TEX)

# Two passes so TOC/cross-refs resolve.
$(PDF): $(TEX)
	$(XELATEX) -interaction=nonstopmode $(TEX) > /dev/null
	$(XELATEX) -interaction=nonstopmode $(TEX) > /dev/null

clean:
	rm -f $(TEX) $(PDF) \
		purkinje_numerics.aux purkinje_numerics.log \
		purkinje_numerics.out purkinje_numerics.toc
