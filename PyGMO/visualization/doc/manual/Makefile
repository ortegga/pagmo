

MANUAL	:= manual

PDFLATEX := pdflatex


TEX = latex
GLOSS = makeglossaries
BIBTEX = bibtex
DVIPS = dvips -t a4 -Ppdf -G0
PS2PDF = ps2pdf -dPDFSETTINGS=/prepress -dEmbedAllFonts=true
#PS2PDF = ps2pdf -dPDFSETTINGS=/prepress -dEmbedAllFonts=true -dEncodeGrayImages=false -dEncodeMonoImages=false -dEncodeColorImages=false



.PHONY: all clean


.SUFFIXES: .tex .aux .bbl .dvi .ps .pdf

.tex.dvi:
	$(TEX) $<
	$(GLOSS) $*
	$(BIBTEX) $*
	$(TEX) $<
	$(TEX) $<

.dvi.ps:
	$(DVIPS) -o $@ $< 

.ps.pdf:
	$(PS2PDF) $<


all: $(MANUAL).pdf


clean:
	$(RM) *.log *.aux *.acn *.acr *.alg *.bbl *.blg *.dvi *.glg *.glo *.gls *.ist *.out *.pdf *.toc *.xdy
