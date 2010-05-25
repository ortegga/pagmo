

FEATURES	:= features

PDFLATEX := pdflatex


.PHONY: all clean


all: $(FEATURES).tex
	$(PDFLATEX) $(FEATURES).tex


clean:
	$(RM) $(FEATURES).aux $(FEATURES).log $(FEATURES).pdf
