#!/bin/sh
var="paper"
pdflatex $var.tex && bibtex $var && pdflatex $var.tex && pdflatex $var.tex
