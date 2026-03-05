#!/bin/bash
cd /home/harshney/Desktop/PC-LiquidGAN
if command -v pdflatex &> /dev/null
then
    pdflatex final_report.tex
    # Run twice to resolve references/TOC
    pdflatex final_report.tex
    echo "Report compiled to final_report.pdf"
else
    echo "pdflatex is not installed. You can compile final_report.tex using Overleaf or another TeX editor."
fi
