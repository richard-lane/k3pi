#!/bin/bash
set -e

# Create DAG for the analysis rules in the snakefile
snakemake --forceall --dag | dot -Tpdf > analysis_dag.pdf
snakemake --forceall --rulegraph | dot -Tpdf > analysis_rules.pdf

