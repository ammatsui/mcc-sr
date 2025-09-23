# mc-symbolic-regression

## Overview
Minimal Criterion Coevolution for Symbolic Regression.  
Equations and data generators evolve together and survive based on error thresholds.

## Quickstart

1. Clone and set up:
    ```
    git clone https://github.com/ammatsui/mcc-sr.git
    cd mc-sr
    python3 -m venv .venv && source .venv/bin/activate
    pip install pytest numpy
    ```
2. To run demo:
    ```
    python -m mc_sr.cli --data data/toy.csv
    ```
3. To run tests:
    ```
    pytest
    ```

## Minimal Working Example

Evolve equations and generators on toy data; CLI prints which PASS or FAIL.
