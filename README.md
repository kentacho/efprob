# EfProb

EfProb is a Python package for probability calculations. For more
background information on channel-based probability, see the
forthcoming book:

- Structured Probabilistic Reasoning http://www.cs.ru.nl/B.Jacobs/PAPERS/ProbabilisticReasoning.pdf

## Installation

Clone the repository and in the directory run:

```
pip install .
```

Add the option `--user` to install it in your home directory. You
might need to replace `pip` with `pip3`.

## Testing

The directory `tests` contains some (non-extensive) tests, which are supposed to be run with `pytest`.
Note: `test_hmm.py` will fail since `hmm` module is not included in the repository yet.
