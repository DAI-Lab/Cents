<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPI Shield](https://img.shields.io/pypi/v/Cents.svg)](https://pypi.python.org/pypi/cents-ml)
[![Downloads](https://pepy.tech/badge/cents-ml)](https://pepy.tech/project/cents-ml)
[![GitHub Actions Build Status](https://github.com/DAI-Lab/Cents/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/DAI-Lab/Cents/actions)


# Cents

A library for generative modeling and evaluation of synthetic household-level electricity load timeseries. This package is still under active development.

- [Documentation](https://dtail.gitbook.io/cents)

# Overview

Cents is a library built for generating *contextual time series data*. Cents supports several generative time series model architectures that can be used to train a time series data generator from scratch on a user-defined dataset. Additionally, Cents provides functionality for loading pre-trained model checkpoints that can be used to generate data instantly.

Cents was used to train the [Watts](https://huggingface.co/michaelfuest/watts) model series.

Feel free to look at our [tutorial notebooks](https://github.com/DAI-Lab/Cents/tree/main/tutorials) to get started.

# Install

## Requirements

**Cents** has been developed and tested on [Python 3.9]((https://www.python.org/downloads/)), [Python 3.10]((https://www.python.org/downloads/)) and [Python 3.11]((https://www.python.org/downloads/)).

We recommend using [Poetry](https://python-poetry.org/docs/) for dependency management. Make sure you have poetry installed before following these setup instructions.

Poetry will automatically create a virtual environment and install all dependencies:

```bash
poetry install
```

Once installed, activate the virtual environment:

```bash
poetry shell
```

This gives you a clean, reproducible setup for development.

## Install from PyPI

If you are only interested in using Cents functionality, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **Cents**:

```bash
pip install cents-ml
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).

## Datasets

If you want to reproduce the pretrained Watts model series from scratch, you will need to download the [PecanStreet DataPort dataset](https://www.pecanstreet.org/dataport/) and place it in an appropriate location specified in `cents/config/dataset/pecanstreet.yaml`. Specifically you will require the following files:

- 15minute_data_austin.csv
- 15minute_data_california.csv
- 15minute_data_newyork.csv
- metadata.csv

# What's next?

New models, new evaluation functionality and new datasets coming soon!
