<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“DAI-Lab” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

<!-- Uncomment these lines after releasing the package to PyPI for version and downloads badges -->
<!--[![PyPI Shield](https://img.shields.io/pypi/v/EnData.svg)](https://pypi.python.org/pypi/EnData)-->
<!--[![Downloads](https://pepy.tech/badge/EnData)](https://pepy.tech/project/EnData)-->
[![Github Actions Shield](https://img.shields.io/github/workflow/status/michael-fuest/EnData/Run%20Tests)](https://github.com/michael-fuest/EnData/actions)
[![Coverage Status](https://codecov.io/gh/michael-fuest/EnData/branch/master/graph/badge.svg)](https://codecov.io/gh/michael-fuest/EnData)



# EnData

A library for generative modeling and evaluation of synthetic household-level electricity load timeseries. This package is still under active development.

- Documentation: (tbd)

# Overview

EnData is a library built for generating *synthetic household-level electric load and generation timeseries*. EnData supports a variety of generative time series models that can be used to train a time series data generator from scratch on a user-defined dataset. Additionally, EnData provides functionality for loading pre-trained model checkpoints that can be used to generate data instantly. Trained models can be evaluated using a series of metrics and visualizations also implemented here.

These supported models include:

- [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS/tree/main)
- [DiffCharge](https://github.com/LSY-Cython/DiffCharge/tree/main)
- [ACGAN](https://arxiv.org/abs/1610.09585)

Feel free to look at our [tutorial notebooks]() to get started.

# Install

## Requirements

**EnData** has been developed and tested on [Python 3.8](https://www.python.org/downloads/), [Python 3.9]((https://www.python.org/downloads/)) and [Python 3.10]((https://www.python.org/downloads/))

Also, although it is not strictly required, the usage of a [virtualenv](https://virtualenv.pypa.io/en/latest/)
is highly recommended in order to avoid interfering with other software installed in the system
in which **EnData** is run.

These are the minimum commands needed to create a virtualenv using python3.8 for **EnData**:

```bash
pip install virtualenv
virtualenv -p $(which python3.8) EnData-venv
```

Afterwards, you have to execute this command to activate the virtualenv:

```bash
source EnData-venv/bin/activate
```

Remember to execute it every time you start a new console to work on **EnData**!

## Install from PyPI

After creating the virtualenv and activating it, we recommend using
[pip](https://pip.pypa.io/en/stable/) in order to install **EnData**:

```bash
pip install EnData
```

This will pull and install the latest stable release from [PyPI](https://pypi.org/).
-->

## Install from source

With your virtualenv activated, you can clone the repository and install it from
source by running `make install` on the `stable` branch:

```bash
git clone git@github.com:michael-fuest/EnData.git
cd EnData
git checkout stable
make install
```

## Install for Development

If you want to contribute to the project, a few more steps are required to make the project ready
for development.

Please head to the [Contributing Guide](https://michael-fuest.github.io/EnData/contributing.html#get-started)
for more details about this process.

# Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **EnData**.

## Generating Data

To get started, define a DataGenerator and specify the name of the model you would like to sample from.

    generator = DataGenerator(model_name="diffusion_ts")

We provide pre-trained model checkpoints that were trained on the [PecanStreet Dataport](https://www.pecanstreet.org/dataport/) dataset. You can use these checkpoints to load a trained model as follows:

    generator.load_trained_model()

These pre-trained models are conditional models, meaning they require a set of conditioning variables to generate synthetic time series data. If you want to generate samples for a random set of conditioning variables, you can do so as follows:

    conditioning_variables = generator.sample_conditioning_variables(random=True)
    synthetic_data = generator.generate(conditioning_variables)

## Training

If you are interested in training these models from scratch on your own datasets, you need to start by creating a time series dataset object:

    your_dataset = TimeSeriesDataset(
        dataframe=your_custom_df,
        time_series_column_name="your_custom_timeseries_column_name,
        conditioning_vars="your_custom_conditioning_dict"
    )

This TimeSeriesDataset takes a [pandas](https://pandas.pydata.org/) Data Frame as input, and ensures compatability with EnData models. You can optionally define a dictionary of conditioning variables, where the keys correspond to the conditioning variable column names, and the values correspond to the number of discrete categories for that variable. Note that variables should already be Integer encoded. The following line of code starts model training:

    generator.fit(your_dataset, "your_custom_timeseries_column_name")

Once the model is trained, synthetic data can be created as before:

    my_custom_conditioning_variables = generator.sample_conditioning_variables(random=True)
    synthetic_data = generator.generate(conditioning_variables)

Please refer to our [documentation]() or to the [tutorials]() for more details.

## Datasets

If you want to reproduce our models from scratch, you will need to download the [PecanStreet DataPort dataset](https://www.pecanstreet.org/dataport/) and place it under the path specified in your configuration file. Specifically you will require the following files:

- 15minute_data_austin.csv
- 15minute_data_california.csv
- 15minute_data_newyork.csv
- metadata.csv

If you want to train models using the [Open Power Systems dataset](https://data.open-power-system-data.org/household_data/), you will need to download the following file:

- household_data_15min_singleindex.csv

# What's next?

For more details about **EnData** and all its possibilities
and features, please check the [documentation site](
https://michael-fuest.github.io/EnData/).

New models, new evaluation functionality and new datasets coming soon!
