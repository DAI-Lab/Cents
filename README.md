# Synthetic Household-Level Electricity Load Timeseries Generation

<img src="docs/images/dai_logo.png" alt="DAI Logo" width="100" height="100">

Repository for experimentation on algorithms for synthetic household-level electricity load timeseries generation.

## Setup Instructions

### Setting Up a Virtual Environment

To set up a virtual environment using `virtualenv`, follow these steps:

1. Install `virtualenv` if you haven't already:

    ```bash
    pip install virtualenv
    ```

2. Create a virtual environment named `venv`:

    ```bash
    virtualenv venv
    ```

3. Activate the virtual environment:

    - On Windows:

        ```bash
        venv\Scripts\activate
        ```

    - On macOS and Linux:

        ```bash
        source venv/bin/activate
        ```

### Installing Dependencies

Once the virtual environment is activated, you can install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## Configuration

### Config File

The configuration for this project is managed via a `config.yml` file located in the `config` directory. This file specifies the paths and columns for the datasets used in the project. Here is an example configuration:

```yaml
datasets:
  pecanstreetdata:
    path: "Users/michaelfuest/Research/DAI/EnData/data/pecanstreet/"
    columns: ["dataid", "building_type", "city", "pv", "car1", "car2", "grid", "solar", "solar2"]
  goinerdata:
    path: "Users/michaelfuest/Research/DAI/EnData/data/goiner/"
    columns: ["pv", "id"]
```

### Pecan Street Dataset

The Pecan Street Dataset can be downloaded [here](https://www.pecanstreet.org/dataport/).

Note: Access to the dataset requires a working university account. Metadata reports and the data dictionary can also be found on the Pecan Street Dataport website.

## License

This project is licensed under the MIT License. For more details, refer to the [LICENSE](LICENSE) file.


