### README.md for Data Exploration and Analysis Project

---

#### Project Overview

This project focuses on the data exploration and analysis of a loan payments test dataset. It utilizes Python for backend operations and Jupyter notebooks for exploratory data analysis (EDA) and visualization. The project has been enhanced to include in-depth evaluations of loan recoveries, potential losses, and risk assessments. Key components include:

- `db_utils.py`: Manages database operations and data transformations.
- `EDA_clean.ipynb`: Initial notebook for basic data exploration and visualization.
- `EDA_analysis.ipynb`: Advanced notebook for detailed analysis and visualization of loan data.

---

#### Installation and Setup

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Ensure Python 3.x is installed. Install required libraries:
    ```bash
    pip install pandas sqlalchemy seaborn matplotlib numpy scipy yaml
    ```
3. **Database Setup**:
    - Set up your database and ensure it's running.
    - Update the `credentials.yaml` file with your database credentials.
4. **Run the `db_utils.py`**: Connects to the database, performs data extraction and transformation.

---

#### db_utils.py

- **Functionality**:
  - Connects to an RDS database using YAML credentials.
  - Extracts and saves data locally.
  - Provides data transformation utilities.

- **Classes and Methods**:
  - `RDSDatabaseConnector`: Manages database connections and queries.
  - `DataTransform`: Methods for data type conversions and formatting.
  - `DataFrameInfo`: Functions to print DataFrame information and statistics.
  - `Plotter`: Methods for plotting distributions, missing values, and outliers.
  - `DataFrameTransform`: Methods for imputing missing values, transforming data, and outlier removal.

- **Usage**:
  - Import classes and utilize methods for data handling and analysis.

---

#### EDA_clean.ipynb

- **Purpose**: Used for initial data exploration and visualization.
- **Features**:
  - Load and clean data using `db_utils.py`.
  - Perform exploratory analysis with plots and statistical summaries.
  - Investigate correlations, outliers, and key data characteristics.
  - Generate a cleaned data file `loan_payments_cleaned.csv` for subsequent analysis.

#### EDA_analysis.ipynb

- **Purpose**: Advanced data analysis and visualization.
- **Prerequisite**: Ensure you run `EDA_clean.ipynb` first to generate the `loan_payments_cleaned.csv` file that this notebook will use.
- **Features**:
- **Features**:
  - In-depth analysis of loan recoveries, charged-off loans, and potential future risks.
  - Visualization of key metrics and insights.
  - Enhanced data processing and statistical analysis techniques.

- **Running the Notebooks**:
  - Begin with `EDA_clean.ipynb` to preprocess data and produce the cleaned dataset.
  - Next, open `EDA_analysis.ipynb` in Jupyter or a compatible IDE.
  - Run cells sequentially for advanced data analysis and visualization based on the cleaned dataset.


---

#### How to Contribute

- Fork the repository.
- Create a new branch for features or fixes.
- Submit pull requests with clear change descriptions.

---

#### Support

For support, open an issue in the repository. We will address it as soon as possible.

---

#### License

This project is licensed under the MIT License.

---
