# Minor Project 2: Predict New York City Airbnb Prices

**Student:** Muthuselvam R  
**Batch:** November 2026  
**Project domain:** Machine Learning – Linear Regression

## Project objective

Build a Linear Regression model to estimate New York City Airbnb listing prices using listing details. The workflow follows the seven steps in the AIML Minor Project 2 instructions.

## Dataset

Download **New York City Airbnb Open Data** by AB NYC from Kaggle:

https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

The expected CSV filename for this notebook is:

`AB_NYC_2019.csv`

After downloading the Kaggle ZIP file, extract the CSV. Do not upload the ZIP file in place of the CSV.

### Upload the dataset in Google Colab

1. Open the notebook in Google Colab.
2. Run this cell before the Step 1 data-loading cell:

```python
from google.colab import files
uploaded = files.upload()
```

3. Select `AB_NYC_2019.csv` from your computer.
4. Confirm that the uploaded filename is exactly `AB_NYC_2019.csv`.
5. Run the notebook cells from the beginning.

The notebook loads the file using:

```python
from pathlib import Path
DATA_FILE = Path("AB_NYC_2019.csv")
```

If you use Google Drive instead, mount Drive and set `DATA_FILE` to the full path of the CSV, for example:

```python
from google.colab import drive
drive.mount("/content/drive")
DATA_FILE = Path("/content/drive/MyDrive/AB_NYC_2019.csv")
```

For a local Jupyter Notebook, place the CSV in the notebook's working directory, or change `DATA_FILE` to the correct file path.

> **Dataset validation:** The assignment describes approximately 49,000 listings. After loading the CSV, inspect `df.shape`. If the row count is substantially lower (for example, 22,059), check that you downloaded and uploaded the complete Kaggle CSV rather than a filtered or partial file. Do not treat a partial dataset as the full dataset.

## Project workflow – all 7 required steps

1. **Exploratory Data Analysis (EDA) and Data Auditing** – inspect dataset shape, columns, data types, numerical/categorical features and missing-value counts/percentages.
2. **Data Cleaning and Handling Missing Values** – review `last_review` and `reviews_per_month`; remove irrelevant identifiers and unused columns.
3. **Outlier Identification and Treatment** – visualize `price` and `minimum_nights`; apply the 95th-percentile price rule and remove 365+ night minimum stays.
4. **Feature Selection and Categorical Encoding** – exclude the high-cardinality `neighbourhood` field and one-hot encode `neighbourhood_group` and `room_type` with the first category dropped.
5. **Feature-Target Separation and Validation Split** – separate X and y and split into 80% training and 20% testing data with a reproducible random state.
6. **Model Training and Evaluation** – train Linear Regression and calculate R², MAE, MSE and RMSE.
7. **Residual Analysis and Interpretation** – compare actual and predicted prices, inspect residuals and compare errors for lower-priced and higher-priced listings.

## Run the project

### Google Colab
Open `Muthuselvam_R_Airbnb_Linear_Regression.ipynb` in Colab, upload the CSV as described above, then run all cells from top to bottom.

### Local Jupyter
Install the dependencies:

```bash
pip install -r requirements.txt
```

Place `AB_NYC_2019.csv` beside the notebook and run all cells.

## Expected outputs

The notebook displays the EDA audit, cleaning results, outlier plots, model metrics, actual-versus-predicted plot, residual plot, residual distribution and price-group error comparison. Generated output files are saved in `airbnb_project_outputs/` when the corresponding code creates them.

**Important:** R², MAE, MSE, RMSE and all dataset-specific counts must be taken from an actual run on the complete dataset. Do not enter estimated or invented results in the final report.

## Repository contents

- `Muthuselvam_R_Airbnb_Linear_Regression.ipynb` – project notebook.
- `README.md` – project overview, dataset setup and execution instructions.
- `requirements.txt` – Python dependencies.

## Deliverables

1. A clean, well-commented Jupyter/Google Colab notebook following the seven project steps.
2. A one-page summary report stating the actual R² and MAE values and explaining their meaning in plain English.
