# Data Science Approach to Predictive Policing in Memphis

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Dataset](#dataset)
3.  [Tools & Technologies](#tools--technologies)
4.  [Key Findings & Insights](#key-findings--insights)
5.  [Methodology](#methodology)
6.  [Setup](#setup)
7.  [Project Structure](#project-structure)
8.  [Contact](#contact)

## Project Overview

This project aims to **apply data science approaches to derive insights from the Memphis Police Department Public Safety Dataset**, with the goal of helping the Memphis Police Department (MPD) strategically deploy its law enforcement resources.

## Dataset

The project utilizes a CSV file named `Memphis_Police_Department__Public_Safety_Incidents_20241110.csv`. This dataset contains **public safety incident data from the MPD, exported until November 10th, 2024**. It's important to note that **each row in the file represents a record taken by a Memphis Police Officer and may contain human errors or false information** provided by suspects and witnesses.

## Tools & Technologies

The project primarily uses Python and several libraries for data manipulation, analysis, and machine learning:

*   **pandas** (`pd`): For data loading and manipulation.
*   **numpy** (`np`): For numerical operations.
*   **matplotlib.pyplot** (`plt`): For data visualization.
*   **seaborn** (`sns`): For enhanced data visualization, particularly for statistical plots.
*   **geopy.distance**: Specifically the `geodesic` function, used to calculate distances.
*   **sklearn.preprocessing**: Includes `StandardScaler`, `MinMaxScaler`, and `LabelEncoder` for data transformation and scaling.
*   **sklearn.model_selection**: Includes `train_test_split` for splitting data into training and testing sets.
*   **sklearn.metrics**: Includes `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, and `confusion_matrix` for evaluating model performance.
*   **warnings**: Used to filter warnings during execution.

## Key Findings & Insights

Through exploratory data analysis and feature engineering, the project unearthed several key insights into crime patterns in Memphis:

*   **Highest Crime Categories**: The most frequent crime categories observed are **LARCENY/THEFT**, **ASSAULT**, and **DEST/DAM/VAND OF PROPERTY**.
*   **Geographic Hotspots**:
    *   **Precincts** with the highest crime counts include **RAINES**, **APPLING FARMS**, and **AUSTIN PEAY**.
    *   **Wards** with the highest crime counts include **822**, **925**, and **821**.
    *   The **Zip codes** with the highest crime frequency are **38109**, **38118**, and **38116**.
*   **Temporal Patterns**:
    *   **Crime trends show fluctuations over the years**, with 2023 having the highest number of reported crimes, followed by 2022 and 2018.
    *   Crime distribution by **hour of day** indicates periods of higher activity.
    *   The **"Night" period (9 PM to 4:59 AM) accounts for the highest number of reported incidents**, followed by "Afternoon" (12 PM to 4:59 PM), "Morning" (5 AM to 11:59 AM), and "Evening" (5 PM to 8:59 PM).
    *   There is a clear distinction in crime occurrences between **weekdays and weekends**, with an `is_weekend` feature indicating whether the crime occurred on a Saturday or Sunday.
*   **Relationship with Downtown**:
    *   Crime frequency is highest in areas **5-10 miles from downtown**, followed by 10-15 miles and 0-5 miles.
    *   Analysis also explores **crime types by their distance from downtown**.
*   **Factors Contributing to Crime Type**:
    *   For numerical features, 'Year', 'Lat', and 'is_memphis' show a low positive correlation with the 'severity' of the crime. 'Month', 'Day', 'Long', and 'is_weekend' show a low negative correlation with 'severity'.
    *   Among categorical features, 'Precinct', 'Street Name', and 'DayOfWeek' show a low positive correlation with 'severity'. 'Zip', 'binned_distance', and 'Full Address (100 Block)' show a low negative correlation with 'severity'.
*   **Predictive Model Performance**: A Multinomial Logistic Regression model was trained to predict crime severity. The model achieved an **accuracy of 0.5846**, **precision of 0.3934**, **recall of 0.5846**, and an **F1 Score of 0.4698** on the test set. The classification report indicates varying performance across different severity classes.

## Methodology

The project followed a **robust data life cycle** to ensure reliable and accurate analysis and predictions:

1.  **Understand the Data**:
    *   Loaded the dataset `Memphis_Police_Department__Public_Safety_Incidents_20241110.csv` using pandas.
    *   Inspected data types (`dtypes`) and descriptive statistics (`describe()`).
    *   Previewed the first few rows (`head()`).
2.  **Data Cleaning**:
    *   Identified and **removed columns with a large number of missing values** that were also not found in the data dictionary (e.g., "Memphis 3.0 Focus Anchors Buffers", "Community Neighborhoods Boundaries 2").
    *   **Dropped rows with missing critical information** in 'UCR Category', 'Group (NIBRS)', and 'Agency Crime ID'.
    *   **Cleaned the 'State' column** by filtering to 'TN', 'AR', 'MS' and removing cities outside the Memphis Metro area like 'SARDIS', 'PONTOTOC', 'CAMDEN'.
    *   **Filled missing 'Lat', 'Long', and 'Location' values** using information extracted from the `100 Block_Coordinates` column, which had no missing values.
    *   **Standardized and corrected 'Zip' codes** by removing extraneous characters (hyphens, trailing strings) and prepending '38' where appropriate to align with Memphis area zip codes, addressing potential human entry errors. Rows with unresolvable `Zip` values were dropped.
    *   **Filled missing '100 Block' values** with 0.0.
    *   **Filled missing 'Street Name', 'City', 'Precinct', and 'Ward' values with 'N/A'**.
    *   **Removed duplicate rows** from the dataset.
    *   **Removed duplicate entries based on the combination of 'Crime ID' and 'UCR Incident Code'**, keeping only the first occurrence to ensure unique crime records.
3.  **Exploratory Data Analysis (EDA)**:
    *   Conducted **EDA to gain a deeper understanding of the data**.
    *   **Plotted distributions** of key categorical features such as 'UCR Category', 'Offense Group (NIBRS)', 'Precinct', 'Ward', and 'Zip'.
    *   Analyzed **temporal patterns** by visualizing crime trends over years and crime distribution by hour of day.
    *   Explored the **relationship between crime frequency/types and distance from downtown Memphis**.
    *   Visualized distributions for newly engineered features like `severity` and `binned_distance`.
4.  **Feature Engineering**:
    *   **Created derived columns to enhance model inputs** and capture critical relationships.
    *   **Converted 'Offense Date' and 'Reported Date' to datetime objects** and extracted 'Year', 'Month', 'Day', 'Hour', and 'DayOfWeek'.
    *   Removed data with an erroneous year (2094).
    *   Calculated **'Distance from Downtown'** (in miles) using geographic coordinates.
    *   Introduced a **'severity' classification** based on the first digit of the 'UCR Incident Code'.
    *   Categorized **'Time of Day'** (Morning, Afternoon, Evening, Night) based on the 'Hour' of offense.
    *   Added a boolean **'is_memphis' indicator** (1 if City is Memphis, 0 otherwise).
    *   Added a boolean **'is_weekend' indicator** (1 for Saturday/Sunday, 0 for weekdays).
    *   **Binned 'Distance from Downtown'** into categories like '0-5', '5-10', '10-15', '15-20', '>20' miles.
5.  **Data Preparation**:
    *   **Prepared the data for machine learning by transforming and scaling it**.
    *   **Removed redundant or highly correlated features** that would not contribute meaningfully to the model or caused multicollinearity, such as various date/time columns, descriptive UCR fields, and coordinate derivatives.
    *   **Scaled numerical features** (`Lat`, `Long`, `Year`, `Month`, `Day`, `is_memphis`, `is_weekend`) using `MinMaxScaler` to normalize their ranges.
    *   **Encoded categorical features** (`Street Name`, `Zip`, `Full Address (100 Block)`, `Precinct`, `DayOfWeek`, `Time of Day`, `binned_distance`) using `LabelEncoder` to convert them into numerical representations suitable for machine learning models.
6.  **Model Exploration**:
    *   A **Multinomial Logistic Regression model** was implemented and trained on the prepared dataset to predict crime 'severity'.
7.  **Result Analysis**:
    *   Analyzed the model's results by computing and displaying **performance metrics** including accuracy, precision, recall, F1-score, and a confusion matrix, along with a classification report. This structured approach ensures that the analysis and predictions are both **reliable and accurate**.

## Setup

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

## Setup Instructions

1. **Create a Virtual Environment** (Recommended)
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install Required Packages**
   ```bash
   pip install jupyter notebook
   pip install pandas numpy matplotlib seaborn
   pip install geopy
   pip install scikit-learn
   ```

3. Download the dataset from https://data.memphistn.gov/Public-Safety/Memphis-Police-Department-Public-Safety-Incidents/puh4-eea4/about_data by clicking on **Export** and then **Download**. The dataset will also be attached herewith the project.

## Running the Notebook

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. Navigate to your `FDS_project.ipynb` file in the Jupyter interface that opens in your web browser.

3. Click on the notebook to open it.

## Project Structure

The repository contains the main Jupyter Notebook:

*   `FDS_Project.ipynb`: This notebook contains all the code for data loading, cleaning, exploratory data analysis, feature engineering, and model training and evaluation.

## Contact

*(Shristi Shrestha)* - [GitHub](https://github.com/shrististha) | [LinkedIn](https://linkedin.com/in/shristi-stha)
