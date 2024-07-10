# Stock Sector Classification Project

This project aims to classify stocks into their respective sectors using historical stock price data and various machine learning techniques. The project is implemented in a Jupyter Notebook and includes data fetching, preprocessing, feature extraction, model training, and prediction.

<br>

To run this project, you'll need to have Python installed along with the following libraries:

* yfinance
* pandas
* requests
* beautifulsoup4
* numpy
* scikit-learn
* tsfresh
* joblib
* imbalanced-learn
* feature-engine

<br>
You can install the required libraries using the following command:

<br>

```bash
pip install yfinance pandas requests beautifulsoup4 numpy scikit-learn tsfresh joblib imbalanced-learn feature-engine
```

<br>

## Project Structure
* **data**/: Directory containing the CSV files with stock symbols for different sectors.
* **financials**/: Financial data CSV files for various sectors.
* **healthcare**/: Healthcare data CSV files.
* **technology**/: Technology data CSV files.
* **energy**/: Energy data CSV files.
* **best_model.pkl**: The saved best model for sector classification.
* **main.ipynb**: The main Jupyter Notebook containing the project code.

<br>

## Data Fetching
The project fetches sector and industry data from the web using requests and BeautifulSoup. This allows us to dynamically get the most up-to-date list of sectors and industries, ensuring the analysis is relevant and current.

<br>

## Data Preprocessing
Reading Financial Data
Financial data is read from CSV files for different sectors. The stock symbols for each sector are extracted to be used for fetching historical stock price data.

<br>

## Fetching Historical Stock Price Data
The yfinance library is used to download historical stock price data for each sector. The adjusted close prices are resampled weekly, and the percentage change is calculated to get the momentum.

<br>

## Momentum Calculation
Momentum is calculated as the weekly percentage change in adjusted close prices. This helps in understanding the stock's performance over time, which is crucial for classification tasks.

<br>

## Feature Extraction
### Using tsfresh
The tsfresh library is utilized to extract features from the time series data. tsfresh automatically calculates a large number of time series characteristics, which can be used to train machine learning models. These features include:

* Statistical Features: Mean, variance, skewness, kurtosis, etc.
* Frequency Domain Features: Fourier coefficients, spectral energy, etc.
* Time Domain Features: Autocorrelation, partial autocorrelation, etc.

The extracted features are more informative and help in improving the performance of the classification models.

<br>

## Model Training
### Handling Missing Data
Missing data is handled using the Iterative Imputer with a K-Nearest Neighbors regressor. This technique iteratively imputes missing values, using the known part of the data to predict the missing values.

<br>

## Feature Selection
Feature selection is performed using a combination of techniques to ensure that the most relevant features are used for training the model:

* **Variance Threshold**: Removes all features with low variance.
* **Correlation Threshold**: Removes highly correlated features.
* **Model-Based Selection**: Uses machine learning models to select the most important features.

<br>

## Balancing the Dataset
The Synthetic Minority Over-sampling Technique (SMOTE) is used to handle class imbalance. This technique generates synthetic samples for the minority class to balance the dataset.


## Model Building and Hyperparameter Tuning
Multiple machine learning models are built and hyperparameter tuning is performed using **RandomizedSearchCV**. The models include:

* **Logistic Regression**
* **Random Forest Classifier**
* **Support Vector Machine**
* **Gradient Boosting Classifier**
* **K-Nearest Neighbors**

The best model is selected based on cross-validation accuracy.

<br>

## Model Evaluation
The selected model is evaluated on a test set using various metrics:

* ***Accuracy***: Measures the proportion of correctly classified instances.
* ***Confusion Matrix***: Provides insights into the classification errors.
* ***Classification*** Report: Includes precision, recall, and F1-score for each class.
* ***F1 Score***: The harmonic mean of precision and recall.

<br>

## Prediction
The trained model is used to predict the sectors of new stocks. The predictions are analyzed to determine the most likely sector for each stock.

<br>

## Acknowledgements
* This project utilizes data from yfinance and web scraping techniques using requests and BeautifulSoup.
* Feature extraction is performed using the tsfresh library.
* Machine learning models are built using scikit-learn.


