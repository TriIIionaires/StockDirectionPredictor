# Stock Direction Predictor
A machine learning model developed in Jupyter Notebook using Python that predicts the next-day market trends. The project uses a random forest classification model to make predictions of whether tomorrow's closing price is higher or lower than the previous day. This project was made to learn about machine learning and its concepts like feature engineering, handling overfitting/underfitting, and evaluating model performance.

## Python Libraries
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![yfinance](https://img.shields.io/badge/yfinance-6001D2?logo=yahoo&logoColor=white)
![pandas-ta](https://img.shields.io/badge/pandas--ta-1B5FD4?logo=pandas&logoColor=white)  
**NOTE:** pandas-ta went closed-source and may not be publicly available

## Features
- **Forward Feature Selection:** The feature engineering process individually adds on indicators with the highest precision score after each run.
- **Classification Model:** The model predicts up (positive) or down (negative) which is easier to predict, but also harder to capitalize on the predictions it outputs since magnitude is not predicted.
- **Random Forest Algorithm:** This algorithm aggregates the mode of many many decision trees to train and make a prediction.

## Running the Project
To run the project in your local environment, follow these steps:
1. Clone the repository to your local machine.
2. Run all the cells above program.
3. Adjust tickers, time frame, confidence, and features to your own liking.
4. Run the program cell and view your results.

## Results
### Baseline
<img width="1539" height="805" alt="Baseline" src="https://github.com/user-attachments/assets/06fe7108-5f6a-48e7-a729-6ce530da7063" />  

### Final
<img width="1539" height="805" alt="Final" src="https://github.com/user-attachments/assets/b20aef2b-9820-4de5-b9d8-64476e8ed727" />
