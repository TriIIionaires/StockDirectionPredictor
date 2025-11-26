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
<table id="T_53ccf">
  <thead>
    <tr>
      <th id="T_53ccf_level0_col0" class="col_heading level0 col0" >Equity</th>
      <th id="T_53ccf_level0_col1" class="col_heading level0 col1" >Symbol</th>
      <th id="T_53ccf_level0_col2" class="col_heading level0 col2" >Accuracy Score</th>
      <th id="T_53ccf_level0_col3" class="col_heading level0 col3" >Precision Score</th>
      <th id="T_53ccf_level0_col4" class="col_heading level0 col4" >Recall Score</th>
      <th id="T_53ccf_level0_col5" class="col_heading level0 col5" >F1 Score</th>
      <th id="T_53ccf_level0_col6" class="col_heading level0 col6" >Bullish Frequency</th>
      <th id="T_53ccf_level0_col7" class="col_heading level0 col7" >Bearish Frequency</th>
      <th id="T_53ccf_level0_col8" class="col_heading level0 col8" >Positive Rate</th>
      <th id="T_53ccf_level0_col9" class="col_heading level0 col9" >Negative Rate</th>
      <th id="T_53ccf_level0_col10" class="col_heading level0 col10" >Positive Predictions</th>
      <th id="T_53ccf_level0_col11" class="col_heading level0 col11" >Negative Predictions</th>
      <th id="T_53ccf_level0_col12" class="col_heading level0 col12" >Total Predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_53ccf_row0_col0" class="data row0 col0" >Visa Inc.</td>
      <td id="T_53ccf_row0_col1" class="data row0 col1" >V</td>
      <td id="T_53ccf_row0_col2" class="data row0 col2" >0.512113</td>
      <td id="T_53ccf_row0_col3" class="data row0 col3" >0.568662</td>
      <td id="T_53ccf_row0_col4" class="data row0 col4" >0.402242</td>
      <td id="T_53ccf_row0_col5" class="data row0 col5" >0.471189</td>
      <td id="T_53ccf_row0_col6" class="data row0 col6" >0.540377</td>
      <td id="T_53ccf_row0_col7" class="data row0 col7" >0.459623</td>
      <td id="T_53ccf_row0_col8" class="data row0 col8" >0.382234</td>
      <td id="T_53ccf_row0_col9" class="data row0 col9" >0.617766</td>
      <td id="T_53ccf_row0_col10" class="data row0 col10" >568</td>
      <td id="T_53ccf_row0_col11" class="data row0 col11" >918</td>
      <td id="T_53ccf_row0_col12" class="data row0 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row1_col0" class="data row1 col0" >International Business Machines</td>
      <td id="T_53ccf_row1_col1" class="data row1 col1" >IBM</td>
      <td id="T_53ccf_row1_col2" class="data row1 col2" >0.535666</td>
      <td id="T_53ccf_row1_col3" class="data row1 col3" >0.557429</td>
      <td id="T_53ccf_row1_col4" class="data row1 col4" >0.662078</td>
      <td id="T_53ccf_row1_col5" class="data row1 col5" >0.605263</td>
      <td id="T_53ccf_row1_col6" class="data row1 col6" >0.537685</td>
      <td id="T_53ccf_row1_col7" class="data row1 col7" >0.462315</td>
      <td id="T_53ccf_row1_col8" class="data row1 col8" >0.638627</td>
      <td id="T_53ccf_row1_col9" class="data row1 col9" >0.361373</td>
      <td id="T_53ccf_row1_col10" class="data row1 col10" >949</td>
      <td id="T_53ccf_row1_col11" class="data row1 col11" >537</td>
      <td id="T_53ccf_row1_col12" class="data row1 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row2_col0" class="data row2 col0" >Procter & Gamble Company (The)</td>
      <td id="T_53ccf_row2_col1" class="data row2 col1" >PG</td>
      <td id="T_53ccf_row2_col2" class="data row2 col2" >0.505384</td>
      <td id="T_53ccf_row2_col3" class="data row2 col3" >0.546900</td>
      <td id="T_53ccf_row2_col4" class="data row2 col4" >0.433249</td>
      <td id="T_53ccf_row2_col5" class="data row2 col5" >0.483486</td>
      <td id="T_53ccf_row2_col6" class="data row2 col6" >0.534320</td>
      <td id="T_53ccf_row2_col7" class="data row2 col7" >0.465680</td>
      <td id="T_53ccf_row2_col8" class="data row2 col8" >0.423284</td>
      <td id="T_53ccf_row2_col9" class="data row2 col9" >0.576716</td>
      <td id="T_53ccf_row2_col10" class="data row2 col10" >629</td>
      <td id="T_53ccf_row2_col11" class="data row2 col11" >857</td>
      <td id="T_53ccf_row2_col12" class="data row2 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row3_col0" class="data row3 col0" >NVIDIA Corporation</td>
      <td id="T_53ccf_row3_col1" class="data row3 col1" >NVDA</td>
      <td id="T_53ccf_row3_col2" class="data row3 col2" >0.481157</td>
      <td id="T_53ccf_row3_col3" class="data row3 col3" >0.541176</td>
      <td id="T_53ccf_row3_col4" class="data row3 col4" >0.339483</td>
      <td id="T_53ccf_row3_col5" class="data row3 col5" >0.417234</td>
      <td id="T_53ccf_row3_col6" class="data row3 col6" >0.547106</td>
      <td id="T_53ccf_row3_col7" class="data row3 col7" >0.452894</td>
      <td id="T_53ccf_row3_col8" class="data row3 col8" >0.343203</td>
      <td id="T_53ccf_row3_col9" class="data row3 col9" >0.656797</td>
      <td id="T_53ccf_row3_col10" class="data row3 col10" >510</td>
      <td id="T_53ccf_row3_col11" class="data row3 col11" >976</td>
      <td id="T_53ccf_row3_col12" class="data row3 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row4_col0" class="data row4 col0" >Coca-Cola Company (The)</td>
      <td id="T_53ccf_row4_col1" class="data row4 col1" >KO</td>
      <td id="T_53ccf_row4_col2" class="data row4 col2" >0.499327</td>
      <td id="T_53ccf_row4_col3" class="data row4 col3" >0.533795</td>
      <td id="T_53ccf_row4_col4" class="data row4 col4" >0.393359</td>
      <td id="T_53ccf_row4_col5" class="data row4 col5" >0.452941</td>
      <td id="T_53ccf_row4_col6" class="data row4 col6" >0.526918</td>
      <td id="T_53ccf_row4_col7" class="data row4 col7" >0.473082</td>
      <td id="T_53ccf_row4_col8" class="data row4 col8" >0.388291</td>
      <td id="T_53ccf_row4_col9" class="data row4 col9" >0.611709</td>
      <td id="T_53ccf_row4_col10" class="data row4 col10" >577</td>
      <td id="T_53ccf_row4_col11" class="data row4 col11" >909</td>
      <td id="T_53ccf_row4_col12" class="data row4 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row5_col0" class="data row5 col0" >Chevron Corporation</td>
      <td id="T_53ccf_row5_col1" class="data row5 col1" >CVX</td>
      <td id="T_53ccf_row5_col2" class="data row5 col2" >0.506729</td>
      <td id="T_53ccf_row5_col3" class="data row5 col3" >0.533648</td>
      <td id="T_53ccf_row5_col4" class="data row5 col4" >0.572152</td>
      <td id="T_53ccf_row5_col5" class="data row5 col5" >0.552230</td>
      <td id="T_53ccf_row5_col6" class="data row5 col6" >0.531629</td>
      <td id="T_53ccf_row5_col7" class="data row5 col7" >0.468371</td>
      <td id="T_53ccf_row5_col8" class="data row5 col8" >0.569987</td>
      <td id="T_53ccf_row5_col9" class="data row5 col9" >0.430013</td>
      <td id="T_53ccf_row5_col10" class="data row5 col10" >847</td>
      <td id="T_53ccf_row5_col11" class="data row5 col11" >639</td>
      <td id="T_53ccf_row5_col12" class="data row5 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row6_col0" class="data row6 col0" >Meta Platforms, Inc.</td>
      <td id="T_53ccf_row6_col1" class="data row6 col1" >META</td>
      <td id="T_53ccf_row6_col2" class="data row6 col2" >0.503946</td>
      <td id="T_53ccf_row6_col3" class="data row6 col3" >0.533499</td>
      <td id="T_53ccf_row6_col4" class="data row6 col4" >0.460385</td>
      <td id="T_53ccf_row6_col5" class="data row6 col5" >0.494253</td>
      <td id="T_53ccf_row6_col6" class="data row6 col6" >0.526494</td>
      <td id="T_53ccf_row6_col7" class="data row6 col7" >0.473506</td>
      <td id="T_53ccf_row6_col8" class="data row6 col8" >0.454340</td>
      <td id="T_53ccf_row6_col9" class="data row6 col9" >0.545660</td>
      <td id="T_53ccf_row6_col10" class="data row6 col10" >403</td>
      <td id="T_53ccf_row6_col11" class="data row6 col11" >484</td>
      <td id="T_53ccf_row6_col12" class="data row6 col12" >887</td>
    </tr>
    <tr>
      <td id="T_53ccf_row7_col0" class="data row7 col0" >Johnson & Johnson</td>
      <td id="T_53ccf_row7_col1" class="data row7 col1" >JNJ</td>
      <td id="T_53ccf_row7_col2" class="data row7 col2" >0.502692</td>
      <td id="T_53ccf_row7_col3" class="data row7 col3" >0.532995</td>
      <td id="T_53ccf_row7_col4" class="data row7 col4" >0.274510</td>
      <td id="T_53ccf_row7_col5" class="data row7 col5" >0.362381</td>
      <td id="T_53ccf_row7_col6" class="data row7 col6" >0.514805</td>
      <td id="T_53ccf_row7_col7" class="data row7 col7" >0.485195</td>
      <td id="T_53ccf_row7_col8" class="data row7 col8" >0.265141</td>
      <td id="T_53ccf_row7_col9" class="data row7 col9" >0.734859</td>
      <td id="T_53ccf_row7_col10" class="data row7 col10" >394</td>
      <td id="T_53ccf_row7_col11" class="data row7 col11" >1092</td>
      <td id="T_53ccf_row7_col12" class="data row7 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row8_col0" class="data row8 col0" >Amazon.com, Inc.</td>
      <td id="T_53ccf_row8_col1" class="data row8 col1" >AMZN</td>
      <td id="T_53ccf_row8_col2" class="data row8 col2" >0.496635</td>
      <td id="T_53ccf_row8_col3" class="data row8 col3" >0.528340</td>
      <td id="T_53ccf_row8_col4" class="data row8 col4" >0.336340</td>
      <td id="T_53ccf_row8_col5" class="data row8 col5" >0.411024</td>
      <td id="T_53ccf_row8_col6" class="data row8 col6" >0.522207</td>
      <td id="T_53ccf_row8_col7" class="data row8 col7" >0.477793</td>
      <td id="T_53ccf_row8_col8" class="data row8 col8" >0.332436</td>
      <td id="T_53ccf_row8_col9" class="data row8 col9" >0.667564</td>
      <td id="T_53ccf_row8_col10" class="data row8 col10" >494</td>
      <td id="T_53ccf_row8_col11" class="data row8 col11" >992</td>
      <td id="T_53ccf_row8_col12" class="data row8 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row9_col0" class="data row9 col0" >Pepsico, Inc.</td>
      <td id="T_53ccf_row9_col1" class="data row9 col1" >PEP</td>
      <td id="T_53ccf_row9_col2" class="data row9 col2" >0.495289</td>
      <td id="T_53ccf_row9_col3" class="data row9 col3" >0.526971</td>
      <td id="T_53ccf_row9_col4" class="data row9 col4" >0.327320</td>
      <td id="T_53ccf_row9_col5" class="data row9 col5" >0.403816</td>
      <td id="T_53ccf_row9_col6" class="data row9 col6" >0.522207</td>
      <td id="T_53ccf_row9_col7" class="data row9 col7" >0.477793</td>
      <td id="T_53ccf_row9_col8" class="data row9 col8" >0.324361</td>
      <td id="T_53ccf_row9_col9" class="data row9 col9" >0.675639</td>
      <td id="T_53ccf_row9_col10" class="data row9 col10" >482</td>
      <td id="T_53ccf_row9_col11" class="data row9 col11" >1004</td>
      <td id="T_53ccf_row9_col12" class="data row9 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row10_col0" class="data row10 col0" >3M Company</td>
      <td id="T_53ccf_row10_col1" class="data row10 col1" >MMM</td>
      <td id="T_53ccf_row10_col2" class="data row10 col2" >0.520861</td>
      <td id="T_53ccf_row10_col3" class="data row10 col3" >0.524070</td>
      <td id="T_53ccf_row10_col4" class="data row10 col4" >0.633598</td>
      <td id="T_53ccf_row10_col5" class="data row10 col5" >0.573653</td>
      <td id="T_53ccf_row10_col6" class="data row10 col6" >0.508748</td>
      <td id="T_53ccf_row10_col7" class="data row10 col7" >0.491252</td>
      <td id="T_53ccf_row10_col8" class="data row10 col8" >0.615074</td>
      <td id="T_53ccf_row10_col9" class="data row10 col9" >0.384926</td>
      <td id="T_53ccf_row10_col10" class="data row10 col10" >914</td>
      <td id="T_53ccf_row10_col11" class="data row10 col11" >572</td>
      <td id="T_53ccf_row10_col12" class="data row10 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row11_col0" class="data row11 col0" >Verizon Communications Inc.</td>
      <td id="T_53ccf_row11_col1" class="data row11 col1" >VZ</td>
      <td id="T_53ccf_row11_col2" class="data row11 col2" >0.524226</td>
      <td id="T_53ccf_row11_col3" class="data row11 col3" >0.523288</td>
      <td id="T_53ccf_row11_col4" class="data row11 col4" >0.515520</td>
      <td id="T_53ccf_row11_col5" class="data row11 col5" >0.519375</td>
      <td id="T_53ccf_row11_col6" class="data row11 col6" >0.498654</td>
      <td id="T_53ccf_row11_col7" class="data row11 col7" >0.501346</td>
      <td id="T_53ccf_row11_col8" class="data row11 col8" >0.491252</td>
      <td id="T_53ccf_row11_col9" class="data row11 col9" >0.508748</td>
      <td id="T_53ccf_row11_col10" class="data row11 col10" >730</td>
      <td id="T_53ccf_row11_col11" class="data row11 col11" >756</td>
      <td id="T_53ccf_row11_col12" class="data row11 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row12_col0" class="data row12 col0" >UnitedHealth Group Incorporated</td>
      <td id="T_53ccf_row12_col1" class="data row12 col1" >UNH</td>
      <td id="T_53ccf_row12_col2" class="data row12 col2" >0.491925</td>
      <td id="T_53ccf_row12_col3" class="data row12 col3" >0.523148</td>
      <td id="T_53ccf_row12_col4" class="data row12 col4" >0.431847</td>
      <td id="T_53ccf_row12_col5" class="data row12 col5" >0.473133</td>
      <td id="T_53ccf_row12_col6" class="data row12 col6" >0.528264</td>
      <td id="T_53ccf_row12_col7" class="data row12 col7" >0.471736</td>
      <td id="T_53ccf_row12_col8" class="data row12 col8" >0.436070</td>
      <td id="T_53ccf_row12_col9" class="data row12 col9" >0.563930</td>
      <td id="T_53ccf_row12_col10" class="data row12 col10" >648</td>
      <td id="T_53ccf_row12_col11" class="data row12 col11" >838</td>
      <td id="T_53ccf_row12_col12" class="data row12 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row13_col0" class="data row13 col0" >Walmart Inc.</td>
      <td id="T_53ccf_row13_col1" class="data row13 col1" >WMT</td>
      <td id="T_53ccf_row13_col2" class="data row13 col2" >0.490579</td>
      <td id="T_53ccf_row13_col3" class="data row13 col3" >0.521739</td>
      <td id="T_53ccf_row13_col4" class="data row13 col4" >0.486692</td>
      <td id="T_53ccf_row13_col5" class="data row13 col5" >0.503607</td>
      <td id="T_53ccf_row13_col6" class="data row13 col6" >0.530956</td>
      <td id="T_53ccf_row13_col7" class="data row13 col7" >0.469044</td>
      <td id="T_53ccf_row13_col8" class="data row13 col8" >0.495289</td>
      <td id="T_53ccf_row13_col9" class="data row13 col9" >0.504711</td>
      <td id="T_53ccf_row13_col10" class="data row13 col10" >736</td>
      <td id="T_53ccf_row13_col11" class="data row13 col11" >750</td>
      <td id="T_53ccf_row13_col12" class="data row13 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row14_col0" class="data row14 col0" >Lockheed Martin Corporation</td>
      <td id="T_53ccf_row14_col1" class="data row14 col1" >LMT</td>
      <td id="T_53ccf_row14_col2" class="data row14 col2" >0.499327</td>
      <td id="T_53ccf_row14_col3" class="data row14 col3" >0.520930</td>
      <td id="T_53ccf_row14_col4" class="data row14 col4" >0.574359</td>
      <td id="T_53ccf_row14_col5" class="data row14 col5" >0.546341</td>
      <td id="T_53ccf_row14_col6" class="data row14 col6" >0.524899</td>
      <td id="T_53ccf_row14_col7" class="data row14 col7" >0.475101</td>
      <td id="T_53ccf_row14_col8" class="data row14 col8" >0.578735</td>
      <td id="T_53ccf_row14_col9" class="data row14 col9" >0.421265</td>
      <td id="T_53ccf_row14_col10" class="data row14 col10" >860</td>
      <td id="T_53ccf_row14_col11" class="data row14 col11" >626</td>
      <td id="T_53ccf_row14_col12" class="data row14 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row15_col0" class="data row15 col0" >Cisco Systems, Inc.</td>
      <td id="T_53ccf_row15_col1" class="data row15 col1" >CSCO</td>
      <td id="T_53ccf_row15_col2" class="data row15 col2" >0.498654</td>
      <td id="T_53ccf_row15_col3" class="data row15 col3" >0.516706</td>
      <td id="T_53ccf_row15_col4" class="data row15 col4" >0.560155</td>
      <td id="T_53ccf_row15_col5" class="data row15 col5" >0.537554</td>
      <td id="T_53ccf_row15_col6" class="data row15 col6" >0.520188</td>
      <td id="T_53ccf_row15_col7" class="data row15 col7" >0.479812</td>
      <td id="T_53ccf_row15_col8" class="data row15 col8" >0.563930</td>
      <td id="T_53ccf_row15_col9" class="data row15 col9" >0.436070</td>
      <td id="T_53ccf_row15_col10" class="data row15 col10" >838</td>
      <td id="T_53ccf_row15_col11" class="data row15 col11" >648</td>
      <td id="T_53ccf_row15_col12" class="data row15 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row16_col0" class="data row16 col0" >Intel Corporation</td>
      <td id="T_53ccf_row16_col1" class="data row16 col1" >INTC</td>
      <td id="T_53ccf_row16_col2" class="data row16 col2" >0.516151</td>
      <td id="T_53ccf_row16_col3" class="data row16 col3" >0.516484</td>
      <td id="T_53ccf_row16_col4" class="data row16 col4" >0.444595</td>
      <td id="T_53ccf_row16_col5" class="data row16 col5" >0.477850</td>
      <td id="T_53ccf_row16_col6" class="data row16 col6" >0.497981</td>
      <td id="T_53ccf_row16_col7" class="data row16 col7" >0.502019</td>
      <td id="T_53ccf_row16_col8" class="data row16 col8" >0.428668</td>
      <td id="T_53ccf_row16_col9" class="data row16 col9" >0.571332</td>
      <td id="T_53ccf_row16_col10" class="data row16 col10" >637</td>
      <td id="T_53ccf_row16_col11" class="data row16 col11" >849</td>
      <td id="T_53ccf_row16_col12" class="data row16 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row17_col0" class="data row17 col0" >Oracle Corporation</td>
      <td id="T_53ccf_row17_col1" class="data row17 col1" >ORCL</td>
      <td id="T_53ccf_row17_col2" class="data row17 col2" >0.475774</td>
      <td id="T_53ccf_row17_col3" class="data row17 col3" >0.514870</td>
      <td id="T_53ccf_row17_col4" class="data row17 col4" >0.348428</td>
      <td id="T_53ccf_row17_col5" class="data row17 col5" >0.415604</td>
      <td id="T_53ccf_row17_col6" class="data row17 col6" >0.534993</td>
      <td id="T_53ccf_row17_col7" class="data row17 col7" >0.465007</td>
      <td id="T_53ccf_row17_col8" class="data row17 col8" >0.362046</td>
      <td id="T_53ccf_row17_col9" class="data row17 col9" >0.637954</td>
      <td id="T_53ccf_row17_col10" class="data row17 col10" >538</td>
      <td id="T_53ccf_row17_col11" class="data row17 col11" >948</td>
      <td id="T_53ccf_row17_col12" class="data row17 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row18_col0" class="data row18 col0" >Microsoft Corporation</td>
      <td id="T_53ccf_row18_col1" class="data row18 col1" >MSFT</td>
      <td id="T_53ccf_row18_col2" class="data row18 col2" >0.479139</td>
      <td id="T_53ccf_row18_col3" class="data row18 col3" >0.511811</td>
      <td id="T_53ccf_row18_col4" class="data row18 col4" >0.411914</td>
      <td id="T_53ccf_row18_col5" class="data row18 col5" >0.456461</td>
      <td id="T_53ccf_row18_col6" class="data row18 col6" >0.530956</td>
      <td id="T_53ccf_row18_col7" class="data row18 col7" >0.469044</td>
      <td id="T_53ccf_row18_col8" class="data row18 col8" >0.427322</td>
      <td id="T_53ccf_row18_col9" class="data row18 col9" >0.572678</td>
      <td id="T_53ccf_row18_col10" class="data row18 col10" >635</td>
      <td id="T_53ccf_row18_col11" class="data row18 col11" >851</td>
      <td id="T_53ccf_row18_col12" class="data row18 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row19_col0" class="data row19 col0" >Wells Fargo & Company</td>
      <td id="T_53ccf_row19_col1" class="data row19 col1" >WFC</td>
      <td id="T_53ccf_row19_col2" class="data row19 col2" >0.494616</td>
      <td id="T_53ccf_row19_col3" class="data row19 col3" >0.502755</td>
      <td id="T_53ccf_row19_col4" class="data row19 col4" >0.483444</td>
      <td id="T_53ccf_row19_col5" class="data row19 col5" >0.492910</td>
      <td id="T_53ccf_row19_col6" class="data row19 col6" >0.508075</td>
      <td id="T_53ccf_row19_col7" class="data row19 col7" >0.491925</td>
      <td id="T_53ccf_row19_col8" class="data row19 col8" >0.488560</td>
      <td id="T_53ccf_row19_col9" class="data row19 col9" >0.511440</td>
      <td id="T_53ccf_row19_col10" class="data row19 col10" >726</td>
      <td id="T_53ccf_row19_col11" class="data row19 col11" >760</td>
      <td id="T_53ccf_row19_col12" class="data row19 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row20_col0" class="data row20 col0" >Goldman Sachs Group, Inc. (The)</td>
      <td id="T_53ccf_row20_col1" class="data row20 col1" >GS</td>
      <td id="T_53ccf_row20_col2" class="data row20 col2" >0.475774</td>
      <td id="T_53ccf_row20_col3" class="data row20 col3" >0.495522</td>
      <td id="T_53ccf_row20_col4" class="data row20 col4" >0.429495</td>
      <td id="T_53ccf_row20_col5" class="data row20 col5" >0.460152</td>
      <td id="T_53ccf_row20_col6" class="data row20 col6" >0.520188</td>
      <td id="T_53ccf_row20_col7" class="data row20 col7" >0.479812</td>
      <td id="T_53ccf_row20_col8" class="data row20 col8" >0.450875</td>
      <td id="T_53ccf_row20_col9" class="data row20 col9" >0.549125</td>
      <td id="T_53ccf_row20_col10" class="data row20 col10" >670</td>
      <td id="T_53ccf_row20_col11" class="data row20 col11" >816</td>
      <td id="T_53ccf_row20_col12" class="data row20 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row21_col0" class="data row21 col0" >Boeing Company (The)</td>
      <td id="T_53ccf_row21_col1" class="data row21 col1" >BA</td>
      <td id="T_53ccf_row21_col2" class="data row21 col2" >0.500000</td>
      <td id="T_53ccf_row21_col3" class="data row21 col3" >0.492596</td>
      <td id="T_53ccf_row21_col4" class="data row21 col4" >0.685440</td>
      <td id="T_53ccf_row21_col5" class="data row21 col5" >0.573234</td>
      <td id="T_53ccf_row21_col6" class="data row21 col6" >0.489906</td>
      <td id="T_53ccf_row21_col7" class="data row21 col7" >0.510094</td>
      <td id="T_53ccf_row21_col8" class="data row21 col8" >0.681696</td>
      <td id="T_53ccf_row21_col9" class="data row21 col9" >0.318304</td>
      <td id="T_53ccf_row21_col10" class="data row21 col10" >1013</td>
      <td id="T_53ccf_row21_col11" class="data row21 col11" >473</td>
      <td id="T_53ccf_row21_col12" class="data row21 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row22_col0" class="data row22 col0" >AT&T Inc.</td>
      <td id="T_53ccf_row22_col1" class="data row22 col1" >T</td>
      <td id="T_53ccf_row22_col2" class="data row22 col2" >0.477793</td>
      <td id="T_53ccf_row22_col3" class="data row22 col3" >0.490476</td>
      <td id="T_53ccf_row22_col4" class="data row22 col4" >0.404450</td>
      <td id="T_53ccf_row22_col5" class="data row22 col5" >0.443329</td>
      <td id="T_53ccf_row22_col6" class="data row22 col6" >0.514132</td>
      <td id="T_53ccf_row22_col7" class="data row22 col7" >0.485868</td>
      <td id="T_53ccf_row22_col8" class="data row22 col8" >0.423957</td>
      <td id="T_53ccf_row22_col9" class="data row22 col9" >0.576043</td>
      <td id="T_53ccf_row22_col10" class="data row22 col10" >630</td>
      <td id="T_53ccf_row22_col11" class="data row22 col11" >856</td>
      <td id="T_53ccf_row22_col12" class="data row22 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row23_col0" class="data row23 col0" >Exxon Mobil Corporation</td>
      <td id="T_53ccf_row23_col1" class="data row23 col1" >XOM</td>
      <td id="T_53ccf_row23_col2" class="data row23 col2" >0.472409</td>
      <td id="T_53ccf_row23_col3" class="data row23 col3" >0.481739</td>
      <td id="T_53ccf_row23_col4" class="data row23 col4" >0.363041</td>
      <td id="T_53ccf_row23_col5" class="data row23 col5" >0.414051</td>
      <td id="T_53ccf_row23_col6" class="data row23 col6" >0.513459</td>
      <td id="T_53ccf_row23_col7" class="data row23 col7" >0.486541</td>
      <td id="T_53ccf_row23_col8" class="data row23 col8" >0.386945</td>
      <td id="T_53ccf_row23_col9" class="data row23 col9" >0.613055</td>
      <td id="T_53ccf_row23_col10" class="data row23 col10" >575</td>
      <td id="T_53ccf_row23_col11" class="data row23 col11" >911</td>
      <td id="T_53ccf_row23_col12" class="data row23 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row24_col0" class="data row24 col0" >Pfizer, Inc.</td>
      <td id="T_53ccf_row24_col1" class="data row24 col1" >PFE</td>
      <td id="T_53ccf_row24_col2" class="data row24 col2" >0.508748</td>
      <td id="T_53ccf_row24_col3" class="data row24 col3" >0.476106</td>
      <td id="T_53ccf_row24_col4" class="data row24 col4" >0.382646</td>
      <td id="T_53ccf_row24_col5" class="data row24 col5" >0.424290</td>
      <td id="T_53ccf_row24_col6" class="data row24 col6" >0.473082</td>
      <td id="T_53ccf_row24_col7" class="data row24 col7" >0.526918</td>
      <td id="T_53ccf_row24_col8" class="data row24 col8" >0.380215</td>
      <td id="T_53ccf_row24_col9" class="data row24 col9" >0.619785</td>
      <td id="T_53ccf_row24_col10" class="data row24 col10" >565</td>
      <td id="T_53ccf_row24_col11" class="data row24 col11" >921</td>
      <td id="T_53ccf_row24_col12" class="data row24 col12" >1486</td>
    </tr>
    <tr>
      <td id="T_53ccf_row25_col0" class="data row25 col0" >Walt Disney Company (The)</td>
      <td id="T_53ccf_row25_col1" class="data row25 col1" >DIS</td>
      <td id="T_53ccf_row25_col2" class="data row25 col2" >0.494616</td>
      <td id="T_53ccf_row25_col3" class="data row25 col3" >0.473684</td>
      <td id="T_53ccf_row25_col4" class="data row25 col4" >0.426778</td>
      <td id="T_53ccf_row25_col5" class="data row25 col5" >0.449010</td>
      <td id="T_53ccf_row25_col6" class="data row25 col6" >0.482503</td>
      <td id="T_53ccf_row25_col7" class="data row25 col7" >0.517497</td>
      <td id="T_53ccf_row25_col8" class="data row25 col8" >0.434724</td>
      <td id="T_53ccf_row25_col9" class="data row25 col9" >0.565276</td>
      <td id="T_53ccf_row25_col10" class="data row25 col10" >646</td>
      <td id="T_53ccf_row25_col11" class="data row25 col11" >840</td>
      <td id="T_53ccf_row25_col12" class="data row25 col12" >1486</td>
    </tr>
  </tbody>
</table>

### Final
<table id="T_5a812">
  <thead>
    <tr>
      <th id="T_5a812_level0_col0" class="col_heading level0 col0" >Equity</th>
      <th id="T_5a812_level0_col1" class="col_heading level0 col1" >Symbol</th>
      <th id="T_5a812_level0_col2" class="col_heading level0 col2" >Accuracy Score</th>
      <th id="T_5a812_level0_col3" class="col_heading level0 col3" >Precision Score</th>
      <th id="T_5a812_level0_col4" class="col_heading level0 col4" >Recall Score</th>
      <th id="T_5a812_level0_col5" class="col_heading level0 col5" >F1 Score</th>
      <th id="T_5a812_level0_col6" class="col_heading level0 col6" >Bullish Frequency</th>
      <th id="T_5a812_level0_col7" class="col_heading level0 col7" >Bearish Frequency</th>
      <th id="T_5a812_level0_col8" class="col_heading level0 col8" >Positive Rate</th>
      <th id="T_5a812_level0_col9" class="col_heading level0 col9" >Negative Rate</th>
      <th id="T_5a812_level0_col10" class="col_heading level0 col10" >Positive Predictions</th>
      <th id="T_5a812_level0_col11" class="col_heading level0 col11" >Negative Predictions</th>
      <th id="T_5a812_level0_col12" class="col_heading level0 col12" >Total Predictions</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td id="T_5a812_row0_col0" class="data row0 col0" >NVIDIA Corporation</td>
      <td id="T_5a812_row0_col1" class="data row0 col1" >NVDA</td>
      <td id="T_5a812_row0_col2" class="data row0 col2" >0.569473</td>
      <td id="T_5a812_row0_col3" class="data row0 col3" >0.672878</td>
      <td id="T_5a812_row0_col4" class="data row0 col4" >0.408291</td>
      <td id="T_5a812_row0_col5" class="data row0 col5" >0.508210</td>
      <td id="T_5a812_row0_col6" class="data row0 col6" >0.544832</td>
      <td id="T_5a812_row0_col7" class="data row0 col7" >0.455168</td>
      <td id="T_5a812_row0_col8" class="data row0 col8" >0.330595</td>
      <td id="T_5a812_row0_col9" class="data row0 col9" >0.669405</td>
      <td id="T_5a812_row0_col10" class="data row0 col10" >483</td>
      <td id="T_5a812_row0_col11" class="data row0 col11" >978</td>
      <td id="T_5a812_row0_col12" class="data row0 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row1_col0" class="data row1 col0" >Chevron Corporation</td>
      <td id="T_5a812_row1_col1" class="data row1 col1" >CVX</td>
      <td id="T_5a812_row1_col2" class="data row1 col2" >0.637919</td>
      <td id="T_5a812_row1_col3" class="data row1 col3" >0.669333</td>
      <td id="T_5a812_row1_col4" class="data row1 col4" >0.641124</td>
      <td id="T_5a812_row1_col5" class="data row1 col5" >0.654925</td>
      <td id="T_5a812_row1_col6" class="data row1 col6" >0.535934</td>
      <td id="T_5a812_row1_col7" class="data row1 col7" >0.464066</td>
      <td id="T_5a812_row1_col8" class="data row1 col8" >0.513347</td>
      <td id="T_5a812_row1_col9" class="data row1 col9" >0.486653</td>
      <td id="T_5a812_row1_col10" class="data row1 col10" >750</td>
      <td id="T_5a812_row1_col11" class="data row1 col11" >711</td>
      <td id="T_5a812_row1_col12" class="data row1 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row2_col0" class="data row2 col0" >Procter & Gamble Company (The)</td>
      <td id="T_5a812_row2_col1" class="data row2 col1" >PG</td>
      <td id="T_5a812_row2_col2" class="data row2 col2" >0.596851</td>
      <td id="T_5a812_row2_col3" class="data row2 col3" >0.664948</td>
      <td id="T_5a812_row2_col4" class="data row2 col4" >0.495519</td>
      <td id="T_5a812_row2_col5" class="data row2 col5" >0.567865</td>
      <td id="T_5a812_row2_col6" class="data row2 col6" >0.534565</td>
      <td id="T_5a812_row2_col7" class="data row2 col7" >0.465435</td>
      <td id="T_5a812_row2_col8" class="data row2 col8" >0.398357</td>
      <td id="T_5a812_row2_col9" class="data row2 col9" >0.601643</td>
      <td id="T_5a812_row2_col10" class="data row2 col10" >582</td>
      <td id="T_5a812_row2_col11" class="data row2 col11" >879</td>
      <td id="T_5a812_row2_col12" class="data row2 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row3_col0" class="data row3 col0" >Oracle Corporation</td>
      <td id="T_5a812_row3_col1" class="data row3 col1" >ORCL</td>
      <td id="T_5a812_row3_col2" class="data row3 col2" >0.564682</td>
      <td id="T_5a812_row3_col3" class="data row3 col3" >0.652361</td>
      <td id="T_5a812_row3_col4" class="data row3 col4" >0.390746</td>
      <td id="T_5a812_row3_col5" class="data row3 col5" >0.488746</td>
      <td id="T_5a812_row3_col6" class="data row3 col6" >0.532512</td>
      <td id="T_5a812_row3_col7" class="data row3 col7" >0.467488</td>
      <td id="T_5a812_row3_col8" class="data row3 col8" >0.318960</td>
      <td id="T_5a812_row3_col9" class="data row3 col9" >0.681040</td>
      <td id="T_5a812_row3_col10" class="data row3 col10" >466</td>
      <td id="T_5a812_row3_col11" class="data row3 col11" >995</td>
      <td id="T_5a812_row3_col12" class="data row3 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row4_col0" class="data row4 col0" >Coca-Cola Company (The)</td>
      <td id="T_5a812_row4_col1" class="data row4 col1" >KO</td>
      <td id="T_5a812_row4_col2" class="data row4 col2" >0.617385</td>
      <td id="T_5a812_row4_col3" class="data row4 col3" >0.649645</td>
      <td id="T_5a812_row4_col4" class="data row4 col4" >0.594805</td>
      <td id="T_5a812_row4_col5" class="data row4 col5" >0.621017</td>
      <td id="T_5a812_row4_col6" class="data row4 col6" >0.527036</td>
      <td id="T_5a812_row4_col7" class="data row4 col7" >0.472964</td>
      <td id="T_5a812_row4_col8" class="data row4 col8" >0.482546</td>
      <td id="T_5a812_row4_col9" class="data row4 col9" >0.517454</td>
      <td id="T_5a812_row4_col10" class="data row4 col10" >705</td>
      <td id="T_5a812_row4_col11" class="data row4 col11" >756</td>
      <td id="T_5a812_row4_col12" class="data row4 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row5_col0" class="data row5 col0" >International Business Machines</td>
      <td id="T_5a812_row5_col1" class="data row5 col1" >IBM</td>
      <td id="T_5a812_row5_col2" class="data row5 col2" >0.613963</td>
      <td id="T_5a812_row5_col3" class="data row5 col3" >0.646518</td>
      <td id="T_5a812_row5_col4" class="data row5 col4" >0.625159</td>
      <td id="T_5a812_row5_col5" class="data row5 col5" >0.635659</td>
      <td id="T_5a812_row5_col6" class="data row5 col6" >0.538672</td>
      <td id="T_5a812_row5_col7" class="data row5 col7" >0.461328</td>
      <td id="T_5a812_row5_col8" class="data row5 col8" >0.520876</td>
      <td id="T_5a812_row5_col9" class="data row5 col9" >0.479124</td>
      <td id="T_5a812_row5_col10" class="data row5 col10" >761</td>
      <td id="T_5a812_row5_col11" class="data row5 col11" >700</td>
      <td id="T_5a812_row5_col12" class="data row5 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row6_col0" class="data row6 col0" >Lockheed Martin Corporation</td>
      <td id="T_5a812_row6_col1" class="data row6 col1" >LMT</td>
      <td id="T_5a812_row6_col2" class="data row6 col2" >0.595483</td>
      <td id="T_5a812_row6_col3" class="data row6 col3" >0.644407</td>
      <td id="T_5a812_row6_col4" class="data row6 col4" >0.505236</td>
      <td id="T_5a812_row6_col5" class="data row6 col5" >0.566398</td>
      <td id="T_5a812_row6_col6" class="data row6 col6" >0.522930</td>
      <td id="T_5a812_row6_col7" class="data row6 col7" >0.477070</td>
      <td id="T_5a812_row6_col8" class="data row6 col8" >0.409993</td>
      <td id="T_5a812_row6_col9" class="data row6 col9" >0.590007</td>
      <td id="T_5a812_row6_col10" class="data row6 col10" >599</td>
      <td id="T_5a812_row6_col11" class="data row6 col11" >862</td>
      <td id="T_5a812_row6_col12" class="data row6 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row7_col0" class="data row7 col0" >Walmart Inc.</td>
      <td id="T_5a812_row7_col1" class="data row7 col1" >WMT</td>
      <td id="T_5a812_row7_col2" class="data row7 col2" >0.590007</td>
      <td id="T_5a812_row7_col3" class="data row7 col3" >0.643200</td>
      <td id="T_5a812_row7_col4" class="data row7 col4" >0.516710</td>
      <td id="T_5a812_row7_col5" class="data row7 col5" >0.573058</td>
      <td id="T_5a812_row7_col6" class="data row7 col6" >0.532512</td>
      <td id="T_5a812_row7_col7" class="data row7 col7" >0.467488</td>
      <td id="T_5a812_row7_col8" class="data row7 col8" >0.427789</td>
      <td id="T_5a812_row7_col9" class="data row7 col9" >0.572211</td>
      <td id="T_5a812_row7_col10" class="data row7 col10" >625</td>
      <td id="T_5a812_row7_col11" class="data row7 col11" >836</td>
      <td id="T_5a812_row7_col12" class="data row7 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row8_col0" class="data row8 col0" >Pepsico, Inc.</td>
      <td id="T_5a812_row8_col1" class="data row8 col1" >PEP</td>
      <td id="T_5a812_row8_col2" class="data row8 col2" >0.628337</td>
      <td id="T_5a812_row8_col3" class="data row8 col3" >0.642857</td>
      <td id="T_5a812_row8_col4" class="data row8 col4" >0.648755</td>
      <td id="T_5a812_row8_col5" class="data row8 col5" >0.645793</td>
      <td id="T_5a812_row8_col6" class="data row8 col6" >0.522245</td>
      <td id="T_5a812_row8_col7" class="data row8 col7" >0.477755</td>
      <td id="T_5a812_row8_col8" class="data row8 col8" >0.527036</td>
      <td id="T_5a812_row8_col9" class="data row8 col9" >0.472964</td>
      <td id="T_5a812_row8_col10" class="data row8 col10" >770</td>
      <td id="T_5a812_row8_col11" class="data row8 col11" >691</td>
      <td id="T_5a812_row8_col12" class="data row8 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row9_col0" class="data row9 col0" >Microsoft Corporation</td>
      <td id="T_5a812_row9_col1" class="data row9 col1" >MSFT</td>
      <td id="T_5a812_row9_col2" class="data row9 col2" >0.574264</td>
      <td id="T_5a812_row9_col3" class="data row9 col3" >0.634752</td>
      <td id="T_5a812_row9_col4" class="data row9 col4" >0.462532</td>
      <td id="T_5a812_row9_col5" class="data row9 col5" >0.535127</td>
      <td id="T_5a812_row9_col6" class="data row9 col6" >0.529774</td>
      <td id="T_5a812_row9_col7" class="data row9 col7" >0.470226</td>
      <td id="T_5a812_row9_col8" class="data row9 col8" >0.386037</td>
      <td id="T_5a812_row9_col9" class="data row9 col9" >0.613963</td>
      <td id="T_5a812_row9_col10" class="data row9 col10" >564</td>
      <td id="T_5a812_row9_col11" class="data row9 col11" >897</td>
      <td id="T_5a812_row9_col12" class="data row9 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row10_col0" class="data row10 col0" >Verizon Communications Inc.</td>
      <td id="T_5a812_row10_col1" class="data row10 col1" >VZ</td>
      <td id="T_5a812_row10_col2" class="data row10 col2" >0.618754</td>
      <td id="T_5a812_row10_col3" class="data row10 col3" >0.633540</td>
      <td id="T_5a812_row10_col4" class="data row10 col4" >0.559671</td>
      <td id="T_5a812_row10_col5" class="data row10 col5" >0.594319</td>
      <td id="T_5a812_row10_col6" class="data row10 col6" >0.498973</td>
      <td id="T_5a812_row10_col7" class="data row10 col7" >0.501027</td>
      <td id="T_5a812_row10_col8" class="data row10 col8" >0.440794</td>
      <td id="T_5a812_row10_col9" class="data row10 col9" >0.559206</td>
      <td id="T_5a812_row10_col10" class="data row10 col10" >644</td>
      <td id="T_5a812_row10_col11" class="data row10 col11" >817</td>
      <td id="T_5a812_row10_col12" class="data row10 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row11_col0" class="data row11 col0" >3M Company</td>
      <td id="T_5a812_row11_col1" class="data row11 col1" >MMM</td>
      <td id="T_5a812_row11_col2" class="data row11 col2" >0.632444</td>
      <td id="T_5a812_row11_col3" class="data row11 col3" >0.633420</td>
      <td id="T_5a812_row11_col4" class="data row11 col4" >0.658143</td>
      <td id="T_5a812_row11_col5" class="data row11 col5" >0.645545</td>
      <td id="T_5a812_row11_col6" class="data row11 col6" >0.508556</td>
      <td id="T_5a812_row11_col7" class="data row11 col7" >0.491444</td>
      <td id="T_5a812_row11_col8" class="data row11 col8" >0.528405</td>
      <td id="T_5a812_row11_col9" class="data row11 col9" >0.471595</td>
      <td id="T_5a812_row11_col10" class="data row11 col10" >772</td>
      <td id="T_5a812_row11_col11" class="data row11 col11" >689</td>
      <td id="T_5a812_row11_col12" class="data row11 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row12_col0" class="data row12 col0" >Exxon Mobil Corporation</td>
      <td id="T_5a812_row12_col1" class="data row12 col1" >XOM</td>
      <td id="T_5a812_row12_col2" class="data row12 col2" >0.592060</td>
      <td id="T_5a812_row12_col3" class="data row12 col3" >0.633333</td>
      <td id="T_5a812_row12_col4" class="data row12 col4" >0.502646</td>
      <td id="T_5a812_row12_col5" class="data row12 col5" >0.560472</td>
      <td id="T_5a812_row12_col6" class="data row12 col6" >0.517454</td>
      <td id="T_5a812_row12_col7" class="data row12 col7" >0.482546</td>
      <td id="T_5a812_row12_col8" class="data row12 col8" >0.410678</td>
      <td id="T_5a812_row12_col9" class="data row12 col9" >0.589322</td>
      <td id="T_5a812_row12_col10" class="data row12 col10" >600</td>
      <td id="T_5a812_row12_col11" class="data row12 col11" >861</td>
      <td id="T_5a812_row12_col12" class="data row12 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row13_col0" class="data row13 col0" >Johnson & Johnson</td>
      <td id="T_5a812_row13_col1" class="data row13 col1" >JNJ</td>
      <td id="T_5a812_row13_col2" class="data row13 col2" >0.598905</td>
      <td id="T_5a812_row13_col3" class="data row13 col3" >0.632686</td>
      <td id="T_5a812_row13_col4" class="data row13 col4" >0.521333</td>
      <td id="T_5a812_row13_col5" class="data row13 col5" >0.571637</td>
      <td id="T_5a812_row13_col6" class="data row13 col6" >0.513347</td>
      <td id="T_5a812_row13_col7" class="data row13 col7" >0.486653</td>
      <td id="T_5a812_row13_col8" class="data row13 col8" >0.422998</td>
      <td id="T_5a812_row13_col9" class="data row13 col9" >0.577002</td>
      <td id="T_5a812_row13_col10" class="data row13 col10" >618</td>
      <td id="T_5a812_row13_col11" class="data row13 col11" >843</td>
      <td id="T_5a812_row13_col12" class="data row13 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row14_col0" class="data row14 col0" >Visa Inc.</td>
      <td id="T_5a812_row14_col1" class="data row14 col1" >V</td>
      <td id="T_5a812_row14_col2" class="data row14 col2" >0.587269</td>
      <td id="T_5a812_row14_col3" class="data row14 col3" >0.631124</td>
      <td id="T_5a812_row14_col4" class="data row14 col4" >0.557962</td>
      <td id="T_5a812_row14_col5" class="data row14 col5" >0.592292</td>
      <td id="T_5a812_row14_col6" class="data row14 col6" >0.537303</td>
      <td id="T_5a812_row14_col7" class="data row14 col7" >0.462697</td>
      <td id="T_5a812_row14_col8" class="data row14 col8" >0.475017</td>
      <td id="T_5a812_row14_col9" class="data row14 col9" >0.524983</td>
      <td id="T_5a812_row14_col10" class="data row14 col10" >694</td>
      <td id="T_5a812_row14_col11" class="data row14 col11" >767</td>
      <td id="T_5a812_row14_col12" class="data row14 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row15_col0" class="data row15 col0" >Pfizer, Inc.</td>
      <td id="T_5a812_row15_col1" class="data row15 col1" >PFE</td>
      <td id="T_5a812_row15_col2" class="data row15 col2" >0.632444</td>
      <td id="T_5a812_row15_col3" class="data row15 col3" >0.629752</td>
      <td id="T_5a812_row15_col4" class="data row15 col4" >0.548991</td>
      <td id="T_5a812_row15_col5" class="data row15 col5" >0.586605</td>
      <td id="T_5a812_row15_col6" class="data row15 col6" >0.475017</td>
      <td id="T_5a812_row15_col7" class="data row15 col7" >0.524983</td>
      <td id="T_5a812_row15_col8" class="data row15 col8" >0.414100</td>
      <td id="T_5a812_row15_col9" class="data row15 col9" >0.585900</td>
      <td id="T_5a812_row15_col10" class="data row15 col10" >605</td>
      <td id="T_5a812_row15_col11" class="data row15 col11" >856</td>
      <td id="T_5a812_row15_col12" class="data row15 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row16_col0" class="data row16 col0" >AT&T Inc.</td>
      <td id="T_5a812_row16_col1" class="data row16 col1" >T</td>
      <td id="T_5a812_row16_col2" class="data row16 col2" >0.587269</td>
      <td id="T_5a812_row16_col3" class="data row16 col3" >0.628521</td>
      <td id="T_5a812_row16_col4" class="data row16 col4" >0.476636</td>
      <td id="T_5a812_row16_col5" class="data row16 col5" >0.542141</td>
      <td id="T_5a812_row16_col6" class="data row16 col6" >0.512663</td>
      <td id="T_5a812_row16_col7" class="data row16 col7" >0.487337</td>
      <td id="T_5a812_row16_col8" class="data row16 col8" >0.388775</td>
      <td id="T_5a812_row16_col9" class="data row16 col9" >0.611225</td>
      <td id="T_5a812_row16_col10" class="data row16 col10" >568</td>
      <td id="T_5a812_row16_col11" class="data row16 col11" >893</td>
      <td id="T_5a812_row16_col12" class="data row16 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row17_col0" class="data row17 col0" >Amazon.com, Inc.</td>
      <td id="T_5a812_row17_col1" class="data row17 col1" >AMZN</td>
      <td id="T_5a812_row17_col2" class="data row17 col2" >0.560575</td>
      <td id="T_5a812_row17_col3" class="data row17 col3" >0.627660</td>
      <td id="T_5a812_row17_col4" class="data row17 col4" >0.387139</td>
      <td id="T_5a812_row17_col5" class="data row17 col5" >0.478896</td>
      <td id="T_5a812_row17_col6" class="data row17 col6" >0.521561</td>
      <td id="T_5a812_row17_col7" class="data row17 col7" >0.478439</td>
      <td id="T_5a812_row17_col8" class="data row17 col8" >0.321697</td>
      <td id="T_5a812_row17_col9" class="data row17 col9" >0.678303</td>
      <td id="T_5a812_row17_col10" class="data row17 col10" >470</td>
      <td id="T_5a812_row17_col11" class="data row17 col11" >991</td>
      <td id="T_5a812_row17_col12" class="data row17 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row18_col0" class="data row18 col0" >Wells Fargo & Company</td>
      <td id="T_5a812_row18_col1" class="data row18 col1" >WFC</td>
      <td id="T_5a812_row18_col2" class="data row18 col2" >0.599589</td>
      <td id="T_5a812_row18_col3" class="data row18 col3" >0.626935</td>
      <td id="T_5a812_row18_col4" class="data row18 col4" >0.540721</td>
      <td id="T_5a812_row18_col5" class="data row18 col5" >0.580645</td>
      <td id="T_5a812_row18_col6" class="data row18 col6" >0.512663</td>
      <td id="T_5a812_row18_col7" class="data row18 col7" >0.487337</td>
      <td id="T_5a812_row18_col8" class="data row18 col8" >0.442163</td>
      <td id="T_5a812_row18_col9" class="data row18 col9" >0.557837</td>
      <td id="T_5a812_row18_col10" class="data row18 col10" >646</td>
      <td id="T_5a812_row18_col11" class="data row18 col11" >815</td>
      <td id="T_5a812_row18_col12" class="data row18 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row19_col0" class="data row19 col0" >Cisco Systems, Inc.</td>
      <td id="T_5a812_row19_col1" class="data row19 col1" >CSCO</td>
      <td id="T_5a812_row19_col2" class="data row19 col2" >0.601643</td>
      <td id="T_5a812_row19_col3" class="data row19 col3" >0.625532</td>
      <td id="T_5a812_row19_col4" class="data row19 col4" >0.581028</td>
      <td id="T_5a812_row19_col5" class="data row19 col5" >0.602459</td>
      <td id="T_5a812_row19_col6" class="data row19 col6" >0.519507</td>
      <td id="T_5a812_row19_col7" class="data row19 col7" >0.480493</td>
      <td id="T_5a812_row19_col8" class="data row19 col8" >0.482546</td>
      <td id="T_5a812_row19_col9" class="data row19 col9" >0.517454</td>
      <td id="T_5a812_row19_col10" class="data row19 col10" >705</td>
      <td id="T_5a812_row19_col11" class="data row19 col11" >756</td>
      <td id="T_5a812_row19_col12" class="data row19 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row20_col0" class="data row20 col0" >Intel Corporation</td>
      <td id="T_5a812_row20_col1" class="data row20 col1" >INTC</td>
      <td id="T_5a812_row20_col2" class="data row20 col2" >0.609856</td>
      <td id="T_5a812_row20_col3" class="data row20 col3" >0.619414</td>
      <td id="T_5a812_row20_col4" class="data row20 col4" >0.554483</td>
      <td id="T_5a812_row20_col5" class="data row20 col5" >0.585153</td>
      <td id="T_5a812_row20_col6" class="data row20 col6" >0.496235</td>
      <td id="T_5a812_row20_col7" class="data row20 col7" >0.503765</td>
      <td id="T_5a812_row20_col8" class="data row20 col8" >0.444216</td>
      <td id="T_5a812_row20_col9" class="data row20 col9" >0.555784</td>
      <td id="T_5a812_row20_col10" class="data row20 col10" >649</td>
      <td id="T_5a812_row20_col11" class="data row20 col11" >812</td>
      <td id="T_5a812_row20_col12" class="data row20 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row21_col0" class="data row21 col0" >Goldman Sachs Group, Inc. (The)</td>
      <td id="T_5a812_row21_col1" class="data row21 col1" >GS</td>
      <td id="T_5a812_row21_col2" class="data row21 col2" >0.581109</td>
      <td id="T_5a812_row21_col3" class="data row21 col3" >0.612557</td>
      <td id="T_5a812_row21_col4" class="data row21 col4" >0.527009</td>
      <td id="T_5a812_row21_col5" class="data row21 col5" >0.566572</td>
      <td id="T_5a812_row21_col6" class="data row21 col6" >0.519507</td>
      <td id="T_5a812_row21_col7" class="data row21 col7" >0.480493</td>
      <td id="T_5a812_row21_col8" class="data row21 col8" >0.446954</td>
      <td id="T_5a812_row21_col9" class="data row21 col9" >0.553046</td>
      <td id="T_5a812_row21_col10" class="data row21 col10" >653</td>
      <td id="T_5a812_row21_col11" class="data row21 col11" >808</td>
      <td id="T_5a812_row21_col12" class="data row21 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row22_col0" class="data row22 col0" >UnitedHealth Group Incorporated</td>
      <td id="T_5a812_row22_col1" class="data row22 col1" >UNH</td>
      <td id="T_5a812_row22_col2" class="data row22 col2" >0.566051</td>
      <td id="T_5a812_row22_col3" class="data row22 col3" >0.612313</td>
      <td id="T_5a812_row22_col4" class="data row22 col4" >0.478544</td>
      <td id="T_5a812_row22_col5" class="data row22 col5" >0.537226</td>
      <td id="T_5a812_row22_col6" class="data row22 col6" >0.526352</td>
      <td id="T_5a812_row22_col7" class="data row22 col7" >0.473648</td>
      <td id="T_5a812_row22_col8" class="data row22 col8" >0.411362</td>
      <td id="T_5a812_row22_col9" class="data row22 col9" >0.588638</td>
      <td id="T_5a812_row22_col10" class="data row22 col10" >601</td>
      <td id="T_5a812_row22_col11" class="data row22 col11" >860</td>
      <td id="T_5a812_row22_col12" class="data row22 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row23_col0" class="data row23 col0" >Boeing Company (The)</td>
      <td id="T_5a812_row23_col1" class="data row23 col1" >BA</td>
      <td id="T_5a812_row23_col2" class="data row23 col2" >0.616701</td>
      <td id="T_5a812_row23_col3" class="data row23 col3" >0.606757</td>
      <td id="T_5a812_row23_col4" class="data row23 col4" >0.625348</td>
      <td id="T_5a812_row23_col5" class="data row23 col5" >0.615912</td>
      <td id="T_5a812_row23_col6" class="data row23 col6" >0.491444</td>
      <td id="T_5a812_row23_col7" class="data row23 col7" >0.508556</td>
      <td id="T_5a812_row23_col8" class="data row23 col8" >0.506502</td>
      <td id="T_5a812_row23_col9" class="data row23 col9" >0.493498</td>
      <td id="T_5a812_row23_col10" class="data row23 col10" >740</td>
      <td id="T_5a812_row23_col11" class="data row23 col11" >721</td>
      <td id="T_5a812_row23_col12" class="data row23 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row24_col0" class="data row24 col0" >Walt Disney Company (The)</td>
      <td id="T_5a812_row24_col1" class="data row24 col1" >DIS</td>
      <td id="T_5a812_row24_col2" class="data row24 col2" >0.610541</td>
      <td id="T_5a812_row24_col3" class="data row24 col3" >0.602963</td>
      <td id="T_5a812_row24_col4" class="data row24 col4" >0.574859</td>
      <td id="T_5a812_row24_col5" class="data row24 col5" >0.588576</td>
      <td id="T_5a812_row24_col6" class="data row24 col6" >0.484600</td>
      <td id="T_5a812_row24_col7" class="data row24 col7" >0.515400</td>
      <td id="T_5a812_row24_col8" class="data row24 col8" >0.462012</td>
      <td id="T_5a812_row24_col9" class="data row24 col9" >0.537988</td>
      <td id="T_5a812_row24_col10" class="data row24 col10" >675</td>
      <td id="T_5a812_row24_col11" class="data row24 col11" >786</td>
      <td id="T_5a812_row24_col12" class="data row24 col12" >1461</td>
    </tr>
    <tr>
      <td id="T_5a812_row25_col0" class="data row25 col0" >Meta Platforms, Inc.</td>
      <td id="T_5a812_row25_col1" class="data row25 col1" >META</td>
      <td id="T_5a812_row25_col2" class="data row25 col2" >0.540603</td>
      <td id="T_5a812_row25_col3" class="data row25 col3" >0.578082</td>
      <td id="T_5a812_row25_col4" class="data row25 col4" >0.465784</td>
      <td id="T_5a812_row25_col5" class="data row25 col5" >0.515892</td>
      <td id="T_5a812_row25_col6" class="data row25 col6" >0.525522</td>
      <td id="T_5a812_row25_col7" class="data row25 col7" >0.474478</td>
      <td id="T_5a812_row25_col8" class="data row25 col8" >0.423434</td>
      <td id="T_5a812_row25_col9" class="data row25 col9" >0.576566</td>
      <td id="T_5a812_row25_col10" class="data row25 col10" >365</td>
      <td id="T_5a812_row25_col11" class="data row25 col11" >497</td>
      <td id="T_5a812_row25_col12" class="data row25 col12" >862</td>
    </tr>
  </tbody>
</table>
