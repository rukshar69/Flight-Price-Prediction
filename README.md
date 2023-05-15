# Flight-Price-Prediction

In the [Streamlit app](https://rukshar69-flight-price-predi-streamlit-flight-prediction-ch3wai.streamlit.app/), we perform EDA on a [dataset](https://github.com/rukshar69/Flight-Price-Prediction/blob/main/Flight%20Dataset/Data_Train.xlsx) containing *Indian* flight info. such as: source, destination,
arrival, departure, duration time, intermediate stoppage, price of ticket, etc. We have trained a **[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)** using this data and the model predicts the **price** of a ticket after providing information about the flight. A dedicated page to predict ticket price is added in the streamlit app.

