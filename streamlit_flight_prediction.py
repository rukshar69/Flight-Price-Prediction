import streamlit as st
import pandas as pd
import plotly.express as px

####Methods#######################
st.cache_data()
def load_data():
    train_data = pd.read_excel('Flight Dataset/Data_Train.xlsx')

    def newd(x):
        if x=='New Delhi':
            return 'Delhi'
        else:
            return x

    train_data['Destination'] = train_data['Destination'].apply(newd)
    return train_data

############################################
st.title('Flight Cost Prediction for India')
st.markdown('In this Streamlit app, we perform EDA on a dataset containing *Indian* flight info. such as: source, destination,\
         arrival, departure, duration time, intermediate stoppage, price of ticket, etc. We have trained a **RandomForestRegressor** using this data\
         and the model predicts the **price** of a ticket after providing information about the flight.')

st.header('EDA')

df = load_data() #load data

with st.expander("A Glimpse at the Data"):
    st.write(df.head())
    st.markdown('Here, **Price** is the target variable. We won\'t use *Route* and *Additional_info* columns during model training')

############## Destination Analysis ##################

with st.expander("Destination Analysis"):
    destination_counts = df['Destination'].value_counts()
    #converting destination_counts to pandas dataframe to use for plotting
    destination_counts_df = pd.DataFrame(destination_counts,)
    destination_counts_df['Destination_pct'] =  round((destination_counts_df['Destination'] / destination_counts_df['Destination'].sum()) * 100, 2)

    st.write(destination_counts_df)

    fig = px.bar(destination_counts_df, y='Destination_pct', labels={
                        "index": "Destination",
                        "Destination_pct": "Count(%)", 
                    })
    #fig.show()
    # Plot!
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('Maximum people are going to Cochin followed by Bangalore and then Delhi in our dataset. So, the top 3 destinations are:\
                - Cochin(42%)\
                - Bangalore(27%)\
                - Delhi(20%)\
                Kolkata receives the least traffic.')