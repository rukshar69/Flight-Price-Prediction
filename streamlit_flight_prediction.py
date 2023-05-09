import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from datetime import datetime
import datetime
import pickle
from predict import predict

####Methods#######################
st.cache_data()
def load_data():
    train_data = pd.read_excel('Flight Dataset/Data_Train.xlsx')

    def newd(x):
        if x=='New Delhi':
            return 'Delhi'
        else:
            return x

    # Extract day and month columns from Date_of_journey column
    train_data['Destination'] = train_data['Destination'].apply(newd)
    train_data['Journey_day'] = pd.to_datetime(train_data['Date_of_Journey'],format='%d/%m/%Y').dt.day
    train_data['Journey_month'] = pd.to_datetime(train_data['Date_of_Journey'],format='%d/%m/%Y').dt.month

    train_data.drop('Date_of_Journey',inplace=True,axis=1)
    train_data.drop(['Route', 'Additional_Info'], axis=1, inplace=True)

    # Extracting hours and minutes from departure and arrival time
    train_data['Dep_hour'] = pd.to_datetime(train_data['Dep_Time']).dt.hour
    train_data['Dep_min'] = pd.to_datetime(train_data['Dep_Time']).dt.minute
    train_data.drop('Dep_Time',axis=1,inplace=True) #drop the departure time column
    train_data['Arrival_hour'] = pd.to_datetime(train_data['Arrival_Time']).dt.hour
    train_data['Arrival_min'] = pd.to_datetime(train_data['Arrival_Time']).dt.minute
    train_data.drop('Arrival_Time',axis=1,inplace=True) #drop the arrival time column

    # Dropping the Duration column and extracting important info(Hour and Minute) from it
    duration = list(train_data['Duration'])
    for i in range(len(duration)):
        if len(duration[i].split()) != 2:
            if 'h' in duration[i]:
                duration[i] = duration[i] + ' 0m'
            else:
                duration[i] = '0h ' + duration[i]
    duration_hour = []
    duration_min = []
    for i in duration:
        h,m = i.split()
        duration_hour.append(int(h[:-1]))
        duration_min.append(int(m[:-1]))
    train_data['Duration_hours'] = duration_hour
    train_data['Duration_mins'] = duration_min
    train_data.drop('Duration',axis=1,inplace=True)

    #change total_stops
    train_data['Total_Stops'].replace({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4},inplace=True)

    return train_data

st.cache_resource()
def load_model():
    model = pickle.load(open('model/flight_rf.pkl','rb'))
    return model

############################################
st.title('Flight Cost Prediction for India')
st.markdown('In this Streamlit app, we perform EDA on a [dataset](https://github.com/rukshar69/Flight-Price-Prediction/blob/main/Flight%20Dataset/Data_Train.xlsx) containing *Indian* flight info. such as: source, destination,\
         arrival, departure, duration time, intermediate stoppage, price of ticket, etc. We have trained a **[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)** using this data\
         and the model predicts the **price** of a ticket after providing information about the flight.')

options = ['EDA', 'Model Prediction']
choice = st.selectbox('Select an option', options)


#choice = 'EDA'
if choice == 'EDA':
    st.header('EDA')

    df = load_data() #load data

    with st.expander("A Glimpse at the Data"):
        st.write(df.head())
        st.markdown('Here, **Price** is the target variable. We won\'t use *Route* and *Additional_info* columns during model training')


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
        st.markdown('Maximum people are going to Cochin followed by Bangalore and then Delhi in our dataset. So, the top 3 destinations are:')
        st.markdown('- Cochin(42%)')
        st.markdown('- Bangalore(27%)')
        st.markdown('- Delhi(20%)')
        st.markdown ('Kolkata receives the least traffic.')

    with st.expander("Source Analysis"):
        source_counts =  pd.DataFrame(df['Source'].value_counts()); 
        source_counts['Source_pct'] = (source_counts['Source']/source_counts['Source'].sum())*100.0
        st.write(source_counts)
        fig = px.bar(source_counts, y='Source_pct', labels={
                        "index": "Source",
                        "Source_pct": "Count(%)", 
                    }, )
        fig.update_traces(marker_color='green')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('A plurality of flight passengers fly from **Delhi**, followed by **Kolkata** and **Bangalore**\
                    respectively. A lot more folks fly from Kolkata than to Kolkata. Delhi and Bangalore has comparatively high inbound and outbound traffic. ')

    with st.expander("Airline vs Price"):
        #sns.catplot(x='Airline',y='Price',data=df.sort_values('Price',ascending=False),kind='boxen',aspect=3,height=6)
        fig = px.box(df.sort_values('Price',ascending=False), x='Airline', y='Price')
        st.plotly_chart(fig,)
        st.markdown('From the plot, we can infer that **Jet Airways** business is the costliest airways while **Spicejet** is comparatively cheaper.')

    with st.expander("Source vs Price"):
        fig = px.box(df.sort_values('Price',ascending=False), x='Source',y='Price',)
        st.plotly_chart(fig)
        st.markdown('The plot says if you\'re going from **Bangalore** to anywhere you have to pay the highest amount of money')
        st.markdown('It\'s comparatively cheaper to travel from **Chennai**')

    with st.expander("Destination vs Price"):
        fig = px.box(df.sort_values('Price',ascending=False), x='Destination',y='Price',)
        st.plotly_chart(fig)
        st.markdown('The plot says if you are going to **Delhi** from anywhere, you have to pay the highest amount of money.')
        st.markdown('It\'s comparatively cheaper to travel to **Kolkata**')

    with st.expander("Total Stops"):
        stops_df = pd.DataFrame(df['Total_Stops'].value_counts())
        fig = px.bar(stops_df, y='Total_Stops', labels={
                            "index": "Number of Stops",
                            "Total_Stops": "Count", 
                        })
        st.plotly_chart(fig)
        st.markdown('Most flights have 1 stop. However, there are quite a number of flights that have no stops in the middle')

    with st.expander("Correlation Map of Features and Prices"):
        #st.write(df.corr())
        fig = px.imshow(df.corr())
        st.plotly_chart(fig)

elif choice == 'Model Prediction':
    st.header('Model Description and Prediction')
    with st.expander("Input features and Output"):

        st.markdown('Here we train a **Random Forest Regressor** model with **Price** as the target variable using the\
                    following input features:')
        
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('- Arrival time');st.markdown('- Departure time'); st.markdown('- Flight Duration')
        with col2:
            st.markdown('- Total stops');st.markdown('- Airline'); st.markdown('- Source'); st.markdown('- Destination')

    with st.expander('Feature Importance'):
        image = Image.open('images/feature_importance.png')
        st.image(image, caption='Feature Importance for Flight Price Prediction')
        st.markdown('- We have used [ExtraTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html) to \
                    determine this chart of feature importance')
        st.markdown('- Total_stops is the feature with the highest feature importance in deciding the Price')
        st.markdown('- After that Journey Day(date of departure) also plays a big role in deciding the Price. Prices are generally higher on weekends.')

    with st.expander('Hyperparameter Tuning'):
        st.markdown('We have performed *Hyperparameter Tuning* using [RandomizedSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html).\
                    RandomizedSearchCV also performs *k-fold cross validation*. The following hyperparameters were tuned:')
        st.markdown('- n_estimators: Number of trees in random forest')
        st.markdown('- max_features: Number of features to consider at every split')
        st.markdown('- max_depth: Maximum number of levels in tree')
        st.markdown('- min_samples_split: Minimum number of samples required to split a node')
        st.markdown('- min_samples_leaf: Minimum number of samples required at each leaf node')
        st.markdown('The best hyperparameters:')
        st.write({'n_estimators': 700,
                    'min_samples_split': 15,
                    'min_samples_leaf': 1,
                    'max_features': 'auto',
                    'max_depth': 20})
        
    with st.expander('Results on Test Data'):
        image = Image.open('images/residual_price.png')
        st.image(image, caption='Residual price(difference between prediction and actual price) vs frequency')
        st.markdown('As we can see that most of the residuals are 0, which means our model is *generalizing well*.')
        image = Image.open('images/scatter_plot_pred_truth.png')
        st.image(image, caption='Scatter plot prediction vs true prices of flights')
        st.markdown('Ideally, it should be a straight line but I guess this is close enough')
        st.markdown('r2 score:  :green[0.80880667420437]')

    st.write('### Flight Price Prediction:')
    with st.form("Flight_prediction_form"):
        st.write("Provide Inputs")

        sources = ['Delhi', 'Kolkata', 'Banglore', 'Mumbai', 'Chennai']
        src_choice = st.selectbox('Source', sources)

        destinations = ['Cochin',  'Banglore', 'Delhi','Hyderabad', 'Kolkata',]
        dest_choice = st.selectbox('Destination', destinations)

        stops = [0,1,2,3,4]
        stop_choice = st.selectbox('No. of Stops', stops)

        airlines = ['Jet Airways', 'IndiGo', 'Air India',
        'Multiple carriers', 'SpiceJet', 'Vistara',
        'GoAir', 'Multiple carriers Premium economy',
        'Jet Airways Business', 'Vistara Premium economy',
        'Trujet']
        airline_choice = st.selectbox('Which Airline?', airlines)

        today = datetime.date.today()
        departure_date = st.date_input('Departure date', today)
        departure_time = st.time_input('Departure time', datetime.time(8, 45))

        today = datetime.date.today()
        arrival_date = st.date_input('Arrival date', today)
        arrival_time = st.time_input('Arrival time', datetime.time(8, 45))
        
        # Every form must have a submit button.
        submitted = st.form_submit_button("Submit")
        if submitted:
            st.write("- Source:", src_choice,)
            st.write("- Destination:", dest_choice)
            st.write("- No. of Stops:", stop_choice)
            st.write("- Airline:", airline_choice)

            if arrival_date<departure_date:
                st.error('Arrival date must be later than Departure date!!!')
            elif arrival_date==departure_date and arrival_time < departure_time:
                st.error('Arrival time must be later than the departure time!!!')
            else:
                st.write("- Departure date:", (departure_date))
                st.write("- Departure time:", (departure_time))
                st.write("- Arrival date:", (departure_date))
                st.write("- Arrival time:", (departure_time))
                
    
    #st.write('outside form '+str(checkbox_val))