import pickle
import pandas as pd
from datetime import datetime

def predict(model, dep_time:str, arrival_time: str,Total_stops:int, airline:str, source:str, destination:str):
    airline_dictionary = {
        'Jet Airways':0, 'IndiGo':0, 'Air India':0,
        'Multiple carriers':0, 'SpiceJet':0, 'Vistara':0,
        'GoAir':0, 'Multiple carriers Premium economy':0,
        'Jet Airways Business':0, 'Vistara Premium economy':0,
        'Trujet':0
    }

    source_dictionary = {
        'Delhi':0,'Kolkata':0,'Mumbai':0,'Chennai':0,
    }

    destination_dictionary = {
        'Cochin':0,'Delhi':0, 'Hyderabad':0, 'Kolkata':0,
    }
    #get journey, arrival, and duration time features
    Journey_day = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").day
    Journey_month = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").month

    Departure_hour = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").hour
    Departure_min = pd.to_datetime(dep_time,format="%Y-%m-%dT%H:%M").minute
    #print(Journey_day,' ', Journey_month, ' ', Departure_hour, ' ', Departure_min)

    Arrival_hour =  pd.to_datetime(arrival_time,format="%Y-%m-%dT%H:%M").hour
    Arrival_min =  pd.to_datetime(arrival_time,format="%Y-%m-%dT%H:%M").minute

    #Total_stops = int(request.form['stops'])
    #duration calculation
    arr_time = datetime.strptime(arrival_time, "%Y-%m-%d %H:%M")
    d_time = datetime.strptime(dep_time, "%Y-%m-%d %H:%M")
    duration = arr_time - d_time
    total_sec = duration.seconds + (duration.days*3600*24 )
    dur_hour, dur_min  = total_sec // 3600, (total_sec %3600)//60

    # dur_hour = abs(Arrival_hour-Departure_hour)
    # dur_min = abs(Arrival_min-Departure_min)

    #create airline feature
    try:
        airline_dictionary[airline] = 1
    except:
        pass 
    airline_dict_keys = list(airline_dictionary.keys())
    Jet_Airways = airline_dictionary[ airline_dict_keys[0]]
    IndiGo = airline_dictionary[airline_dict_keys[1]]
    Air_India = airline_dictionary[airline_dict_keys[2]]
    Multiple_carriers = airline_dictionary[airline_dict_keys[3]]
    SpiceJet =airline_dictionary[ airline_dict_keys[4]]
    Vistara =airline_dictionary[ airline_dict_keys[5]]
    GoAir =airline_dictionary[ airline_dict_keys[6]]
    Multiple_carriers_Premium_economy = airline_dictionary[airline_dict_keys[7]]
    Jet_Airways_Business =airline_dictionary[ airline_dict_keys[8]]
    Vistara_Premium_economy =airline_dictionary[ airline_dict_keys[9]]
    Trujet = airline_dictionary[airline_dict_keys[10]]


    try:
        source_dictionary[source] = 1
    except:
        pass 
    src_dict_keys = list(source_dictionary.keys())
    s_Delhi = source_dictionary[src_dict_keys[0]]
    s_Kolkata =  source_dictionary[src_dict_keys[1]]
    s_Mumbai =  source_dictionary[src_dict_keys[2]]
    s_Chennai =  source_dictionary[src_dict_keys[3]]

    try:
        destination_dictionary[destination] = 1
    except:
        pass 
    dest_dict_keys = list(destination_dictionary.keys())
    d_Cochin = destination_dictionary[dest_dict_keys[0]]
    d_Delhi = destination_dictionary[dest_dict_keys[1]]
    d_Hyderabad = destination_dictionary[dest_dict_keys[2]]
    d_Kolkata = destination_dictionary[dest_dict_keys[3]]

    #prediction
    output = model.predict([[Total_stops,
            Journey_day,
            Journey_month,
            Departure_hour,
            Departure_min,
            Arrival_hour,
            Arrival_min,
            dur_hour,
            dur_min,
            Air_India,
            GoAir,
            IndiGo,
            Jet_Airways,
            Jet_Airways_Business,
            Multiple_carriers,
            Multiple_carriers_Premium_economy,
            SpiceJet,
            Trujet,
            Vistara,
            Vistara_Premium_economy,
            s_Chennai,
            s_Delhi,
            s_Kolkata,
            s_Mumbai,
            d_Cochin,
            d_Delhi,
            d_Hyderabad,
            d_Kolkata]])
    
    output = round(output[0],2)
    return output

if __name__ == '__main__':
    #print('Hello Rukshar!!')
    model = pickle.load(open('model/flight_rf.pkl','rb'))
    price = predict(model,dep_time= '2020-09-24 13:22', arrival_time='2020-09-25 13:23', Total_stops=0,airline='IndiGo',
            source = 'Bangalore', destination='Delhi')
    