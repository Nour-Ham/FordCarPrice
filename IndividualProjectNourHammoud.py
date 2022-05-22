import streamlit as st
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
st.set_page_config(
    page_title="Ford Price Prediction",
    layout ="wide")
    

df = pd.read_csv("C:/Users/nourh/OneDrive/Desktop/streamlitapp/ford.csv")

st.markdown(
    '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@4.5.3/dist/css/bootstrap.min.css" integrity="sha384-TX8t27EcRE3e/ihU7zmQxVncDAy5uIKz4rEkgIXeMed4M0jlfIDPvg6uqKI2xXr2" crossorigin="anonymous">',
    unsafe_allow_html=True,
)
query_params = st.experimental_get_query_params()

tabs = ["Home", "DescriptiveGeneral", "DescriptiveSpecific", "Analysis", "Predictor"]
if "tab" in query_params:
    active_tab = query_params["tab"][0]
else:
    active_tab = "Home"

if active_tab not in tabs:
    st.experimental_set_query_params(tab="Home")
    active_tab = "Home"

li_items = "".join(
    f"""
    <li class="nav-item">
        <a class="nav-link{' active' if t==active_tab else ''}" href="/?tab={t}">{t}</a>
    </li>
    """
    for t in tabs
)
tabs_html = f"""
    <ul class="nav nav-tabs">
    {li_items}
    </ul>
"""

st.markdown(tabs_html, unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

if active_tab == "Home":
    st.title("""
        Ford Price Predictor""")
    spectra = st.file_uploader("Upload Your File!", type={"csv", "txt"})
    if spectra is not None:
        spectra_df = pd.read_csv(spectra)
        st.write(spectra_df.head())
    if spectra is not None:
        
        st.caption ("Data Types in our dataset:")
        st.caption("Categorical: model, transmission, fuel type")
        st.caption("Numerical: price, mileage, tax, mpg, engineSize,year")
        
elif active_tab == "DescriptiveGeneral":
    st.title("General Overview") 

# create Six columns
    kpi4, kpi5, kpi6 , kpi1, kpi2, kpi3= st.columns(6)

# fill in those six columns with respective metrics or KPIs
    
    x1= df['model'].value_counts().idxmax()
    x2 = df['fuelType'].value_counts().idxmax()
    x3 = df['transmission'].value_counts().idxmax()
    x4 = int(df['price'].mean())
    x5 = int(df['price'].max())
    x6 = int(df['price'].min())
    kpi1.metric("Frequent Model",x1)

    kpi2.metric("Frequent Fuel Type",x2)

    kpi3.metric("Frequent Transmission",x3)
    
    kpi4.metric("Average Car price",x4)
    kpi5.metric("Maximum Car price",x5)
    kpi6.metric("Minimum Car price",x6)
    

    ax1,ax2 = st.columns(2)
    with ax1:
        st.header("Price Distribution")
        option = st.radio("Plot",["Plot1","Plot2"])
        if option == "Plot1":  
            st.pyplot((sns.boxenplot(y=df.price , color = 'lightskyblue').figure))
        if option == "Plot2":
            st.pyplot((sns.distplot(x=df.price, color = 'lightskyblue').figure))
        st.caption("**Based on the above, it seems that it is more skewed toward the left**"
                   )       
    with ax2:
        st.header("Variable Frequency")
        option = st.radio("Variable:", ['model', 'Fuel Type','transmission'])
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
        if option == 'transmission':
            st.pyplot((sns.countplot(y='transmission', data=df, color = "lightskyblue")).figure)
            st.caption("**Cars with manual transmission are sold the most**")
        if option == 'model':
            st.pyplot((sns.countplot(y='model', data=df, color = "lightskyblue")).figure)
            st.caption("**Fiesta and Focus are the cars that are available the most**")

        if option == 'Fuel Type':
            st.pyplot((sns.countplot(y='fuelType', data=df,  color = "lightskyblue")).figure)
            st.caption("**Petrol and Diesel fuel type cars are sold the most**") 

elif active_tab == "DescriptiveSpecific":
    df = pd.read_csv("C:/Users/nourh/OneDrive/Desktop/streamlitapp/ford.csv")
    st.title("Variation Of Price with variables")
    ax1,ax2 = st.columns(2)

    with ax1: 
            selectbox = st.selectbox( "Price Correlation with",["Tax", "Mpg", "Mileage", "Year", "EngineSize"])
            if selectbox == "Tax":
                ax3 = sns.scatterplot(x=df.tax,y=df.price, color = "lightskyblue")
                st.pyplot(ax3.figure)
                st.caption("**There's no significant correlation between price and tax.**")
            if selectbox == "Mpg":
                ax4 = sns.scatterplot(x=df.mpg,y=df.price, color = "lightskyblue")
                st.pyplot(ax4.figure)
                st.caption("**Prices have a negative correlation with Mpg.**")
                st.caption("**As Mpg increases price decreases**.")
            if selectbox == "Mileage":
                ax5 = sns.scatterplot(x=df.mileage,y=df.price, color = "lightskyblue")
                st.pyplot(ax5.figure)
                st.caption("**Prices have a negative correlation with mileage**")
                st.caption("**As mileage increases price decreases**.")
            if selectbox == "Year":
                ax6 = sns.scatterplot(x=df.year,y=df.price, color = "lightskyblue")
                st.pyplot(ax6.figure)
                st.caption("**Prices have a positive correlation with the year.**")
                st.caption("**As the year of manufacturing the car increases, the car prices increase**")
            if selectbox == "EngineSize":
                ax7 = sns.scatterplot(x=df.engineSize,y=df.price, color = "lightskyblue")
                st.pyplot(ax7.figure)
                st.caption("**There's no significant correlation between price and engineSize.**")
    with ax2: 
        option = st.radio("Variable:", ['model', 'Fuel Type','transmission'])
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
        if option == 'model':           
            model_price = sns.barplot(x = 'model',y = 'price',data = df, color = 'lightskyblue')
            model_price.tick_params(axis='x',rotation=90)
            st.pyplot(model_price.figure)
            st.caption("**Cars with the highest price are Mustang and Edge. The highest the price the lower the number of cars available**")
            
        if option == 'Fuel Type':
           
                fueltype_price = sns.barplot(x = 'fuelType',y = 'price',data = df, color = 'lightskyblue')
                st.pyplot(fueltype_price.figure)
                st.caption("**Cars with the highest price are of hybrid fuel which is rare**")
                
        if option == 'transmission':
                transmission_price = sns.barplot(x = 'transmission',y = 'price',data = df, color = 'lightskyblue')
                st.pyplot(transmission_price.figure)
                st.caption("**Automatic and Semi-auto transmissions cars are the highest priced cars** ")


elif active_tab == "Analysis":
    df = pd.read_csv("C:/Users/nourh/OneDrive/Desktop/streamlitapp/ford.csv")
#Check if there are any null values.
    df.isnull().sum()
#Step 1:Get the IRQ
    IQR_price= df['price'].quantile(q=.75)- df['price'].quantile(q=.25)
#Step 2: Get The upper and lower bound for the price column 
    price_lower_bound =df['price'].quantile(q=.25) - 1.5 * IQR_price
    price_upper_bound = df['price'].quantile(q=.75)+ 1.5 * IQR_price
    print(price_lower_bound,price_upper_bound)

#Step 3: Remoce outliers from the price column 
    df = df[(df['price'] > price_lower_bound) & (df['price'] < price_upper_bound)]

#transform the categorical variables into dummy
    dummies = pd.get_dummies(df.model)
    dummies2 = pd.get_dummies(df.transmission)
    dummies3 = pd.get_dummies(df.fuelType)
#Join new columns in one dataset
    df2 = pd.concat([df,dummies,dummies2,dummies3],axis=1)
#Let's drop the old columns.
    df2 = df2.drop(['model','transmission','fuelType'],axis=1)
#Index Reseting
    df2 = df2.reset_index(drop=True)
    df=df2
#Scaling Data -  MinMaxScaler preserves the shape of the original distribution.
    mms = MinMaxScaler()
    inputs = ['price','year','mileage','tax','mpg','engineSize']
    df[inputs] = mms.fit_transform(df[inputs])
#Variables
    x = df.drop('price',axis=1)
    y= df.price
#Splitting data 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    @st.cache()
    def RFR():
        RF= RandomForestRegressor(n_estimators=10, min_samples_leaf=0.05)
        RF.fit(x_train, y_train)
        return RF.predict(x_test)
    @st.cache()
    def RFR_mse():
        RF= RandomForestRegressor(n_estimators=10, min_samples_leaf=0.05)
        RF.fit(x_train, y_train)
        y_pred_RFR = RF.predict(x_test)
        return mean_squared_error(y_test,y_pred_RFR)
    @st.cache()
    def RFR_mae():
        RF= RandomForestRegressor(n_estimators=10, min_samples_leaf=0.05)
        RF.fit(x_train, y_train)
        y_pred_RFR = RF.predict(x_test)
        return mean_absolute_error(y_test,y_pred_RFR)
    @st.cache()
    def DTR():
        DTR = DecisionTreeRegressor()
        DTR.fit(x_train,y_train)
        return DTR.predict(x_test)
    @st.cache()
    def DTR_mae():
        DTR = DecisionTreeRegressor()
        DTR.fit(x_train,y_train)
        y_pred_DTR = DTR.predict(x_test)
        return mean_absolute_error(y_test,y_pred_DTR) 
    @st.cache()
    def DTR_mse():
        DTR = DecisionTreeRegressor()
        DTR.fit(x_train,y_train)
        y_pred_DTR = DTR.predict(x_test)
        return mean_squared_error(y_test,y_pred_DTR)
    @st.cache() 
    def LR():
        LR = LinearRegression()
        LR.fit(x_train,y_train)        
        return LR.predict(x_test)
    @st.cache() 
    def LR_mse():
        LR = LinearRegression()
        LR.fit(x_train,y_train)        
        y_pred_LR= LR.predict(x_test)
        return mean_squared_error(y_test,y_pred_LR) 
    @st.cache() 
    def LR_mae():
        LR = LinearRegression()
        LR.fit(x_train,y_train)        
        y_pred_LR= LR.predict(x_test)
        return mean_absolute_error(y_test,y_pred_LR) 
    #int(DTR_mae())
    kpi3,kpi4,kpi1,kpi2,kpi5,kpi6= st.columns(6)
    
    kpi3.metric("MSE of DTR Model",round(DTR_mse(),4))
 
    kpi4.metric("MAE of DTR Model",round(DTR_mae(),4))
   
    kpi1.metric("MSE of LR Model 10^16",int(LR_mse()/1000000000000000))

    kpi2.metric("MAE of LR Model X 10^4 ",int(LR_mae()/10000))
    
    kpi5.metric("MSE of RFR Model",round(RFR_mse(),4))
    
    kpi6.metric("MAE of RFR Model",round(RFR_mae(),4))
    
    Plot = st.radio("Model Plots", ["RandomForestRegressor","DecisionTreeRegressor","LinearRegression"])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
         

    if Plot == "DecisionTreeRegressor":
       st.pyplot(sns.scatterplot(x = y_test,y = DTR()).figure, figsize= (2, 2))
    if Plot == "RandomForestRegressor":
        st.pyplot(sns.scatterplot(x = y_test,y = RFR()).figure, figsize= (2, 2))
    if Plot == "LinearRegression": 
       st.pyplot(sns.scatterplot(x = y_test,y = LR()).figure, figsize= (2, 2))
elif active_tab == "Predictor":
    st.header("Ford Cars Price Predictor")
    df = pd.read_csv("C:/Users/nourh/OneDrive/Desktop/streamlitapp/ford.csv")
#Check if there are any null values.
    df.isnull().sum()
#Step 1:Get the IRQ
    IQR_price= df['price'].quantile(q=.75)- df['price'].quantile(q=.25)
#Step 2: Get The upper and lower bound for the price column 
    price_lower_bound =df['price'].quantile(q=.25) - 1.5 * IQR_price
    price_upper_bound = df['price'].quantile(q=.75)+ 1.5 * IQR_price
    print(price_lower_bound,price_upper_bound)

#Step 3: Remoce outliers from the price column 
    df = df[(df['price'] > price_lower_bound) & (df['price'] < price_upper_bound)]

#transform the categorical variables into dummy
    dummies = pd.get_dummies(df.model)
    dummies2 = pd.get_dummies(df.transmission)
    dummies3 = pd.get_dummies(df.fuelType)
#Join new columns in one dataset
    df2 = pd.concat([df,dummies,dummies2,dummies3],axis=1)
#Let's drop the old columns.
    df2 = df2.drop(['model','transmission','fuelType'],axis=1)
#Index Reseting
    df2 = df2.reset_index(drop=True)
    df=df2
#Scaling Data -  MinMaxScaler preserves the shape of the original distribution.
#mms = MinMaxScaler()
#inputs = ['year','price','mileage','tax','mpg','engineSize']
#df[inputs] = mms.fit_transform(df[inputs])
#Variables
    x = df.drop('price',axis=1)
    y= df.price
#Splitting data 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    @st.cache(allow_output_mutation=True)
    def RFR():
        RFR= RandomForestRegressor(n_estimators=10, min_samples_leaf=0.05)
        RFR=RFR.fit(x, y)
        return RFR
    # loading in the model to predict on the data
    import pickle
    pickle_out = open("RFR.pkl", "wb")
    pickle.dump(RFR, pickle_out)
    pickle_out.close()
    # loading in the model to predict on the data    
    pickle_in = open('RFR.pkl', 'rb')
    RFR= pickle.load(pickle_in)
    
    ax1,ax2 = st.columns(2)
    with ax1:
            year = st.slider('Year The Car Manufatured', 1996,2020)
            model = st.selectbox('model',('Fiesta',' Fiesta','Focus','Kuga','EcoSport','C-MAX','Ka+','Mondeo','B-MAX',
                                  'S-MAX','Grand C-MAX','Galaxy','Edge','KA','Puma','Tourneo Custom','Grand Tourneo Connect','Mustang','Tourneo Connect',
                                  'Fusion','Streetka','Ranger','Transit Tourneo','Escort','Focus','Transit Tourneo'))
            transmission = st.selectbox('transmission', ('Manual','Automatic','Semi-Auto'))
        
            mileage = st.slider('How many mileage?', 0,18000) 
    with ax2:
        fuelType= st.selectbox('fuelType', ('Petrol','Diesel','Hybrid','Electric','Other'))
        tax= st.slider('Tax Amount', 0, 600)
        mpg = st.slider('MPG', 0, 250)
        engineSize =  st.slider('engineSize', 0, 5)
        
        data = {'year':[year],'model':[model],'transmission':[transmission], 
                'fuelType':[fuelType],'mileage':[mileage],'mpg':[mpg],'engineSize':[engineSize], 
                'tax':[tax],}
        features = pd.DataFrame(data)
        
        
      
    cars = pd.read_csv("C:/Users/nourh/OneDrive/Desktop/streamlitapp/ford.csv")
    cars.fillna(0, inplace=True)
    cars = cars.drop(columns=['price'])
    df = pd.concat([features,cars],axis=0)
    #df = input_df
    
    encode = ['model','transmission', 'fuelType']
    for col in encode: 
            dummy = pd.get_dummies(df[col], prefix=col )
            df = pd.concat([df,dummy], axis=1)
            del df[col]  
 
    df = df[:1]
    df.fillna(0, inplace=True)
    #st.write(df)
    features = ['Fiesta',' Fiesta','Focus','Kuga','EcoSport','C-MAX','Ka+','Mondeo','B-MAX',
                                  'S-MAX','Grand C-MAX','Galaxy','Edge','KA','Puma','Tourneo Custom','Grand Tourneo Connect','Mustang','Tourneo Connect',
                                  'Fusion','Streetka','Ranger','Transit Tourneo','Escort','Focus','Manual','Automatic','Semi-Auto',
                                  'Petrol','Diesel','Hybrid','Electric','Other','Transit Tourneo']
   
    features = pd.DataFrame ([features], columns= ['Fiesta',' Fiesta','Focus','Kuga','EcoSport','C-MAX','Ka+','Mondeo','B-MAX',
                                  'S-MAX','Grand C-MAX','Galaxy','Edge','KA','Puma','Tourneo Custom','Grand Tourneo Connect','Mustang','Tourneo Connect',
                                  'Fusion','Streetka','Ranger','Transit Tourneo','Escort','Focus','Manual','Automatic','Semi-Auto',
                                  'Petrol','Diesel','Hybrid','Electric','Other','Transit Tourneo'])
    
    df = df [features]
    #df = df[:1]
    df.fillna(0, inplace=True)
    #df.dropna()
    #st.write(df)
    #st.write(features)
    df = df.drop(columns = ['model_Fiesta'])
    #df = df.drop(df.loc[df.index==' Fiesta'].index)
    #st.write(df.columns)

    #df= df.reindex(df)
    #df = df.loc[:,~df.columns.duplicated()]
    prediction = RFR().predict(df)   

    # If button is pressed
    if st.button("Get me the Price"):
        result = prediction
        st.success('The Car is for $ {}'.format(result))
else:
    st.error("Something has gone terribly wrong.")


   

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
                <style>
            MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


