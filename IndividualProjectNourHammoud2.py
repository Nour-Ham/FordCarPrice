import streamlit as st 
#import plotly.expressiconda  as px  # interactive charts
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressorss
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
df = pd.read_csv("C:/Users/nourh/OneDrive/Desktop/streamlitapp/ford.csv")

############################################## STREAMLIT ############################################3
st.set_page_config(
    page_title="Ford Price Prediction",
    layout ="wide")
    
# ----------- Sidebar-----------------
def main():
    page = st.sidebar.selectbox(
        "Select a Page",
        [
            "Homepage",
           "DescriptiveGeneral", 
           "DescriptiveSpecific",
           "Analysis",
           "Predictor"
        ]
    )

    if page == "Homepage":
        Homepage()
    if page == "DescriptiveGeneral":
        DescriptiveGeneral()
    if page == "DescriptiveSpecific":
        DescriptiveSpecific()
    if page == "Predictor":
        Predictor()
    if page == "Analysis":
        Analysis()

def Homepage():
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

def DescriptiveGeneral ():
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


            

def DescriptiveSpecific():
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


      
   
def Analysis():              
    ################################# ML CODE##############################
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
    dummies4 = pd.get_dummies(df.year)
#Join new columns in one dataset
    df2 = pd.concat([df,dummies,dummies2,dummies3, dummies4],axis=1)
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
    #@st.cache()
    #def model_train():
     #   all_models = [LinearRegression(),RandomForestRegressor(),DecisionTreeRegressor()]
      #  scores = []
       # for i in all_models:
        #    model = i
        #    model.fit(x_train,y_train)
        #    y_predicted = model.predict(x_test)
        #    mse = mean_squared_error(y_test,y_predicted)
        ##    mae = mean_absolute_error(y_test,y_predicted)
        #    scores.append({
        #        'model': i,
        #        'mean_squared_error':mse,
        #        'mean_absolute_error':mae
   # })
    #    return pd.DataFrame(scores,columns=['model','mean_squared_error','mean_absolute_error'])
    #model_train()
    #st.text(model_train())
    
   # DTR = DecisionTreeRegressor()
   # DTR.fit(x_train,y_train)
   # y_predicted_DTR= DTR.predict(x_test)
   # mse_DTR = mean_squared_error(y_test,y_predicted_DTR)
   # mae_DTR = mean_absolute_error(y_test,y_predicted_DTR)
        
    #LR = LinearRegression()
    #LR.fit(x_train,y_train)        
    #y_predicted_LR= LR.predict(x_test)
    
    #mse_LR = mean_squared_error(y_test,y_predicted_LR)
    #mae_LR = mean_absolute_error(y_test,y_predicted_LR)

    #RFR = RandomForestRegressor()
    #RFR.fit(x_train,y_train)        
    #y_predicted_RFR= RFR.predict(x_test)
    
    #mse_RFR = mean_squared_error(y_test,y_predicted_RFR)
    #mae_RFR = mean_absolute_error(y_test,y_predicted_RFR)
    
    
    #kpi3,kpi4,kpi1,kpi2,kpi5,kpi6= st.columns(6)
    
   # kpi3.metric("MSE of DTR Model",round(mse_DTR, 4))
 
    #kpi4.metric("MAE of DTR Model",round(mae_DTR,4))
   
  #  kpi1.metric("MSE of LR Model x10^16",int(mse_LR/1000000000000000))

   # kpi2.metric("MAE of LR Model x 10^4",int(mae_LR/10000))
    
   # kpi5.metric("MSE of RFR Model",round(mse_RFR, 4))
    
   # kpi5.metric("MAE of RFR Model",round(mae_RFR, 4))
        
 
    Plot = st.radio("Model Plots", ["DecisionTreeRegressor","LinearRegression"])
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center}</style>', unsafe_allow_html=True)
    st.markdown("**Decision Tree Regressor have the lowest error and has a better fit**")
    st.markdown("**It will be used in our prediction**")  
         
    if Plot == "DecisionTreeRegressor":
        st.pyplot(sns.scatterplot(x = y_test,y = y_predicted_DTR).figure, figsize= (2, 2))
    if Plot == "LinearRegression": 
        st.pyplot(sns.scatterplot(x = y_test,y = y_predicted_LR).figure, figsize= (2, 2))
 
def Predictor():
    ###################################### Creating the model ####################
    ## Title
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
    mms = MinMaxScaler()
    inputs = ['year','price','mileage','tax','mpg','engineSize']
    df[inputs] = mms.fit_transform(df[inputs])
#Variables
    x = df.drop('price',axis=1)
    y= df.price
#Splitting data 
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
    DTR = DecisionTreeRegressor()
    DTR.fit(x_train,y_train)
    
    @st.cache() 
    def predicted_predicted():
        LR = LinearRegression()
        LR.fit(x_train,y_train)        
        return LR
    predicted_predicted
    # pickling the model
    import pickle
    pickle_out = open("LR.pkl", "wb")
    pickle.dump(LR, pickle_out)
    pickle_out.close()
    # loading in the model to predict on the data
    pickle_in = open('LR.pkl', 'rb')
    LR= pickle.load(pickle_in)

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
    prediction = LR.predict(df)   

    # If button is pressed
    if st.button("Get me the Price"):
         result = prediction
         st.success('The Car is for $ {}'.format(result))
   
if __name__ == "__main__":
   main() 
   

# ---- HIDE STREAMLIT STYLE ----
#hide_st_style = """
#                <style>
 #           MainMenu {visibility: hidden;}
 #           footer {visibility: hidden;}
 #           header {visibility: hidden;}
 #           </style>
 #           """
#st.markdown(hide_st_style, unsafe_allow_html=True)


