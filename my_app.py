import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pmdarima as pm
from dateutil.relativedelta import relativedelta
import streamlit as st
import time


def clean(df):
    df = df.drop(51)
    df.drop(df.columns[0],axis=1,inplace=True)
    df.rename(columns={"الفترة":"year"}, inplace=True)
    df["year"] = pd.date_range(start='1970',end='2021', freq='Y')
    df.set_index('year', inplace=True)
    return df

def rename_indicies(df):
    a_to_e = {
    "متوسط": "middle",
    "إبتدائي": "primary",
    "ثانوي": "high"
    }

    df.rename(columns=a_to_e,level=0, inplace=True)

    sub_df_1 = df["primary"]
    sub_df_2 = df["middle"]
    sub_df_3 = df["high"]

    sub_df_1 = sub_df_1.reset_index()
    sub_df_2 = sub_df_2.reset_index()
    sub_df_3 = sub_df_3.reset_index()

    # rename headers
    # "m_primary", "m_middle", "m_high"

    a_to_e = {
         'ذكور ': "primary_males",
        'إناث ':"primary_females",
        'المجموع ': "primary_total"
    }

    sub_df_1.rename(columns=a_to_e, inplace=True)
    a_to_e = {
         'ذكور ': "middle_males",
        'إناث ':"middle_females",
        'المجموع ': "middle_total"
    }

    sub_df_2.rename(columns=a_to_e, inplace=True)
    sub_df_2.drop(columns=['year'],inplace=True)
    a_to_e = {
         'ذكور ': "high_males",
        'إناث ':"high_females",
        'المجموع ': "high_total"
    }
    sub_df_3.rename(columns=a_to_e, inplace=True)
    sub_df_3.drop(columns=['year'],inplace=True)

    cleaned_df = pd.concat([sub_df_1,sub_df_2,sub_df_3], axis='columns')
    cleaned_df.set_index('year', inplace=True)
    
    return cleaned_df

@st.cache(allow_output_mutation=True)
def fit(col):
    model = pm.auto_arima(col, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0, 
                      D=0, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      stepwise=True)
    time.sleep(10)
    return model

def forecast(df,choice, model, n_periods, title):
    # Forecast
    # INPUT
    fc, confint = model.predict(n_periods=n_periods, return_conf_int=True)
    index_of_fc = pd.date_range(df.index[-1], df.index[-1] + relativedelta(years=n_periods-1), freq='Y')

    # make series for plotting purpose
    fc_series = pd.Series(fc, index=index_of_fc)
    lower_series = pd.Series(confint[:, 0], index=index_of_fc)
    upper_series = pd.Series(confint[:, 1], index=index_of_fc)

    # Plot
    plt.plot(df[choice])
    plt.plot(fc_series, color='darkgreen')
    plt.fill_between(lower_series.index, 
                     lower_series, 
                     upper_series, 
                     color='k', alpha=.15)

    plt.title(title)
    return plt

st.title("Number of Students in Public Education")

# User should be able to load & browse the data

# Read data
df = pd.read_csv("data/SAMA_StatisticalReport_2021.csv", header=[0,1])
df = clean(df)
df = rename_indicies(df)
# st.dataframe(df)



# User should be able to plot a category
options = st.multiselect(
     'Please select a series to plot:',
     list(df.columns),
     list(df.columns))

st.write('You selected:', options)

st.line_chart(df[options])

# User should be able to train a model
# User Inputs:
option= st.selectbox("Choose a series to train the model on",
list(df.columns),
6)
years = st.slider('How many years to forecast in the future?', 1, 10, 5)

# Train:
with st.spinner('Training the model...'):
    model = fit(df[option])
st.success('Done!')


# Predict :
plt = forecast(df=df,
         choice=option, 
         model=model,
         n_periods=years,
         title=f"Forcast of Number of {option} Students by {2020+years}")

st.pyplot(plt)

# User should be able to predict using a trained model
