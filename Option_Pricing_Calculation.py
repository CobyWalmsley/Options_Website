import streamlit as st
from streamlit.components.v1 import html
import base64
import os
import math
import pandas as pd
from scipy.stats import norm
import math
from matplotlib import pyplot as plt
import numpy as np

import yfinance as yf
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.interpolate import griddata


def e_call_price(stock_price,strike_price,time,risk_free_rate,vol):
    d1 = (math.log(stock_price/strike_price)+(time*(risk_free_rate+((vol**2)/2))))/(vol*math.sqrt(time))
    d2 = d1-(vol*math.sqrt(time))
    call_price = (stock_price*norm.cdf(d1))-(strike_price*math.exp(-risk_free_rate*time))*norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = math.exp(-((d1**2)/2))/(math.sqrt(2*math.pi)*stock_price*vol*math.sqrt(time))
    N_prime = (math.exp(-((d1**2)/2)))/math.sqrt(2*math.pi)
    theta = ((-stock_price*N_prime*vol/(2*math.sqrt(time)))-((risk_free_rate*strike_price*math.exp(-risk_free_rate*time))*norm.cdf(d2)))/365
    vega = (stock_price*N_prime*math.sqrt(time))/100
    rho = (strike_price*time*math.exp(-risk_free_rate*time)*norm.cdf(d2))/100
    return(call_price,delta,gamma,theta,vega,rho)

def e_put_price(stock_price,strike_price,time,risk_free_rate,vol):
    d1 = (math.log(stock_price/strike_price)+(time*(risk_free_rate+((vol**2)/2))))/(vol*math.sqrt(time))
    d2 = d1-(vol*math.sqrt(time))
    put_price = ((strike_price*math.exp(-risk_free_rate*time))*norm.cdf(-d2))-stock_price*norm.cdf(-d1)
    delta = norm.cdf(d1)-1
    gamma = math.exp(-((d1**2)/2))/(math.sqrt(2*math.pi)*stock_price*vol*math.sqrt(time))
    N_prime = (math.exp(-((d1**2)/2)))/math.sqrt(2*math.pi)
    theta = ((-stock_price*N_prime*vol/(2*math.sqrt(time)))+((risk_free_rate*strike_price*math.exp(-risk_free_rate*time))*norm.cdf(-d2)))/365
    vega = (stock_price*N_prime*math.sqrt(time))/100
    rho = (-strike_price*time*math.exp(-risk_free_rate*time)*norm.cdf(-d2))/100
    return(put_price,delta,gamma,theta,vega,rho)


def e_call_vol(stock_price,strike_price,time,risk_free_rate,call_price):
    errors=[]
    for n in range(1,3000):
        vol=n/1000
        d1 = (math.log(stock_price/strike_price)+(time*(risk_free_rate+((vol**2)/2))))/(vol*math.sqrt(time))
        d2 = d1-(vol*math.sqrt(time))
        call_price_guess = (stock_price*norm.cdf(d1))-(strike_price*math.exp(-risk_free_rate*time))*norm.cdf(d2)
        error = call_price_guess-call_price
        errors.append(error)
        if len(errors)>3:
            if abs(errors[-2])<abs(errors[-1]):
                if abs(errors[-3])>abs(errors[-2]):
                    return(vol-.001)

def e_put_vol(stock_price,strike_price,time,risk_free_rate,put_price):
    errors=[]
    for n in range(1,3000):
        vol=n/1000
        d1 = (math.log(stock_price/strike_price)+(time*(risk_free_rate+((vol**2)/2))))/(vol*math.sqrt(time))
        d2 = d1-(vol*math.sqrt(time))
        put_price_guess = ((strike_price*math.exp(-risk_free_rate*time))*norm.cdf(-d2))-stock_price*norm.cdf(-d1)
        error = put_price_guess-put_price
        errors.append(error)
        if len(errors)>3:
            if abs(errors[-2])<abs(errors[-1]):
                if abs(errors[-3])>abs(errors[-2]):
                    return(vol-.001)

def grab_options_data(tick,date,risk_free_rate,contract_type):
    expiration = datetime.strptime(date, "%Y-%m-%d")
    today = datetime.today()
    days_to_expiry = (expiration - today).days  
    time_horizon = days_to_expiry / 365 
    ticker = yf.Ticker(tick)
    chain = ticker.option_chain(date)
    expiration = datetime.strptime(date, "%Y-%m-%d")
    today = datetime.today()
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    days_to_expiry = (expiration - today).days  
    time_horizon = days_to_expiry / 365 
    if contract_type=='Call':
        frame = pd.DataFrame({"Strike":chain.calls['strike'],"Price":chain.calls['ask'],"Time":time_horizon,'Rate':risk_free_rate,'Stock Price':current_price})
    if contract_type=='Put':
        frame = pd.DataFrame({"Strike":chain.puts['strike'],"Price":chain.puts['ask'],"Time":time_horizon,'Rate':risk_free_rate,'Stock Price':current_price})
    last_zero_index = frame[frame['Price'] == 0].index.max()
    return(frame)



def get_call_frame(tick,risk_free_rate,contract_type):
    ticker = yf.Ticker(tick)
    dates = ticker.options
    frame_list = []
    for date in dates:
        frame=grab_options_data(tick,date,risk_free_rate,contract_type)
        frame_list.append(frame)
    
    final_frame = pd.concat(frame_list)
    final_frame = final_frame[final_frame['Price'] > 0]
    final_frame = final_frame[final_frame['Time'] > 0]
    if contract_type=='Call':
        final_frame['Vol'] = final_frame.apply(lambda row:e_call_vol(row['Stock Price'],row['Strike'],row['Time'],row['Rate'],row['Price']),axis=1)
    if contract_type=='Put':
        final_frame['Vol'] = final_frame.apply(lambda row:e_put_vol(row['Stock Price'],row['Strike'],row['Time'],row['Rate'],row['Price']),axis=1)
    return(final_frame)
   
def display_vol_surface(ticker,rate,contract_type):
    frame = get_call_frame(ticker,rate,contract_type)
    clean_df = pd.DataFrame({'Time':frame['Time'],'Vol':frame['Vol'],'Strike':frame['Strike']})
    x = clean_df['Strike'].values
    y = clean_df['Time'].values
    z = clean_df['Vol'].values

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi, yi), method='linear')  


    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xi, yi, zi, cmap='viridis', edgecolor='k')

    ax.set_xlabel('Strike')
    ax.set_ylabel('Time')
    ax.set_zlabel('Implied Volatility')
    plt.title('Volatility Surface')
    plt.show()

#display_vol_surface('AI',.03,'Call')


def binomial_model(t_days,vol,risk_free_rate,stock_price,strike_price,contract_type):
    d_t = 1/365
    u = math.exp(vol*math.sqrt(d_t))
    d = 1/u
    p = (math.exp(risk_free_rate*d_t)-d)/(u-d)
    print(p)
    print(1-p)
    layer_list = []
    option_list = []
    for n in range(1,t_days+1):
        vector=np.ones(n)
        layer_list.append(vector.copy())
        option_list.append(vector.copy())
    
    layer_list[0][0]=stock_price
    for layer in range(len(layer_list)-1):
        for price in range(len(layer_list[layer])):
            layer_list[layer+1][price] = layer_list[layer][price]+u
        layer_list[layer+1][-1]=layer_list[layer][-1]-d

    if contract_type=='Call':
        option_list[-1] = strike_price-layer_list[-1]
        option_list[-1][option_list[-1]<0]=0
    
    if contract_type=='Put':
        option_list[-1] = layer_list[-1]-strike_price
        option_list[-1][option_list[-1]<0]=0
    
    for layer in range(len(option_list)-2,-1,-1):
        for price in range(len(option_list[layer])):
            exp_value = math.exp(-(risk_free_rate*d_t))*   ((p*option_list[layer+1][price])+((1-p)*option_list[layer+1][price+1]))
            if contract_type=='Put':
                int_value = strike_price-layer_list[layer][price]
            if contract_type=='Call':
                int_value = layer_list[layer][price]-strike_price
            option_list[layer][price]=max(exp_value,int_value)
    return(option_list[0])


st.set_page_config(layout="wide")

# ---- Helper to load and encode local image ----
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/png;base64,{encoded}"

# Make sure the file is in the same folder
img_path = "lehigh logo.png"  # You can also do "images/lehigh_logo.png"
if not os.path.exists(img_path):
    st.error(f"Image file not found: {img_path}")
else:
    img_base64_str = get_base64_image(img_path)

    # ---- Sticky banner injected with base64 image ----
    st.markdown(f"""
        <style>
        .block-container {{
            padding-top: 6rem !important;
        }}
        .lehigh-sticky-banner {{
            position: fixed;
            top: 0;
            left: 16rem;
            right: 0;
            z-index: 9999;
            background-color: #5D432C;
            color: white;
            padding: 18px 28px;
            display: flex;
            align-items: center;
            gap: 18px;
            font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
            border-bottom: 2px solid #4A3420;
        }}
        .lehigh-sticky-banner img {{
            height: 56px;
        }}
        .lehigh-sticky-banner .title {{
            font-size: 26px;
            font-weight: 700;
            line-height: 1.2;
            margin: 0;
        }}
        .lehigh-sticky-banner .subtext {{
            font-size: 15px;
            margin: 2px 0 0 0;
            opacity: 0.9;
        }}
        </style>

        <div class="lehigh-sticky-banner">
            <img src="{img_base64_str}" alt="Lehigh Logo">
            <div>
                <div class="title">Options and Volatility | American and European Option Pricing + Greeks</div>
                <div class="subtext">Coby Walmsley • Lehigh University Masters in Financial Engineering</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

col1,col2 = st.columns(2)

with col2:
    with st.form("calc_form"):
        submitted = st.form_submit_button("Calculate Option Prices and Greeks")
        price = st.number_input("Stock Price",min_value=0)
        strike = st.number_input("Strike Price",min_value=0)
        vol = st.number_input("Volatility")
        days_to_maturity = st.number_input("Days To Maturity", min_value=1, max_value=1000, step=1)
        risk_free_rate = st.number_input("Risk Free Rate",step=.001,value=.02)
        contract_type = st.selectbox("Contract Type",['Call','Put'],key="sector_select")
        
                                    #binomial_model(t_days,vol,risk_free_rate,stock_price,strike_price,contract_type)
                                    #e_call_price(stock_price,strike_price,time,risk_free_rate,vol)

        if submitted:
            try:
                with st.spinner("Calculating..."):
                    if contract_type=='Call':
                        e_price,delta,gamma,theta,vega,rho = e_call_price(price,strike,days_to_maturity/365,risk_free_rate,vol)
                    if contract_type=='Put':
                        e_price,delta,gamma,theta,vega,rho = e_put_price(price,strike,days_to_maturity/365,risk_free_rate,vol)
                    a_price = binomial_model(days_to_maturity,vol,risk_free_rate,price,strike,contract_type)

                    st.session_state.e_price = e_price
                    st.session_state.a_price = a_price[0]
                    st.session_state.delta = delta
                    st.session_state.gamma = gamma
                    st.session_state.theta = theta
                    st.session_state.vega = vega
                    st.session_state.rho=rho
                    st.session_state.contract_type = contract_type

            except:
                print("Invalid Calculation Data")

    with col1:
        if "e_price" in st.session_state:
            e_price = st.session_state.get("e_price")
            a_price = st.session_state.get("a_price")
            delta = st.session_state.get("delta")
            gamma=st.session_state.get("gamma")
            theta=st.session_state.get("theta")
            vega=st.session_state.get("vega")
            rho=st.session_state.get("rho")
            contract_type = st.session_state.get("contract_type")
            st.title("Option Prices")
            st.write(f"European {contract_type} Option Price: ${round(e_price,3)}")
            st.write(f"American {contract_type} Option Price: ${round(a_price,3)}")
            st.title("Greeks")
            st.write("Greeks Displayed Are Only Valid For European Options")
            st.write(f"Delta: {round(delta,4)}")
            st.write(f"Gamma: {round(gamma,4)}")
            st.write(f"Theta: {round(theta,4)}")
            st.write(f"Vega: {round(vega,4)}")
            st.write(f"Rho: {round(rho,4)}")


    



