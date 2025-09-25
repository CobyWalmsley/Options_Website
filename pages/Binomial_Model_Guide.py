import streamlit as st
from streamlit.components.v1 import html
import base64
import os
import math

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
                <div class="title">Options and Volatility | Binomial Pricing Model Guide</div>
                <div class="subtext">Coby Walmsley • Lehigh University Masters in Financial Engineering</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.title('Binomial Option Pricing Model')
st.write('Why use the Binomial Option Pricing Model when the Black-Scholes Model Exists?')
st.write('The Black-Scholes Model can only be used for when the option is exercised on its expiration date, i.e., a European Option.')
st.write('But an American Option can be exercised on any date before the expiration date, making the Black-Scholes model useless.')
st.write('We need a new approach.')
st.title('Methodology:')
st.write('We have to start with what we know. We know that after a certain period of time, our stock could have lots of different possible prices.')
st.write('We know that every day (or other period of time), the stock can either move up or down. If we know (or forecast) the yearly volatility, we can calculate how much the stock is most likely to move in one time period.')
st.write('The factors for these move sizes can be calculated with these equations:')
st.image("u and d.png",use_container_width=True)
st.write('In these equations, u is the size of a move up and d is the size of a move down.')
st.write('We also need the the probability of the stock moving up or down in the calculated amounts. This is also proportional to volatility and called risk-neutral probability.')
st.image("p_star.png",use_container_width=True)
st.write('p is the probability the stock moves up, so 1-p is the probability the stock moves down.')
st.write('These odds often work out to be nearly 50/50.')
st.write('q is the dividend yield of the stock. For no dividends, set q=0.')
st.write('Calculating the future possible stock prices will leave you with a tree that looks like this:')
st.image("tree.png",width=700)
st.title('Step 2: Discounting Option Values')
st.write('It is easy to understand that on the date of maturity, the value of an option is the profit you recieve if you were to excersize that option.')
st.write('At maturity, a call option with a strike price of 105 would be worth 5 dollars if the stock price was 110 dollars. Note that this is independent of any premium paid for the option.')
st.write('We have already calculated a series of possible prices for our stock on the date of maturity. For each of these prices, we calculate the intrinsic value of the option.')
st.image('big_tree.png',width=700)


st.write('All of these future values are possible, and so we will use them to create an expected value for the future price of the option at the previous time step.')
st.write('However, we cannot ignore the time value of money, so we need to discount each of these expected values to create the values at the previous layer: the Discounted Expected Values.')
st.write('The equation we will be using includes a factor for discounting using a continuously compounding discount rate, as all options contracts are priced according to.')
st.write('For each node at the previous step, we use this equation to calculate the expected value of the option at maturity.')
st.image("induction.png",use_container_width=True)
st.write('All this equation is saying is that you take the probability that the stock moves up times the price if the stock moves up and add it to the down probability times the price if the stock goes down.')
st.write('Then discount this factor back one time step and you have your current value.')
st.title('Step 3: Compare to Intrinsic Value')
st.write('If we were dealing with European Options, we could just use the previous formula to fill the whole tree until we were left with one value at t=0, which would be equal to our current price.')
st.write('But with American options you can exercise your option at any time.')
st.write('We assume that you as an investor behave logically and always want the most money possible.')
st.write('So for every node, we calculate the intrinsic value of the option, meaning the profit you would get if you excersized the option right now.')
st.write('You compare your expected value to your intrinsic value and whichever is higher becomes your current option value because the investor always takes the most profitable option.')
st.image("compare.png",use_container_width=True)
st.write('This is another complicated equation that just says that you choose the higher value between expected and intrinsic value as you discount back through your time steps.')
st.write('Once you have discounted all the way back to the present (comparing to intrinsic values as you go), your final value for contract price is your first node is the current option price according to the binomial model.')
st.image("tree2.png",width=700)






