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
                <div class="title">Options and Volatility | Black-Scholes Model + Greeks Guide</div>
                <div class="subtext">Coby Walmsley • Lehigh University Masters in Financial Engineering</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.title('Black-Scholes Option Pricing Model')
    st.write("The Black-Scholes Pricing Model is an equation for pricing European Options, which can only be exercised on the date of their maturity.")
    st.write("The Black-Scholes Equation is Shown Below.")
    st.image("Black-Scholes_eq.png",use_container_width=True)
    st.title("The Greeks")
    st.write("The Greeks are a series of factors that determine the price change in the option contract given the change in another variable.")
    st.write("This change of Price-Change-Relative-To-X can be represented with partial derivitaves of the Black-Scholes Equation.")
    st.title('Delta')
    st.write('Delta answers the question: How much will my option price move if the price of the underlying moves $1?')
    st.image("Delta.png",use_container_width=True)
    st.title('Gamma')
    st.write('Gamma answers the question: How much will my option price move if delta changes by 1?')
    st.image("Gamma.png",use_container_width=True)
    st.write('Equation for N`, necessary for Gamma and Vega:')
    st.image("N_prime.png",use_container_width=True)
    st.title('Theta')
    st.write('Option contract prices decay with time.')
    st.write('Theta answers the question: How much will my option price change every day if everything else is held constant?')
    st.image("Theta.png",use_container_width=True)
    st.title('Vega')
    st.write('Theta answers the question: How much will my option price change if implied volatility changes by 1%?')
    st.image("Vega.png",use_container_width=True)
    st.title('Rho')
    st.write('Rho answers the question: How much will my option price change if the risk free rate changes by 1%?')
    st.image("Rho.png",use_container_width=True)

