# Imports
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os, sys
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import math

import gui # formatting script in app folder
import MVEF # function script in app folder

# Configuration
st.set_page_config(page_icon="🚀", page_title="Recommendations", layout="wide")

# Load Data
data = st.session_state['data']
data_frame = data.clean_data

# Put Logo in Top Right (Can use markdown)
pic_col1, pic_col2, pic_col3 = st.columns(3, gap='large')
steelcaseLogo = Image.open(
    'assets/SteelcaseLogo.png')
with pic_col3:
    st.image(steelcaseLogo, width=100)

# Header
gui.colored_header(label='Find the best supplier portfolio',
                   description='Each material group has an ideal supplier portfolio to reduce risk and optimize '
                               'performance.',
                   color_name='blue-60')

# Create filters and layout -> for view to display
filter_l1_1, filter_l1_2, filter_l1_3 = st.columns([5, 2, 3], gap='small') # Creates a columnated page
with filter_l1_1:
    data_obj = st.session_state['data']
    today = pd.to_datetime("today")
    data = data_obj.SplitData('2019-01-01', today) # Using all historical data
    mgs = set(list(data['Material_Group'])) # Getting all material groups into a list
    mgs = [i for i in mgs if type(i)==str]
    MG = st.selectbox(
        "Select the material group you are interested in:",
        mgs
    )
with filter_l1_2:
    VIEW = st.radio(
        "Select your preferred view 👉",
        ('Portfolio Weights', 'Distributions', 'Mean-Variance')
    )
with filter_l1_3:
    st.subheader('Fine-tune recommendations')
    st.caption('Specify algorithm parameters')
    st.write(
        '<hr style="background-color: #3d9df3"; margin-top: 0; margin-bottom: 0; height: 1px; border: none; '
        'border-radius: 1px;">',
        unsafe_allow_html=True,
    )

# Main Views + Fine-Tune Filters
Main_View, Fine_Tune = st.columns([7, 3], gap='small')
with Fine_Tune:
    st.write('Number of Suppliers')
    st.caption('The minimum number of suppliers that will be recommended')
    NUMBER = st.radio(
        "Select #",
        (2,3,4)
    )
    st.write(
        '<hr style="background-color: #3d9df3"; margin-top: 0; margin-bottom: 0; height: 1px; border: none; '
        'border-radius: 1px;">',
        unsafe_allow_html=True,
    )

# Build visualizations
with Main_View:
    try:
        data_obj = st.session_state['data']
        data = data_obj.SplitData('2019-01-01', today)
        # Run function to generate insights on suppliers in each material group
        # Require at least 15 samples to recommend a supplier
        MVEF_Clean, sigma, perf_columns, sqrt_lamda_star, mvp = MVEF.data_process(mg=MG, thresh_n=15, data=data)
        w_martrix = MVEF.portfolio_generator(columns=perf_columns,
                                             increments=[1, .9, .8, .7, .6, .5, .5, .4, .4, .3, .3, .3, .2, .2, .2, .2,
                                                         .2, .1, .1, .1, .1, .1, .1, .1, .1, .1, .1, 0, 0, 0, 0, 0, 0,
                                                         0, 0, 0, 0])
        MVEF_Data = MVEF.point_generator(w_matrix=w_martrix, cov_matrix=sigma, MVEF_Clean=MVEF_Clean)
        pres_data = pd.merge(MVEF_Data, w_martrix, how='left', left_on='Point', right_index=True)
        pres_data = pres_data.rename(columns={
            'Point': 'Portfolio',
            'Mean': 'Expected On-Time %',
            'Std': 'Risk (Std)',
        })
        # No point in showing data for only one possible supplier
        if len(perf_columns) == 1:
            st.warning(f'Only one supplier with enough data in material group: {perf_columns[0].split("OTP")[0]}', icon="⚠️")
        else:
            # Build main portfolio weights view
            if VIEW == 'Portfolio Weights':
                st.subheader('Best supplier portfolios')
                st.write(
                    '<hr style="background-color: #3d9df3"; margin-top: 0; margin-bottom: 0; height: 1px; border: none; '
                    'border-radius: 1px;">',
                    unsafe_allow_html=True,
                )
                pres_data = pres_data.sort_values(['Expected On-Time %', 'Risk (Std)'], ascending=False)
                N = len(perf_columns)
                pres_data = pres_data[pres_data.eq(0).sum(1).lt(N-NUMBER+1)]
                options = pres_data.head(10)
                for col in perf_columns:
                    options[col] = options[col] * 100
                options['Expected On-Time %'] = options['Expected On-Time %'] * 100
                options = options.style.format(subset=perf_columns+['Expected On-Time %'], formatter="{:.1f}%")
                st.table(options)
            # Build mean-variance view
            if VIEW == 'Mean-Variance':
                pres_data = pres_data.sort_values(['Expected On-Time %', 'Risk (Std)'], ascending=False)
                N = len(perf_columns)
                pres_data = pres_data[pres_data.eq(0).sum(1).lt(N-NUMBER+1)]
                fig = px.scatter(pres_data, x="Risk (Std)", y="Expected On-Time %",
                                 hover_data=['Portfolio']+perf_columns,
                                 color="Expected On-Time %", color_continuous_scale=px.colors.sequential.Viridis,
                                 size_max=15)
                data['late_bin'] = np.where(data['days_late']>0, 1, 0)
                current_mean = data[data['Material_Group']==MG]['late_bin'].mean()
                current_std = data[data['Material_Group']==MG]['late_bin'].std()
                current_data = pd.DataFrame({'Expected On-Time %': [current_mean],
                                             'Risk (Std)': [current_std],
                                             'Portfolio': ['Current Portfolio'],
                                             'Color': 'Gold'
                                             })
                st.subheader('Expected On-Time % vs Risk')
                st.caption('Each point represents a possible portfolio of suppliers')
                st.write(
                    '<hr style="background-color: #3d9df3"; margin-top: 0; margin-bottom: 0; height: 1px; border: none; '
                    'border-radius: 1px;">',
                    unsafe_allow_html=True,
                )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True, show_legend=True)
            # Build distributions view
            if VIEW == 'Distributions':
                PFO = st.selectbox(
                    "Select the portfolio you are interested in:",
                    set(list(pres_data['Portfolio']))
                )
                p = list(pres_data[pres_data['Portfolio']==PFO]['Expected On-Time %'])[0]
                alpha = math.ceil(p * 20)
                beta = 20 - alpha
                if beta <= 0:
                    beta = .5
                z = np.random.beta(alpha, beta, size=1000)
                fig = ff.create_distplot(hist_data = [z],
                                         group_labels = ['Expected Distribution of Outcomes'],
                                         bin_size = .005,
                                         show_hist=True,
                                         histnorm='probability density',
                                         show_rug=True,
                                         colors=['#A6ACEC'],
                                         )
                fig.update_layout(title_text=f'Probability Distribution of On-Time % for Portfolio: {PFO}',
                                  yaxis_title="Probability Density",
                                  xaxis_title="On-Time Percentage",
                                  )
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True, show_legend=True)
    except Exception as e:
        st.warning(f'Not enough clean data for Material Group {MG}', icon="⚠️")