# Imports
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import date, timedelta
from PIL import Image
import os, sys
import gui
import math
import datetime
from math import log

# Getting Directory
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

#Imports from functions we wrote
from data import data_prep
from data import consts

# Configuration
st.set_page_config(page_icon="🏭", page_title="Steelcase Data Analytics", layout="wide")

# Main directory path ('Tool Dev')
# main_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
# st.session_state['directory'] = main_dir

# Cache Data
@st.cache(allow_output_mutation=True)
# Function for ingesting data
def read_data():
    # Collecting data
    data_path = 'data/model_data3.csv'
    # Call data class and parameters from data module
    data_obj = data_prep.IngestionPipe(features=consts.features, target=consts.target, dates=consts.dates,
                                       post_date=consts.clean_end_date, pre_date=consts.clean_start_date)
    data_obj.dataAllocation(path=data_path)
    return data_obj

# Load in and Save Data
data = read_data()
#st.session_state allows the data to be accessed in other pages
st.session_state['data'] = data

# Get Today and earliest day in data
today = date.today()
st.session_state['today'] = today
earliest_date = pd.to_datetime(data.clean_data['Scheduled_relevant_delivery_date'][0], utc=False).strftime('%Y-%m-%d')

# Trying to Put Logo in Top Right (Can use markdown)
pic_col1, pic_col2, pic_col3 = st.columns(3, gap='large')
steelcaseLogo = Image.open(
    'assets/SteelcaseLogo.png')
with pic_col3:
    st.image(steelcaseLogo, width=100)

# Header
gui.colored_header(label='Home Page',
                   description='Gives a high level overview of the performance of each material group.',
                   color_name='blue-60')

###CHANGE the date
start_date = str(today - timedelta(days = 365))
end_date = str(today)
requested_data = data.SplitData(start_date, end_date)
requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)


mat_grp_ranks = requested_data['Material_Group'].value_counts()
mat_grp_ranks = mat_grp_ranks.reset_index()
number_of_grps = st.selectbox("Number of Material Groups to Display (Ranked By Number of POs):",
                              ('5','10','20','30','All'))
if number_of_grps == 'All':
    number_of_grps = len(mat_grp_ranks)
for group in range(int(number_of_grps)):
    group_id = mat_grp_ranks['index'][group]
    group_df = requested_data[requested_data['Material_Group'] == group_id]
    with st.expander('Material Group: ' + str(group_id) + ', Total Number of Orders: ' + str(len(group_df)), expanded = True):
        #On Time Percentage
        col1,col2,col3 = st.columns([.75,1.5,2])
        with col1:
            st.markdown("""
                <h4 align = "center">
                    KPI Name
                </h4>
            """, unsafe_allow_html=True
                        )
        with col2:
            st.markdown("""
                <h4 align = "center">
                    Last Year Performance
                </h4>
            """, unsafe_allow_html=True
                        )
        with col3:
            st.markdown("""
                <h4 align = "center">
                    Plot
                </h4>
            """, unsafe_allow_html=True
                        )
        #st.write(
        #    gui.updated_home_line(color_name='blue-30')
        #)
        st.markdown(
            f'<hr style="background-color: #3d9df3; margin-top: 0; margin-bottom: 0; height: 2px; border: none; border-radius: 3px;">',
            unsafe_allow_html=True
        )
        col1,col2,col3 = st.columns([.75,1.5,2])
        with col1:
            st.markdown("""
                <h6 align = "center">
                    Late Percentage
                </h6>
            """, unsafe_allow_html=True
                        )
        with col2:
            on_time_perc = round(100 - 100 * len(group_df[(group_df["days_late"] > 0)]) / len(group_df), 2)
            late_perc_str = str(round(100 - on_time_perc,2)) + '%'
            st.markdown(f"""
                <h6 align = "center">
                    {late_perc_str}
                </h6>
            """, unsafe_allow_html=True
                        )
        with col3:
            grouped = group_df.resample('M', on='Scheduled_relevant_delivery_date')
            viz_data = grouped.mean()['late_bin'].reset_index()
            viz_data = viz_data.rename(columns = {"Scheduled_relevant_delivery_date": "Scheduled Delivery Date",
                                       "late_bin": "Late Percentage"})
            viz_data["Late Percentage"] = (viz_data["Late Percentage"] * 100)
            st.line_chart(
                viz_data,
                x = "Scheduled Delivery Date",
                y = "Late Percentage",
                use_container_width = True,
                height = 175
            )
        st.markdown(
            f'<hr style="background-color: #3d9df3; margin-top: 0; margin-bottom: 0; height: 1px; border: none; border-radius: 3px;">',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns([.75,1.5,2])
        with col1:
            st.markdown("""
                <h6 align = "center">
                    Average Days Late (When Late)
                </h6>
            """, unsafe_allow_html=True
                        )
        with col2:
            late_df = group_df[group_df['late_bin'] > 0]
            late_days = late_df['days_late'].mean()
            if math.isnan(late_days):
                late_days = 0
            late_days_str = str(round(late_days, 2)) + ' Days'
            st.markdown(f"""
                <h6 align = "center">
                    {late_days_str}
                </h6>
            """, unsafe_allow_html=True
                        )
        with col3:
            grouped = late_df.resample('M', on='Scheduled_relevant_delivery_date')
            if len(grouped) > 0:
                viz_data = grouped.mean()['days_late'].reset_index()
                viz_data = viz_data.rename(columns={"Scheduled_relevant_delivery_date": "Scheduled Delivery Date",
                                                    "days_late": "Days Late"})
                st.line_chart(
                    viz_data,
                    x="Scheduled Delivery Date",
                    y="Days Late",
                    use_container_width=True,
                    height=175
                )


