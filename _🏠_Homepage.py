# Imports
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta
from PIL import Image
import os
import gui
import math

#Imports from functions we wrote
from data import data_prep
from data import consts

# Page Configuration
st.set_page_config(page_icon="🏠", page_title="Homepage", layout="wide")

# Cache Data
@st.cache(allow_output_mutation=True)
# Function for ingesting data
def read_data():
    # Collecting data
    data_path =  'data/model_data3.csv'
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

# Place Logo In Top Right Corner
pic_col1, pic_col2, pic_col3 = st.columns(3, gap='large')
steelcaseLogo = Image.open(
    'assets/SteelcaseLogo.png')
with pic_col3:
    st.image(steelcaseLogo, width=100)

# Header
gui.colored_header(label='Home Page',
                   description='Gives a high level overview of the performance of each material group.',
                   color_name='blue-60')

# Getting the data for the correct dates
start_date = str(today - timedelta(days = 365))
end_date = str(today)
requested_data = data.SplitData(start_date, end_date)
requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)

# Creating Number of Material Group Dropdown
mat_grp_ranks = requested_data['Material_Group'].value_counts()
mat_grp_ranks = mat_grp_ranks.reset_index()
number_of_grps = st.selectbox("Number of Material Groups to Display (Ranked By Number of POs):",
                              ('5','10','20','30','All'))

# Handling case where user selects 'all'. Assigning a number
if number_of_grps == 'All':
    number_of_grps = len(mat_grp_ranks)

# Get data for each of the material groups that were selected and loop through each group
for group in range(int(number_of_grps)):
    group_id = mat_grp_ranks['index'][group]
    group_df = requested_data[requested_data['Material_Group'] == group_id]

    # Creating expanders for each material group with a header
    with st.expander('Material Group: ' + str(group_id) + ', Total Number of Orders: ' + str(len(group_df)), expanded = True):
        # Creating Titles for each column
        #KPI Name
        col1,col2,col3 = st.columns([.75,1.5,2])
        with col1:
            st.markdown("""
                <h4 align = "center">
                    KPI Name
                </h4>
            """, unsafe_allow_html=True
                        )
        #Last Year Performance
        with col2:
            st.markdown("""
                <h4 align = "center">
                    Last Year Performance
                </h4>
            """, unsafe_allow_html=True
                        )
        #Plot
        with col3:
            st.markdown("""
                <h4 align = "center">
                    Plot
                </h4>
            """, unsafe_allow_html=True
                        )

        #Draw a line for the header
        st.markdown(
            f'<hr style="background-color: #3d9df3; margin-top: 0; margin-bottom: 0; height: 2px; border: none; border-radius: 3px;">',
            unsafe_allow_html=True
        )
        # Creating Late Percentage Row
        col1,col2,col3 = st.columns([.75,1.5,2])

        #KPI Name - Late Percentage
        with col1:
            st.markdown("""
                <h6 align = "center">
                    Late Percentage
                </h6>
            """, unsafe_allow_html=True
                        )

        #Last Year Performance - Late Percentage
        with col2:
            on_time_perc = round(100 - 100 * len(group_df[(group_df["days_late"] > 0)]) / len(group_df), 2)
            late_perc_str = str(round(100 - on_time_perc,2)) + '%'
            st.markdown(f"""
                <h6 align = "center">
                    {late_perc_str}
                </h6>
            """, unsafe_allow_html=True
                        )

        # Plot - Late Percentage for Each Material Grop
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

        # Draw a line between late percentage and average days late
        st.markdown(
            f'<hr style="background-color: #3d9df3; margin-top: 0; margin-bottom: 0; height: 1px; border: none; border-radius: 3px;">',
            unsafe_allow_html=True
        )

        # Average Days Late Row
        col1, col2, col3 = st.columns([.75,1.5,2])

        #KPI Name - Average Days Late
        with col1:
            st.markdown("""
                <h6 align = "center">
                    Average Days Late (When Late)
                </h6>
            """, unsafe_allow_html=True
                        )

        #Last Year Performance - Average Days Late
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

        #Plot - Average Days Late for each material group
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


