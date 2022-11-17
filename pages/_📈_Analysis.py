# Imports
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from datetime import date
from PIL import Image
import os, sys
import gui
import datetime
from math import log

# Getting Directory
sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))

#Imports from functions we wrote
from data import data_prep
from data import consts
from data import forecastingcodev2

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
                   description='Explore performance at the supplier, material group, or material level of detail',
                   color_name='blue-60')

# Filters for the page
# First level filters
filter_l1_1, filter_l1_2, filter_l1_3, filter_l1_4 = st.columns(4, gap='small')
with filter_l1_1:
    start_date1 = st.date_input('Start Date', datetime.date(2019,1,1))
    start_date = str(start_date1)
with filter_l1_2:
    end_date1 = st.date_input('End Date', today)
    end_date = str(end_date1)
with filter_l1_3:
    lod = st.selectbox(
        "Preferred level of detail:",
        ("Supplier", "Material Group", "Material", "Full Supply Chain"),
    )
with filter_l1_4:
    view = st.selectbox(
        "Preferred view:",
        ("Performance over time", "Number of deliveries over time", "Scatter view"),
    )
# Second Level Filters
filter_l2_1, filter_l2_2, filter_l2_3, filter_l2_4 = st.columns(4, gap='small')
with filter_l2_1:
    st.write('Selected time frame: ' + start_date + '  to  ' + end_date)
with filter_l2_3:
    st.write('Selected LOD: ' + lod)
with filter_l2_4:
    st.write('Selected View: ' + view)
gui.colored_header(label='', description='', color_name='blue-30')

# Main Page Set Up. Begin with Filters on right, charts on left.
viz_col, filter_col = st.columns([5, 1])

# Set up data according to lod
entity_map = {
    'Supplier': 'Vendor_number',
    'Material Group': 'Material_Group',
    'Material': 'Material',
}
requested_data = data.SplitData(start_date, end_date)
requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)
if lod != 'Full Supply Chain':
    entities = requested_data[entity_map[lod]].unique()
else:
    entities = requested_data
try:
    entities.sort()
except:
    pass
# Create filter column
with filter_col:
    if lod != 'Full Supply Chain':
        st.write(f'Filter by {lod}s')
        #group_by = st.checkbox(f'Group all {lod}s?')
    if view == 'Performance over time':
        forecast_check = st.checkbox(f'Forecast?')
    if lod != 'Full Supply Chain':
        lod_filter = st.multiselect(
            "Filter:",
            entities,
            entities[:2]
        )


def performance_over_time():
    ##requested_data = data.SplitData(start_date, end_date)
    #requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)
    #if
    #entities = requested_data[entity_map[lod]].unique()


    indexDataUsed = gui.indexData[(gui.indexData["Date"] <= end_date) & (gui.indexData["Date"] >= start_date)]
    # Line Chart for On-Time Delivery Percentage By Month
    dl_fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    if lod == 'Full Supply Chain':
        viz_data = requested_data.resample('M', on='Scheduled_relevant_delivery_date').late_bin.mean() * 100
        viz_data = viz_data.reset_index()
        viz_data['late_bin'] = np.where(
            viz_data['late_bin'].isna(),
            viz_data['late_bin'].shift(-1),
            viz_data['late_bin'])
        dl_fig1.add_scatter(
            x=indexDataUsed["Date"],
            y=indexDataUsed["Value"],
            name="Global Supply Chain Pressure Index (GSCPI)",
            secondary_y=True)
        if forecast_check:
            viz_data['Forecast'] = 'Supplier Data'
            viz_data = viz_data.rename(columns={'Scheduled_relevant_delivery_date': 'Scheduled Delivery Date',
                                                'late_bin':'Late Percentage'})
            viz_data = viz_data.dropna()
            forecast_test = forecastingcodev2.forecasting(requested_data,5, 'Full')
            projections = forecast_test[0]
            projections = projections.rename(columns = {'predicted_percentage_late' : 'Late Percentage',
                                          'time': 'Scheduled Delivery Date'})
            projections['Forecast'] = 'Forecast'
            projections['Late Percentage'] = projections['Late Percentage'] * 100
            projections = pd.concat([viz_data.tail(1), projections]).reset_index(drop = True)
            dl_fig1.add_scatter(
                x = viz_data['Scheduled Delivery Date'],
                y = viz_data['Late Percentage'],
                name = 'Real Performance',
                secondary_y = False)
            dl_fig1.add_scatter(
                x = projections['Scheduled Delivery Date'],
                y = projections['Late Percentage'],
                name = 'Forecasted Performance',
                secondary_y = False)
        else:
            dl_fig1.add_scatter(
                x=viz_data["Scheduled_relevant_delivery_date"],
                y=viz_data["late_bin"],
                name=f"Average {lod} Performance over Time",
                secondary_y=False)
        dl_fig1.update_xaxes(title_text="Months")
        dl_fig1.update_yaxes(title_text="Late Delivery Percentage")
        dl_fig1.update_layout(title="Late Delivery Percentages")
        dl_fig1.update_yaxes(title_text="Global Supply Chain Pressure Index (GSCPI)", secondary_y=True)
        st.plotly_chart(dl_fig1, use_container_width=True, show_legend=True)
    else:
        requested_data_p = data.SplitData(start_date, end_date)
        requested_data_p['late_bin'] = np.where(requested_data_p['days_late'] > 0, 1, 0)
        grouped = requested_data_p.groupby(entity_map[lod]).resample('M', on='Scheduled_relevant_delivery_date')
        viz_data = grouped.mean()['late_bin'].reset_index()
        viz_data = viz_data.reset_index()
        viz_data =viz_data[viz_data[entity_map[lod]].isin(lod_filter)]
        ent = entity_map[lod]
        viz_data['late_bin'] = viz_data.groupby([ent])['late_bin'].ffill()
        dl_fig1.add_trace(go.Scatter(
            x=indexDataUsed["Date"],
            y=indexDataUsed["Value"],
            name="Global Supply Chain Pressure Index (GSCPI)"),
            secondary_y=True)
        if forecast_check:
            for entity in lod_filter:
                entity_data = viz_data[viz_data[entity_map[lod]] == entity]
                entity_data['Forecast'] = 'Supplier Data'
                entity_data = entity_data.dropna()
                try:
                    forecast_test = forecastingcodev2.forecasting(entity_data, 5,lod)
                    projections = forecast_test[0]
                    projections = projections.rename(columns={'predicted_percentage_late': 'Late Percentage',
                                                             'time': 'Scheduled Delivery Date'})
                    projections['Forecast'] = 'Forecast'
                    entity_data = entity_data.rename(
                        columns={'Scheduled_relevant_delivery_date': 'Scheduled Delivery Date',
                                 'late_bin': 'Late Percentage'})
                    projections = pd.concat([entity_data.tail(1), projections]).reset_index(drop=True)
                    dl_fig1.add_scatter(
                        x=projections['Scheduled Delivery Date'],
                        y=100*projections['Late Percentage'],
                        name='Forecasted Performance for ' + lod + ' ' + str(entity),
                        secondary_y=False)
                except:
                    st.error("WARNING: Not enough data to give an accurate forecast for " + lod + ' ' + str(entity), icon = '🚨')
                entity_data = entity_data.rename(
                    columns={'Scheduled_relevant_delivery_date': 'Scheduled Delivery Date',
                             'late_bin': 'Late Percentage'})
                dl_fig1.add_scatter(
                    x=entity_data['Scheduled Delivery Date'],
                    y=100*entity_data['Late Percentage'],
                    name='Real Performance for ' + lod + ' ' + str(entity),
                    secondary_y=False)
        else:
            for entity in lod_filter:
                dl_fig1.add_trace(
                    go.Scatter(x=viz_data[(viz_data[entity_map[lod]] == entity)]['Scheduled_relevant_delivery_date'],
                        y=viz_data[(viz_data[entity_map[lod]] == entity)]['late_bin'],
                        name=str(entity)
                        ))
        dl_fig1.update_xaxes(title_text="Months")
        dl_fig1.update_yaxes(title_text="Late Delivery Percentage")
        dl_fig1.update_layout(title="Late Delivery Percentages")
        dl_fig1.update_yaxes(title_text="Global Supply Chain Pressure Index (GSCPI)", secondary_y=True)
        st.plotly_chart(dl_fig1, use_container_width=True, show_legend=True)


def number_of_deliveries():
    requested_data = data.SplitData(start_date, end_date)
    requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)
    if lod != 'Full Supply Chain':
        entities = requested_data[entity_map[lod]].unique()

    # Line Chart for number of deliveries by month
    if lod == 'Full Supply Chain':
        dl_fig2 = make_subplots(specs=[[{"secondary_y": False}]])
        viz_data = requested_data.resample('M', on='Scheduled_relevant_delivery_date').PO_Number.count()
        viz_data = viz_data.reset_index()
        dl_fig2.add_trace(go.Scatter(
            x=viz_data["Scheduled_relevant_delivery_date"],
            y=viz_data["PO_Number"],
            name=f"Number of deliveries by {lod}over Time")
        )
        dl_fig2.update_xaxes(title_text="Months")
        dl_fig2.update_yaxes(title_text="Number of PO Lines")
        dl_fig2.update_layout(title="Number of Deliveries over Time")
        st.plotly_chart(dl_fig2, use_container_width=True, show_legend=True)
    else:
        dl_fig2 = make_subplots(specs=[[{"secondary_y": False}]])
        viz_data = requested_data.groupby(entity_map[lod]).resample('M',
                                                                    on='Scheduled_relevant_delivery_date').PO_Number.count()
        viz_data = viz_data.reset_index()
        for entity in lod_filter:
            dl_fig2.add_trace(
                go.Scatter(x=viz_data[(viz_data[entity_map[lod]] == entity)]['Scheduled_relevant_delivery_date'],
                           y=viz_data[(viz_data[entity_map[lod]] == entity)]['PO_Number'],
                           name=str(entity)
                           ))
        dl_fig2.update_xaxes(title_text="Months")
        dl_fig2.update_yaxes(title_text="Number of PO Lines")
        dl_fig2.update_layout(title="Number of Deliveries over Time")
        st.plotly_chart(dl_fig2, use_container_width=True, show_legend=True)


def scatter_View():
    requested_data = data.SplitData(start_date, end_date)
    requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)


    # Scatter View
    group_by = st.checkbox(f'Group all {lod}s?')
    if group_by:
        late_perc_data = requested_data.groupby(entity_map[lod], as_index=False, dropna=False)['late_bin'].mean()
        late_days_data = requested_data[(requested_data['days_late'] > 0)]
        size_data = requested_data.groupby(entity_map[lod], as_index=False, dropna=False)['PO_Number'].count()
        late_severity_data = late_days_data.groupby(entity_map[lod], as_index=False, dropna=False)['days_late'].mean()
        comb_df = pd.DataFrame({lod: late_perc_data[entity_map[lod]],
                               'Average Days Late': late_severity_data['days_late'],
                               'Percentage Late': late_perc_data['late_bin'],
                               '# Of Times Ordered': size_data['PO_Number']})
        comb_df['Average Days Late'] = np.where((comb_df['Average Days Late'].isna()) | (comb_df['Average Days Late']==0),
                                                0, comb_df['Average Days Late'])
        conditions = [
            comb_df['Average Days Late'] > 7,
            comb_df['Percentage Late'] > 0.10,
            comb_df['Average Days Late'] <= 7,
            comb_df['Percentage Late'] <= 0.10
        ]
        values = [
            'Not Meeting Expectations', 'Not Meeting Expectations',
            'Meeting Expectations', 'Meeting Expectations'
        ]
        def logger(x):
            return x**(1/1.5)
        comb_df['logscale'] = comb_df['# Of Times Ordered'].apply(logger)
        comb_df[f'{lod} Performance'] = np.select(conditions, values)
        late_std = comb_df['Average Days Late'].std()
        late_mean = comb_df['Average Days Late'].mean()
        colors = {'Not Meeting Expectations': 'red', 'Meeting Expectations': 'green'}
        fig = px.scatter(comb_df, x='Percentage Late', y='Average Days Late',
                         title=f'{lod} Performance', size='logscale',
                         color = comb_df[f'{lod} Performance'], color_discrete_map= colors,
                         hover_name=f'{lod}')
        fig.add_shape(type='rect', x0=0, x1=0.10, y0=0, y1=7, line=dict(color='green'))
        fig.update_layout(yaxis_range=[0, late_mean + (late_std*1.5)])
        fig.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, show_legend=True)
    else:
        requested_data2 = requested_data[requested_data[entity_map[lod]].isin(lod_filter)]
        late_perc_data = requested_data2.groupby(entity_map[lod], as_index=False, dropna=False)['late_bin'].mean()
        late_days_data = requested_data2[(requested_data2['days_late'] > 0)]
        size_data = requested_data2.groupby(entity_map[lod], as_index=False, dropna=False)['PO_Number'].count()
        late_severity_data = late_days_data.groupby(entity_map[lod], as_index=False, dropna=False)['days_late'].mean()
        comb_df = pd.DataFrame({lod: late_perc_data[entity_map[lod]],
                                'Average Days Late': late_severity_data['days_late'],
                                'Percentage Late': late_perc_data['late_bin'],
                                '# Of Times Ordered': size_data['PO_Number']})
        comb_df['Average Days Late'] = np.where(
            (comb_df['Average Days Late'].isna()) | (comb_df['Average Days Late'] == 0),
            0, comb_df['Average Days Late'])
        conditions = [
            comb_df['Average Days Late'] > 7,
            comb_df['Percentage Late'] > 0.10,
            comb_df['Average Days Late'] <= 7,
            comb_df['Percentage Late'] <= 0.10
        ]
        values = [
            'Not Meeting Expectations', 'Not Meeting Expectations',
            'Meeting Expectations', 'Meeting Expectations'
        ]

        def logger(x):
            return x ** (1 / 1.5)

        comb_df['logscale'] = comb_df['# Of Times Ordered'].apply(logger)
        comb_df[f'{lod} Performance'] = np.select(conditions, values)
        late_std = comb_df['Average Days Late'].std()
        late_mean = comb_df['Average Days Late'].mean()
        colors = {'Not Meeting Expectations': 'red', 'Meeting Expectations': 'green'}
        fig = px.scatter(comb_df, x='Percentage Late', y='Average Days Late',
                         title=f'{lod} Performance', size='logscale',
                         color = comb_df[f'{lod} Performance'], color_discrete_map= colors,
                         hover_name=f'{lod}')
        fig.add_shape(type='rect', x0=0, x1=0.10, y0=0, y1=7, line=dict(color='green'))
        fig.update_layout(yaxis_range=[0, max(10,late_mean + (late_std * 1.5))])
        fig.update_layout(xaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True, show_legend=True)

# Create View
with viz_col:
    if view == "Performance over time":
        performance_over_time()
    elif view == 'Number of deliveries over time':
        number_of_deliveries()
    elif view == 'Scatter view':
        scatter_View()

def metrics():
    requested_data = data.SplitData(start_date, end_date)
    requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)
    if lod != 'Full Supply Chain':
        entities = requested_data[entity_map[lod]].unique()
    else:
        entities = requested_data

    # Write Metrics
    if lod == 'Full Supply Chain':
        inclusion = entities
    else:
        inclusion = lod_filter
    if lod != 'Full Supply Chain':
        metric_data = requested_data[requested_data[entity_map[lod]].isin(inclusion)]
    else:
        metric_data = requested_data
    if len(metric_data) > 0:
        m1, m2, m3, m4 = st.columns(4, gap="small")
        first_date = pd.to_datetime(start_date, format='%Y-%m-%d')
        second_date = pd.to_datetime(end_date, format='%Y-%m-%d')
        date_difference = second_date - first_date
        previous_date = first_date - date_difference
        min_date = pd.to_datetime('2019-03-01', format='%Y-%m-%d')
        if previous_date < min_date:
            previous_date = min_date
        prev_data = data.SplitData(previous_date, start_date)
        if lod != 'Full Supply Chain':
            prev_data = prev_data[prev_data[entity_map[lod]].isin(inclusion)]

        # Metric 1: On-time percentage
        on_time_perc = round(100 - 100 * len(metric_data[(metric_data["days_late"] > 0)]) / len(metric_data), 2)
        if len(prev_data) == 0:
            old_on_time_perc = on_time_perc
        else:
            old_on_time_perc = round(100 - 100 * len(prev_data[(prev_data["days_late"] > 0)]) / len(prev_data), 2)
        m1.metric(label="On-Time Delivery Percentage:", value=(str(on_time_perc) + '%'),
                        delta=str(round(on_time_perc - old_on_time_perc, 2)) + '%')

        # Metric 2: Number of POs completed
        distinct_POs = len(metric_data['PO_Number'].value_counts())
        if len(prev_data) == 0:
            old_distinct_POs = distinct_POs
        else:
            old_distinct_POs = len(prev_data['PO_Number'].value_counts())
        m2.metric(label='Completed POs', value= "{:,}".format(int(distinct_POs)), delta="{:,}".format(int(distinct_POs - old_distinct_POs)))

        # Metric 3: Number of Suppliers
        num_supp = len(metric_data['Vendor_number'].value_counts())
        if len(prev_data) == 0:
            old_num_supp = num_supp
        else:
            old_num_supp = len(prev_data['Vendor_number'].value_counts())
        m3.metric(label='Suppliers Used', value = "{:,}".format(int(num_supp)), delta="{:,}".format(int(num_supp - old_num_supp)))

        # Metric 4 - Cost Data
        invoice_cost = metric_data['Invoice ($)'].sum()
        if len(prev_data) == 0:
            old_invoice_cost = invoice_cost
        else:
            old_invoice_cost = prev_data['Invoice ($)'].sum()
        m4.metric(label='Total Invoice Cost',
                value="{:,}".format(round(invoice_cost), 2),
                delta="{:,}".format(round(invoice_cost - old_invoice_cost), 2))

    else:
        st.error(f'There is currently no data available for the chosen filter and/or time frame. '
                 f'Please select at least 1 {lod} from the Filter dropdown or adjust the time frame.', icon = '🚨')

    gui.colored_header(label='', description='', color_name='blue-30')

metrics()