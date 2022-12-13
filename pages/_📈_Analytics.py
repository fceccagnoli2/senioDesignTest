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


#Imports from functions we wrote
from data import data_prep
from data import consts
from data import forecastingcodev2

# Configuration
st.set_page_config(page_icon="📈", page_title="Analytics", layout="wide")

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

#Steelcase Logo in Top Right (Can use markdown)
pic_col1, pic_col2, pic_col3 = st.columns(3, gap='large')
steelcaseLogo = Image.open(
    'assets/SteelcaseLogo.png')
with pic_col3:
    st.image(steelcaseLogo, width=100)

# Header
gui.colored_header(label='Analysis',
                   description='Explore performance at the supplier, material group, or material level of detail',
                   color_name='blue-60')

# Filters for the page
# First level filters
filter_l1_1, filter_l1_2, filter_l1_3, filter_l1_4 = st.columns(4, gap='small')

#Start Date
with filter_l1_1:
    start_date1 = st.date_input('Start Date', datetime.date(2019,3,1))
    start_date = str(start_date1)

#End Date
with filter_l1_2:
    end_date1 = st.date_input('End Date', today)
    end_date = str(end_date1)

#Preferred LOD
with filter_l1_3:
    lod = st.selectbox(
        "Preferred level of detail:",
        ("Supplier", "Material Group", "Material", "Full Supply Chain"),
    )
# Preffered view
with filter_l1_4:
    view = st.selectbox(
        "Preferred view:",
        ("Performance over time", "Number of deliveries over time", "Scatter view"),
    )

# Second Level Filters
filter_l2_1, filter_l2_2, filter_l2_3, filter_l2_4 = st.columns(4, gap='small')

#Displaying selected information
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
    'Supplier': 'Supplier',
    'Material Group': 'Material_Group',
    'Material': 'Material',
}

#Get requested data and entities
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
    #Handles Full Supply Chain issue
    if lod != 'Full Supply Chain':
        st.write(f'Filter by {lod}s')

    #Forecast button for Performance over time
    if view == 'Performance over time':
        forecast_check = st.checkbox(f'Forecast?')

    #Displays full supply chian
    if lod != 'Full Supply Chain':
        lod_filter = st.multiselect(
            "Filter:",
            entities,
            entities[:1]
        )


# Shows a plot of late delivery percentage vs time.
# Overlays this plot with Global Supply Chain index
# Allows for forecasting
def performance_over_time():
    # Get data
    indexDataUsed = gui.indexData[(gui.indexData["Date"] <= end_date) & (gui.indexData["Date"] >= start_date)]

    # Line Chart for On-Time Delivery Percentage By Month
    dl_fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    #Full Supply Chain handling
    if lod == 'Full Supply Chain':
        viz_data = requested_data.resample('M', on='Scheduled_relevant_delivery_date').late_bin.mean() * 100
        viz_data = viz_data.reset_index()
        viz_data['late_bin'] = np.where(
            viz_data['late_bin'].isna(),
            viz_data['late_bin'].shift(-1),
            viz_data['late_bin'])

        #Add GSCPI to the plot
        dl_fig1.add_scatter(
            x=indexDataUsed["Date"],
            y=indexDataUsed["Value"],
            name="Global Supply Chain Pressure Index (GSCPI)",
            secondary_y=True)

        # Handle forecasting functionality
        if forecast_check:
            viz_data['Forecast'] = 'Supplier Data'
            viz_data = viz_data.rename(columns={'Scheduled_relevant_delivery_date': 'Scheduled Delivery Date',
                                                'late_bin':'Late Percentage'})
            viz_data = viz_data.dropna()

            #Calling forecasting function
            forecast_test = forecastingcodev2.forecasting(requested_data,5, 'Full')
            projections = forecast_test[0]
            bound1 = forecast_test[1]
            bound2 = forecast_test[2]
            projections = projections.rename(columns = {'predicted_percentage_late' : 'Late Percentage',
                                          'time': 'Scheduled Delivery Date'})
            projections['Forecast'] = 'Forecast'
            projections['Late Percentage'] = projections['Late Percentage']
            projections = pd.concat([viz_data.tail(1), projections]).reset_index(drop = True)
            projections['Late Percentage'][0] = projections['Late Percentage'][0] / 100

            #Display real performance
            dl_fig1.add_scatter(
                x = viz_data['Scheduled Delivery Date'],
                y = viz_data['Late Percentage'],
                name = 'Real Performance',
                secondary_y = False)

            #Display forecasting performance
            dl_fig1.add_scatter(
                x = projections['Scheduled Delivery Date'],
                y = 100*projections['Late Percentage'],
                name = 'Forecasted Performance',
                secondary_y = False)

        # If not forecasted, display performance
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

    #When Full Supply Chain is not selected
    else:
        #Get data ready
        requested_data_p = data.SplitData(start_date, end_date)
        requested_data_p['late_bin'] = np.where(requested_data_p['days_late'] > 0, 1, 0)
        grouped = requested_data_p.groupby(entity_map[lod]).resample('M', on='Scheduled_relevant_delivery_date')
        viz_data = grouped.mean()['late_bin'].reset_index()
        viz_data = viz_data.reset_index()
        viz_data =viz_data[viz_data[entity_map[lod]].isin(lod_filter)]
        ent = entity_map[lod]
        viz_data['late_bin'] = viz_data.groupby([ent])['late_bin'].ffill()

        #Plot GSCPI
        dl_fig1.add_trace(go.Scatter(
            x=indexDataUsed["Date"],
            y=indexDataUsed["Value"],
            name="Global Supply Chain Pressure Index (GSCPI)"),
            secondary_y=True)

        #Handle foreacsting
        if forecast_check:

            #Forecast for every entity
            for entity in lod_filter:

                #Get data for each entity
                entity_data = viz_data[viz_data[entity_map[lod]] == entity]
                entity_data['Forecast'] = 'Supplier Data'
                entity_data = entity_data.dropna()

                #Handle issue when there is not enough data to forecast
                try:

                    #Call forecasting function
                    forecast_test = forecastingcodev2.forecasting(entity_data, 5,lod)
                    projections = forecast_test[0]
                    bound1 = forecast_test[1]
                    bound2 = forecast_test[2]
                    projections = projections.rename(columns={'predicted_percentage_late': 'Late Percentage',
                                                             'time': 'Scheduled Delivery Date'})
                    projections['Forecast'] = 'Forecast'
                    entity_data = entity_data.rename(
                        columns={'Scheduled_relevant_delivery_date': 'Scheduled Delivery Date',
                                 'late_bin': 'Late Percentage'})
                    projections = pd.concat([entity_data.tail(1), projections]).reset_index(drop=True)

                    #Plot the forecasted performances
                    dl_fig1.add_scatter(
                        x=projections['Scheduled Delivery Date'],
                        y=100*projections['Late Percentage'],
                        name='Forecasted Performance for ' + lod + ' ' + str(entity),
                        secondary_y=False)

                #Display error message
                except:
                    st.error("WARNING: Not enough data to give an accurate forecast for " + lod + ' ' + str(entity), icon = '🚨')

                #Plot real performance
                entity_data = entity_data.rename(
                    columns={'Scheduled_relevant_delivery_date': 'Scheduled Delivery Date',
                             'late_bin': 'Late Percentage'})
                dl_fig1.add_scatter(
                    x=entity_data['Scheduled Delivery Date'],
                    y=100*entity_data['Late Percentage'],
                    name='Real Performance for ' + lod + ' ' + str(entity),
                    secondary_y=False)


        else:
            # Just plot each entity's performances when not forecasted

            for entity in lod_filter:
                dl_fig1.add_trace(
                    go.Scatter(x=viz_data[(viz_data[entity_map[lod]] == entity)]['Scheduled_relevant_delivery_date'],
                        y=viz_data[(viz_data[entity_map[lod]] == entity)]['late_bin']*100,
                        name=str(entity)
                        ))

        dl_fig1.update_xaxes(title_text="Months")
        dl_fig1.update_yaxes(title_text="Late Delivery Percentage")
        dl_fig1.update_layout(title="Late Delivery Percentages")
        dl_fig1.update_yaxes(title_text="Global Supply Chain Pressure Index (GSCPI)", secondary_y=True)
        st.plotly_chart(dl_fig1, use_container_width=True, show_legend=True)

    #Show dataframe of confidence bounds for predictions
    try:
        bound_df = pd.DataFrame()
        bound_df['Prediction Lower Bound'] = bound2['predicted_percentage_late']
        bound_df['Actual Prediction'] = projections['Late Percentage']
        bound_df['Prediction Upper Bound'] = bound1['predicted_percentage_late']
        bound_df = bound_df * 100
        bound_df['Time'] = bound1['time']
        st.write("Projection Confidence Intervals:")
        st.write(bound_df)
    except:
        return

# Show a line chart of the number of deliveries over time
def number_of_deliveries():

    # Get requested data
    requested_data = data.SplitData(start_date, end_date)
    requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)

    # Plot number of deliveries over time when Full Supply Chain is selected.
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

    # Plot number of deliveries over time when Full Supply Chain is not selected
    else:
        dl_fig2 = make_subplots(specs=[[{"secondary_y": False}]])
        for entity in lod_filter:
            entity_requested_data = requested_data[requested_data[entity_map[lod]] == entity]
            viz_data = entity_requested_data.resample('M', on='Scheduled_relevant_delivery_date').PO_Number.count()
            viz_data = viz_data.reset_index()
            dl_fig2.add_trace(
                go.Scatter(x=viz_data['Scheduled_relevant_delivery_date'],
                           y=viz_data['PO_Number'],
                           name=str(entity)
                           ))
        dl_fig2.update_xaxes(title_text="Months")
        dl_fig2.update_yaxes(title_text="Number of PO Lines")
        dl_fig2.update_layout(title="Number of Deliveries over Time")
        st.plotly_chart(dl_fig2, use_container_width=True, show_legend=True)


# Show a scatter plot that determines if certain entities are meeting Steelcase's performance standards.
def scatter_View():

    # Show a question when full supply chain is not selected
    if lod!= 'Full Supply Chain':
        with filter_col:
            show_all = st.checkbox(f'Show all {lod}s?')

    #Get data
    requested_data = data.SplitData(start_date, end_date)
    requested_data['late_bin'] = np.where(requested_data['days_late'] > 0, 1, 0)

    # Handle Full supply chain case
    if lod == 'Full Supply Chain':
        late_perc_data = requested_data['late_bin'].mean()*100
        late_days_data = requested_data[(requested_data['days_late'] > 0)]
        late_severity_data = late_days_data['days_late'].mean()
        size_data = len(requested_data['PO_Number'].unique())
        data_test = [[late_perc_data, late_severity_data, size_data]]
        comb_df = pd.DataFrame(data_test, columns = ['Percentage Late', 'Average Days Late', '# Of Times Ordered'])
        comb_df['Average Days Late'] = np.where(
            (comb_df['Average Days Late'].isna()) | (comb_df['Average Days Late'] == 0),
            0, comb_df['Average Days Late'])

        # Determine colors of points
        conditions = [
            comb_df['Average Days Late'] > 7,
            comb_df['Percentage Late'] > 10,
            comb_df['Average Days Late'] <= 7,
            comb_df['Percentage Late'] <= 10
        ]
        values = [
            'Not Meeting Expectations', 'Not Meeting Expectations',
            'Meeting Expectations', 'Meeting Expectations'
        ]
        # Used to determine sizes of points
        def logger(x):
            return x**(1/1.5)
        comb_df['logscale'] = comb_df['# Of Times Ordered'].apply(logger)
        comb_df[f'Full Supply Chain Performance'] = np.select(conditions, values)
        late_std = comb_df['Average Days Late'].std()
        late_mean = comb_df['Average Days Late'].mean()
        colors = {'Not Meeting Expectations': 'red', 'Meeting Expectations': 'green'}

        # Plot all the points
        fig = px.scatter(comb_df, x='Percentage Late', y='Average Days Late',
                         title=f'Full Supply Chain Performance', size='logscale',
                         color = comb_df[f'Full Supply Chain Performance'], color_discrete_map= colors,
                         )
        fig.add_shape(type='rect', x0=0, x1=10, y0=0, y1=7, line=dict(color='green'))
        fig.update_layout(yaxis_range=[0, late_mean + (late_std*1.5)])
        fig.update_layout(xaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, show_legend=True)
        return

    # If user wants to show all entities in a group
    if show_all:
        # Get data
        late_perc_data = requested_data.groupby(entity_map[lod], as_index=False, dropna=False)['late_bin'].mean()
        late_days_data = requested_data[(requested_data['days_late'] > 0)]
        size_data = requested_data.groupby(entity_map[lod], as_index=False, dropna=False)['PO_Number'].count()
        late_severity_data = late_days_data.groupby(entity_map[lod], as_index=False, dropna=False)['days_late'].mean()
        comb_df = pd.DataFrame({lod: late_perc_data[entity_map[lod]],
                               'Average Days Late': late_severity_data['days_late'],
                               'Percentage Late': late_perc_data['late_bin']*100,
                               '# Of Times Ordered': size_data['PO_Number']})
        comb_df['Average Days Late'] = np.where((comb_df['Average Days Late'].isna()) | (comb_df['Average Days Late']==0),
                                                0, comb_df['Average Days Late'])
        # Set up colors of points
        conditions = [
            comb_df['Average Days Late'] > 7,
            comb_df['Percentage Late'] > 10,
            comb_df['Average Days Late'] <= 7,
            comb_df['Percentage Late'] <= 10
        ]
        values = [
            'Not Meeting Expectations', 'Not Meeting Expectations',
            'Meeting Expectations', 'Meeting Expectations'
        ]

        # Used to determine sizes of points
        def logger(x):
            return x**(1/1.5)
        comb_df['logscale'] = comb_df['# Of Times Ordered'].apply(logger)
        comb_df[f'{lod} Performance'] = np.select(conditions, values)
        late_std = comb_df['Average Days Late'].std()
        late_mean = comb_df['Average Days Late'].mean()
        colors = {'Not Meeting Expectations': 'red', 'Meeting Expectations': 'green'}

        # Plot all the points
        fig = px.scatter(comb_df, x='Percentage Late', y='Average Days Late',
                         title=f'{lod} Performance', size='logscale',
                         color = comb_df[f'{lod} Performance'], color_discrete_map= colors,
                         hover_name=f'{lod}')
        fig.add_shape(type='rect', x0=0, x1=10, y0=0, y1=7, line=dict(color='green'))
        fig.update_layout(yaxis_range=[0, late_mean + (late_std*1.5)])
        fig.update_layout(xaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, show_legend=True)

    else:
        # Get data
        requested_data2 = requested_data[requested_data[entity_map[lod]].isin(lod_filter)]
        late_perc_data = requested_data2.groupby(entity_map[lod], as_index=False, dropna=False)['late_bin'].mean()
        late_days_data = requested_data2[(requested_data2['days_late'] > 0)]
        size_data = requested_data2.groupby(entity_map[lod], as_index=False, dropna=False)['PO_Number'].count()
        late_severity_data = late_days_data.groupby(entity_map[lod], as_index=False, dropna=False)['days_late'].mean()
        comb_df = pd.DataFrame({lod: late_perc_data[entity_map[lod]],
                                'Average Days Late': late_severity_data['days_late'],
                                'Percentage Late': late_perc_data['late_bin']*100,
                                '# Of Times Ordered': size_data['PO_Number']})
        comb_df['Average Days Late'] = np.where(
            (comb_df['Average Days Late'].isna()) | (comb_df['Average Days Late'] == 0),
            0, comb_df['Average Days Late'])

        # Determine colors of points
        conditions = [
            comb_df['Average Days Late'] > 7,
            comb_df['Percentage Late'] > 10,
            comb_df['Average Days Late'] <= 7,
            comb_df['Percentage Late'] <= 10
        ]
        values = [
            'Not Meeting Expectations', 'Not Meeting Expectations',
            'Meeting Expectations', 'Meeting Expectations'
        ]

        # Determines size of points
        def logger(x):
            return x ** (1 / 1.5)

        comb_df['logscale'] = comb_df['# Of Times Ordered'].apply(logger)
        comb_df[f'{lod} Performance'] = np.select(conditions, values)
        late_max = comb_df['Average Days Late'].max()
        colors = {'Not Meeting Expectations': 'red', 'Meeting Expectations': 'green'}

        # Plot the points
        fig = px.scatter(comb_df, x='Percentage Late', y='Average Days Late',
                         title=f'{lod} Performance', size='logscale',
                         color = comb_df[f'{lod} Performance'], color_discrete_map= colors,
                         hover_name=f'{lod}')
        fig.add_shape(type='rect', x0=0, x1=10, y0=0, y1=7, line=dict(color='green'))
        fig.update_layout(yaxis_range=[0, max(10,late_max*1.25)])
        fig.update_layout(xaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, show_legend=True)

# Show View
with viz_col:
    if view == "Performance over time":
        performance_over_time()
    elif view == 'Number of deliveries over time':
        number_of_deliveries()
    elif view == 'Scatter view':
        scatter_View()

# Create metrics
def metrics():

    # Get data
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

    # Set up the format of the page and get data
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
        num_supp = len(metric_data['Supplier'].value_counts())
        if len(prev_data) == 0:
            old_num_supp = num_supp
        else:
            old_num_supp = len(prev_data['Supplier'].value_counts())
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

    # Display error issue
    else:
        st.error(f'There is currently no data available for the chosen filter and/or time frame. '
                 f'Please select at least 1 {lod} from the Filter dropdown or adjust the time frame.', icon = '🚨')

    gui.colored_header(label='', description='', color_name='blue-30')

# Calling metrics function
metrics()