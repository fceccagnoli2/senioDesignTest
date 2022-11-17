import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import sqlite3
from datetime import date, timedelta
from PIL import Image
import os
# os.chdir("Analytics-Tool/tool_dev/app")
# sys.path.append(os.path.norath(os.getcwd() + os.sep + os.pardir))
from data import data_prep
from data import data_prep
from data import consts

# Main directory path ('Tool Dev')
# main_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
# st.session_state['directory'] = main_dir

#Chnage

# data = pd.read_csv('data/model_data3.csv')
# Cache Data
@st.cache(allow_output_mutation=True)
# Function for ingesting data
def read_data():
    
    # Need to find the correct file path for referencing functions and data
    data_path = 'Analytics-Tool/tool_dev/app/data/model_data3.csv'
    # Call data class and parameters from data module
    data_obj = data_prep.IngestionPipe(features=consts.features, target=consts.target, dates=consts.dates,
                                       post_date=consts.clean_end_date, pre_date=consts.clean_start_date)
    data_obj.dataAllocation(path=data_path)
    return data_obj

#Get Today
today = date.today()
st.session_state['today'] = today

# Load in and Save Data
data = read_data()
st.session_state['data'] = data

# Trying to Put Logo in Top Right (Can use markdown)
col1, col2, col3 = st.columns(3, gap='large')
steelcaseLogo = Image.open(
    'assets/SteelcaseLogo.png')
with col3:
    st.image(steelcaseLogo, width=100)


## Format Title ##
def head():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Supply Chain Overview
        </h1>

    """, unsafe_allow_html=True
                )
head()

# Build Main view
with st.form('inputs', clear_on_submit=True):
    # Date submission form
    in_col1, in_col2 = st.columns(2, gap='small')
    with in_col1:
        start_date = st.text_input('Start Date (e.g. 2019-01-01)', '2019-01-01')
    with in_col2:
        end_date = st.text_input('End Date (e.g. ' + str(today) + ')', today)

    st.write('The current time frame is: ' + start_date + '  to  ' + end_date)
    submitted = st.form_submit_button("Analyze")

# Handle Analyze Button
if submitted:
    viz_data = data.SplitData(start_date, end_date)
    viz_data['late_bin'] = np.where(viz_data['days_late']>0,1,0)
    first_date = pd.to_datetime(start_date, format='%Y-%m-%d')
    second_date = pd.to_datetime(end_date, format='%Y-%m-%d')
    date_difference = second_date - first_date
    previous_date = first_date - date_difference
    min_date = pd.to_datetime('2019-01-01', format='%Y-%m-%d')
    if previous_date < min_date:
        previous_date = min_date
    prev_data = data.SplitData(previous_date,start_date)
    # Metrics
    col1, col2, col3 = st.columns(3)

    # Metric 1: On-time percentage
    on_time_perc = round(100 - 100 * len(viz_data[(viz_data["days_late"] > 0)]) / len(viz_data), 2)
    if len(prev_data) == 0:
        old_on_time_perc = on_time_perc
    else:
        old_on_time_perc = round(100 - 100 * len(prev_data[(prev_data["days_late"] > 0)]) / len(prev_data), 2)
    col1.metric(label="On-Time Delivery Percentage:", value=(str(on_time_perc) + '%'),
                    delta=str(round(on_time_perc - old_on_time_perc, 2)) + '%')

    # Metric 2: Number of POs completed
    distinct_POs = len(viz_data['PO_Number'].value_counts())
    if len(prev_data) == 0:
        old_distinct_POs = distinct_POs
    else:
        old_distinct_POs = len(prev_data['PO_Number'].value_counts())
    col2.metric(label='Completed POs', value= "{:,}".format(int(distinct_POs)), delta="{:,}".format(int(distinct_POs - old_distinct_POs)))

    # Metric 3: Number of Suppliers
    num_supp = len(viz_data['Vendor_number'].value_counts())
    if len(prev_data) == 0:
        old_num_supp = num_supp
    else:
        old_num_supp = len(prev_data['Vendor_number'].value_counts())
    col3.metric(label='Suppliers Used', value = "{:,}".format(int(num_supp)), delta="{:,}".format(int(num_supp - old_num_supp)))

    # Metric 4 - Cost Data
    invoice_cost = viz_data['Invoice ($)'].sum()
    if len(prev_data) == 0:
        old_invoice_cost = invoice_cost
    else:
        old_invoice_cost = prev_data['Invoice ($)'].sum()
    st.metric(label='Total Invoice Cost',
            value="{:,}".format(round(invoice_cost), 2),
            delta="{:,}".format(round(invoice_cost - old_invoice_cost), 2))

    # Line Chart for On-Time Delivery Percentage By Month
    on_p1 = viz_data.resample('W', on = 'Scheduled_relevant_delivery_date').late_bin.mean() * 100
    on_p1 = on_p1.rolling(4).mean()
    dl_fig1 = px.line(x=on_p1.index, y= on_p1, title='Late Order Percentage',
                     labels = {'x': 'Scheduled Delivery Date', 'y' : 'Percentage of Late Deliveries'})
    st.plotly_chart(dl_fig1, use_container_width=True)

    error_codes = st.expander("Occurences of Error Codes", expanded=False)
    with error_codes:
        # Bar Chart For Error Codes - Missing Error Codes
        error_count = viz_data['error_code'].value_counts()
        error_count = pd.DataFrame(error_count)
        conditions = [
            error_count.index == 'I',
            error_count.index == 'U',
            error_count.index == 'D',
            error_count.index == 'W',
            error_count.index == 'A',
            error_count.index == 'C',
            error_count.index == 'B',
            error_count.index == 'F',
            error_count.index == 'E',
            error_count.index == 'V',
            error_count.index == 'G',
            error_count.index == 'H',
            error_count.index == 'T',
            error_count.index == 'Y',
            error_count.index == 'Z'
        ]
        values = ['I: Pending Receipt', 'U: Late Order', 'D: Supplier Was Late', 'W: Business Continuity', 'A: Receiving Delivery Issues',
                  'C: Transportation Issues', 'B: Planning/Scheduling Issues', 'F: ISC Late', 'E: Less Than Vendor Lead Time',
                  'V: Vendor Service COM/CSM', 'G: Lost In Transit', 'H: Obselete - Do Not Use', 'T: Transportation Damage',
                  'Y: Yard Management Error', 'Z: Customers Order Delay']
        error_count['error_code_description'] = np.select(conditions, values)
        dl_fig = px.bar(x=error_count.index, y=error_count['error_code'],
                        title = 'Occurences of Error Codes',
                        labels = {'x': 'Error Codes', 'y': 'Occurences'},
                        color = error_count['error_code_description'])
        st.plotly_chart(dl_fig, use_container_width=True)

    mat_group_exp = st.expander("Material Group Performance")
    with mat_group_exp:
        mat_col1,mat_col2 = st.columns(2)
        with mat_col1:
            # Data For Poorly Performing Material Groups
            late_perc_data = viz_data.groupby('Material_Group',as_index = False,dropna = False).mean()
            late_days_data = viz_data[(viz_data['days_late'] > 0)]
            size = viz_data.groupby('Material_Group', as_index = False, dropna = False).count()
            money = viz_data.groupby('Material_Group', as_index=False, dropna = False).sum()
            late_group_data = late_days_data.groupby('Material_Group', as_index=False).mean()
            mat_group_df = pd.DataFrame({'Material Group' : late_perc_data['Material_Group'],
                                         'Average Days Late' : late_group_data['days_late'],
                                         'Percentage Late' : late_perc_data['late_bin'],
                                         '# Of Times Ordered': size['Material'],
                                         '$ Spent per Group': money['Invoice ($)']})

            st.write(mat_group_df)
        conditions = [
            mat_group_df['Average Days Late'] > 10,
            mat_group_df['Percentage Late'] > 0.15,
            mat_group_df['Average Days Late'] <= 10,
            (mat_group_df['Percentage Late'] <= 0.15)
        ]
        values = [
            'Not Meeting Expectations','Not Meeting Expectations',
            'Meeting Expectations', 'Meeting Expectations'
        ]
        mat_group_df['Material Group Performance'] = np.select(conditions, values)
        mat_group_df = mat_group_df.loc[mat_group_df['$ Spent per Group'] != 0]
        with mat_col2:
            #Visualization For Dataframe Above
            mat_group_df['color'] = np.where(
                (mat_group_df['Percentage Late'] < .05) & (mat_group_df['Average Days Late'] < 10), 'Best',
                np.where(mat_group_df['Percentage Late'] < .05, 'Good',
                         np.where(mat_group_df['Average Days Late'] < 10, 'Bad', 'Worst')))
            fig = px.scatter(mat_group_df, x = 'Percentage Late', y = 'Average Days Late',
                             # color_discrete_map={"Best": 'green','Good':'blue','Bad':'orange','Worst':'red'},
                             title = 'Material Group Performance', size = '$ Spent per Group',
                             color = "Material Group Performance", color_discrete_sequence = ['red','green'],
                             hover_name = 'Material Group',size_max = 30)
            fig.add_shape(type='rect', x0=0, x1=0.15, y0=0, y1=10, line=dict(color='green'))
            st.write(fig)

    mat_exp = st.expander("Material Performance")
    with mat_exp:
        mat_grp_col1, mat_grp_col2 = st.columns(2)
        with mat_grp_col1:
            late_mat_perc_data = viz_data.groupby('Material', as_index = False, dropna = False).mean()
            late_mat_days_data = viz_data[(viz_data['days_late'] > 0)]
            mat_size = viz_data.groupby('Material', as_index=False, dropna = False).count()
            late_mat_data = late_mat_days_data.groupby('Material', as_index=False, dropna = False).mean()
            mat_money = viz_data.groupby('Material', as_index=False, dropna = False).sum()
            mat_df = pd.DataFrame({'Material': late_mat_perc_data['Material'],
                                        'Average Days Late': late_mat_data['days_late'],
                                        'Percentage Late': late_mat_perc_data['late_bin'],
                                        '# Of Times Ordered': mat_size['Material_Group'],
                                        '$ Spent per Material': mat_money['Invoice ($)']})
            st.write(mat_df)

        conditions = [
            mat_df['Average Days Late'] > 10,
            mat_df['Percentage Late'] > 0.15,
            mat_df['Average Days Late'] <= 10,
            mat_df['Percentage Late'] <= 0.15
        ]
        values = [
            'Not Meeting Expectations','Not Meeting Expectations',
            'Meeting Expectations', 'Meeting Expectations'
        ]
        mat_df['Material Performance'] = np.select(conditions, values)
        mat_df = mat_df.loc[mat_df['$ Spent per Material'] != 0]
        with mat_grp_col2:
            mat_df['color'] = np.where(
                (mat_df['Percentage Late'] < .05) & (mat_df['Average Days Late'] < 10), 'Best',
                np.where(mat_df['Percentage Late'] < .05, 'Good',
                         np.where(mat_df['Average Days Late'] < 10, 'Bad', 'Worst')))
            fig = px.scatter(mat_df, x = 'Percentage Late', y = 'Average Days Late',
                             # color_discrete_map={"Best": 'green', 'Good': 'blue', 'Bad': 'orange', 'Worst': 'red'},
                             title = 'Material Performance', size = '$ Spent per Material',
                             color = 'Material Performance', color_discrete_sequence = ['red', 'green'],
                             hover_name = 'Material',size_max = 15)
            fig.add_shape(type='rect', x0=0, x1=0.15, y0=0, y1=10, line=dict(color='green'))
            st.write(fig)

    sup_exp = st.expander("Supplier Performance")
    with sup_exp:
        sup_1,sup_2 = st.columns(2)
        with sup_1:
            late_sup_perc_data = viz_data.groupby('Supplier', as_index = False, dropna = False).mean()
            late_sup_days_data = viz_data[(viz_data['days_late'] > 0)]
            sup_size = viz_data.groupby('Supplier', as_index=False, dropna = False).count()
            late_sup_data = late_mat_days_data.groupby('Supplier', as_index=False, dropna = False).mean()
            sup_money = viz_data.groupby('Supplier', as_index=False, dropna=False).sum()
            sup_df = pd.DataFrame({'Supplier': late_sup_perc_data['Supplier'],
                                         'Average Days Late': late_sup_data['days_late'],
                                         'Percentage Late': late_sup_perc_data['late_bin'],
                                         '# Of Times Ordered': sup_size['Material_Group'],
                                         '$ Spent per Supplier': sup_money['Invoice ($)']})
            st.write(sup_df)

        conditions = [
            sup_df['Average Days Late'] > 10,
            sup_df['Percentage Late'] > 0.15,
            sup_df['Average Days Late'] <= 10,
            sup_df['Percentage Late'] <= 0.15
        ]
        values = [
            'Not Meeting Expectations','Not Meeting Expectations',
            'Meeting Expectations', 'Meeting Expectations'
        ]
        sup_df['Supplier Performance'] = np.select(conditions, values)
        sup_df = sup_df.loc[sup_df['$ Spent per Supplier'] != 0]
        with sup_2:
            sup_df['color'] = np.where(
                (sup_df['Percentage Late'] < .05) & (sup_df['Average Days Late'] < 10), 'Best',
                np.where(sup_df['Percentage Late'] < .05, 'Good',
                         np.where(sup_df['Average Days Late'] < 10, 'Bad', 'Worst')))
            fig = px.scatter(sup_df, x='Percentage Late', y='Average Days Late',
                             # color_discrete_map={"Best": 'green', 'Good': 'blue', 'Bad': 'orange', 'Worst': 'red'},
                             title='Supplier Performance', size='$ Spent per Supplier',
                             color = 'Supplier Performance', color_discrete_sequence= ['red', 'green'],
                             hover_name='Supplier', size_max=30)
            fig.add_shape(type = 'rect', x0 = 0, x1 = 0.15, y0=0 ,y1 = 10, line = dict(color = 'green'))
            st.write(fig)
