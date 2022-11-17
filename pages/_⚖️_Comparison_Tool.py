import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import plotly.express as px
import os, sys
# sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
from data import forecastingcodev2
from plotly.subplots import make_subplots

##Trying to Put Logo in Top Right (Can use markdown)##
col1,col2,col3 = st.columns(3, gap = 'large')
# main_dir = st.session_state['directory']
steelcaseLogo = Image.open('assets/SteelcaseLogo.png')
with col3:
    st.image(steelcaseLogo, width = 100)

# Load Data
data = st.session_state['data']
data_frame = data.data
data_frame['late_bin'] = np.where(data_frame['days_late']>0,1,0)

## Format Title ##
def head():
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Compare Suppliers
        </h1>

    """, unsafe_allow_html=True
                )
head()


incol1, incol2, incol3, incol4 = st.columns(4, gap = 'small')
error = False
with incol1:
    vendor = st.text_input('Vendor Number', '556264')
    if vendor != '':
        vendor_frame = data_frame[data_frame['Vendor_number'] == int(vendor)]
        if len(vendor_frame) == 0:
            st.write('Vendor number is not recognized')
        else:
            supplier_name = vendor_frame['Supplier'].unique()[0]
    else:
        st.write('Enter a vendor number')

with incol2:
    if vendor != '':
        vendor_frame = data_frame[data_frame['Vendor_number'] == int(vendor)]
        mat_groups = vendor_frame['Material_Group'].unique()
        material_group = st.selectbox(
            'Which material group are you interested in?',
            mat_groups
        )
        if material_group == None:
            error = True
    else:
        error = True
with incol3:
    switch_cost= st.text_input('Switching cost', '35000')
    if switch_cost == '':
        error = True
        st.write('Please input a switching cost')
    else:
        switch_cost = int(switch_cost)
with incol4:
    lcost = st.text_input('Cost', '300')
    if lcost == '':
        error = True
        st.write('Please input a cost')

if (error == False):
    st.write(f'You have selected Supplier: {supplier_name}' + ' and Material Group: ' + str(material_group))
    # Compare Suppliers
    material_frame = data_frame[data_frame['Material_Group'] == material_group]
    suppliers = material_frame['Vendor_number'].unique()
    supplierNames = pd.merge(pd.DataFrame(suppliers, columns = ['Vendor_number']), material_frame, how='left', on='Vendor_number')['Supplier'].unique()
    supplier_proj = {}
    for supplier in suppliers:
        sup_mat_frame = material_frame[material_frame['Vendor_number'] == supplier]
        try:
            forecasted_3_months = forecastingcodev2.forecasting(sup_mat_frame, 3, 'Supplier and Materiel Group')
            forecasted_3_months = forecasted_3_months[0]
            supplier_proj[supplier] = forecasted_3_months['predicted_percentage_late'].mean()
        except:
            temp_supplier_name = sup_mat_frame['Supplier'].unique()[0]
            st.error("WARNING: Not enough data to give an accurate forecast for " + str(temp_supplier_name), icon = '🚨')
            supplier_proj[supplier] = None
    quantity_estimator = vendor_frame[vendor_frame['Material_Group'] == material_group]
    quantity_estimator = quantity_estimator.resample('M', on='Scheduled_relevant_delivery_date')['Purchase Order Qty'].mean()
    # st.write(quantity_estimator)
    q_estimate = quantity_estimator.mean()
    #st.write(supplier_proj)
    cost_data = {
        'Material Group':[material_group]*len(suppliers),
        'Supplier': supplierNames,
        'Supplier #':supplier_proj.keys(),
        'Expected Late Percentage':supplier_proj.values(),
                 }
    cost_frame = pd.DataFrame(cost_data)
    if pd.isna(q_estimate):
        q_estimate = 0
    cost_frame['Expected Late Percentage'] = np.where(cost_frame['Expected Late Percentage'] < 0, 0, cost_frame['Expected Late Percentage'])
    cost_frame['Expected Late Percentage'] = np.where(cost_frame['Expected Late Percentage'] > 1, 1, cost_frame['Expected Late Percentage'])
    cost_frame['Switching Cost'] = np.where(cost_frame['Supplier #'] == int(vendor), 0, switch_cost)
    cost_frame['Supplier Cost'] = int(q_estimate) * int(lcost) * cost_frame['Expected Late Percentage']
    cost_frame['Marginal Savings'] = cost_frame['Supplier Cost'] - cost_frame[cost_frame['Supplier #'] == int(vendor)]['Supplier Cost']
    base = cost_frame[cost_frame['Supplier #'] == int(vendor)].reset_index()['Supplier Cost'][0]
    cost_frame['Marginal Savings'] = np.where(cost_frame['Marginal Savings'].isna(), cost_frame['Supplier Cost'] - base, cost_frame['Marginal Savings'])
    cost_frame['Marginal Savings'] = cost_frame['Marginal Savings']*-1
    cost_frame['Break Even Months'] = cost_frame['Switching Cost']/cost_frame['Marginal Savings']
    cost_frame['Break Even Months'] = np.where(cost_frame['Break Even Months'] < 0, 'Never',
                                               cost_frame['Break Even Months'])
    cost_frame['Break Even Months'] = np.where(cost_frame['Break Even Months'].isnull(), 0,
                                               cost_frame['Break Even Months'])
    cost_frame['Expected Late Percentage'] = cost_frame['Expected Late Percentage'] * 100
    st.dataframe(cost_frame, width = 1000)

    # Supplier Performance Visual
    st.markdown("""
        <h1 style='text-align: center; margin-bottom: -35px;'>
        Supplier Performance For Material Group
        </h1>

    """, unsafe_allow_html=True
                )

    dl_fig1 = make_subplots(specs=[[{"secondary_y": False}]])
    sup_mat_frame = material_frame[material_frame['Vendor_number'] == int(vendor)]
    sup_mat_frame = sup_mat_frame.resample('M', on='Scheduled_relevant_delivery_date').late_bin.mean() * 100
    sup_mat_frame = sup_mat_frame.reset_index()
    sup_mat_frame['Vendor_number'] = vendor
    sup_mat_frame['Material_Group'] = material_group
    try:
        forecast_test = forecastingcodev2.forecasting(sup_mat_frame, 5, 'Supplier and Materiel Group')
        projections = forecast_test[0]

        projections = projections.rename(columns={'predicted_percentage_late': 'Late Percentage',
                                        'time': 'Scheduled Delivery Date'})
        projections['Forecast'] = 'Forecast'
        projections['Late Percentage'] = projections['Late Percentage']
        sup_mat_frame = sup_mat_frame.rename(
            columns={'Scheduled_relevant_delivery_date': 'Scheduled Delivery Date',
                     'late_bin': 'Late Percentage'})
        projections = pd.concat([sup_mat_frame.tail(1), projections]).reset_index(drop=True)
        dl_fig1.add_scatter(
            x=projections['Scheduled Delivery Date'],
            y=projections['Late Percentage'],
            name='Forecasted Performance',
            secondary_y=False)
    except:
        st.error("WARNING: Not enough data to give an accurate forecast for " + str(supplier_name), icon = '🚨')
    sup_mat_frame = sup_mat_frame.rename(
        columns={'Scheduled_relevant_delivery_date': 'Scheduled Delivery Date',
            'late_bin': 'Late Percentage'})
    dl_fig1.add_scatter(
        x=sup_mat_frame['Scheduled Delivery Date'],
        y=sup_mat_frame['Late Percentage'],
        name='Real Performance',
        showlegend = True,
        secondary_y=False)
    dl_fig1.update_xaxes(title_text="Months")
    dl_fig1.update_yaxes(title_text="Late Delivery Percentage")
    dl_fig1.update_layout(title="Late Delivery Percentages for " + str(supplier_name) + ' in Material Group ' + str(material_group))
    st.plotly_chart(dl_fig1, use_container_width=True, show_legend=True)