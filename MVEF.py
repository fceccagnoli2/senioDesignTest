# Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import itertools
import os, sys
from pandas.tseries.offsets import MonthEnd

# Import Data
# sys.path.append(os.path.normpath(os.getcwd() + os.sep + os.pardir))
# main_dir = os.path.normpath(os.getcwd() + os.sep + os.pardir)
# data_path = main_dir + '/data/model_data3.csv'
# data = pd.read_csv(data_path)
#
# # Parameters
# mg = '057'
# thresh_n = 15
# increments = [1,.9,.8,.7,.6,.5,.5,.4,.4,.3,.3,.3,.2,.2,.2,.2,.2,.1,.1,.1,.1,.1,.1,.1,.1,.1,.1,0,0,0,0,0,0,0,0,0,0]

# Process Data
def data_process(mg, thresh_n, data):
    # Create initial data frame
    mg_data = data[data['Material_Group'] == mg]
    mg_data['late_bin'] = np.where(mg_data['days_late'] > 0, 1, 0)
    mg_data['PO_Document_Date'] = pd.to_datetime(mg_data['PO_Document_Date'], format='%Y-%m-%d')
    grouped = mg_data.groupby('Supplier').resample('M', on='PO_Document_Date')
    late_p = grouped.mean()['late_bin'].reset_index()
    late_p['OTP'] = 1 - late_p['late_bin']
    # Continuous data axis
    dates = list(late_p['PO_Document_Date'].unique())
    MVEF_Ingest = pd.DataFrame({'Month': dates})
    # For every supplier, left join so that we can see where the date gaps are
    keep_columns = ['Month']
    perf_columns = []
    for supplier in set(late_p['Supplier']):
        sup_data = late_p[late_p['Supplier'] == supplier][['PO_Document_Date', 'OTP']]
        non_nulls = [x for x in sup_data['OTP'] if x > -1000]
        if len(non_nulls) < thresh_n:
            continue
        sup_data = sup_data.rename(
            columns={'OTP': f'{supplier} OTP', 'PO_Document_Date': f'{supplier}_date'})
        keep_columns.append(f'{supplier} OTP')
        perf_columns.append(f'{supplier} OTP')
        MVEF_Ingest = pd.merge(MVEF_Ingest, sup_data, how='left', left_on='Month', right_on=f'{supplier}_date')
    MVEF_Ingest = MVEF_Ingest[keep_columns]
    MVEF_Clean = MVEF_Ingest[perf_columns].interpolate(method='linear', axis=0, limit_direction='forward')
    MVEF_Clean['Month'] = MVEF_Ingest['Month']
    MVEF_Clean = MVEF_Clean.dropna()
    MVEF_Clean = MVEF_Clean[keep_columns]
    # Find Covariance Matrix
    sigma = MVEF_Clean[perf_columns].cov()
    # MVP and Minimum Variance
    inverse_sigma = pd.DataFrame(np.linalg.pinv(sigma.values), sigma.columns, sigma.index)
    lambda_star = (1 / inverse_sigma.dot([1] * len(sigma)).sum())
    sqrt_lamda_star = lambda_star ** (1 / 2)  # Minimum std that can be achieved
    # minimum variance portfolio
    mvp = (lambda_star * inverse_sigma.dot([1] * len(sigma)))
    return MVEF_Clean, sigma, perf_columns, sqrt_lamda_star, mvp

# Function to create weight combinations to test
def portfolio_generator(columns, increments):
    subsets = set()
    for subset in tqdm(itertools.permutations(increments, len(columns))):
        if sum(subset) == 1:
            subsets.add(subset)
    weights = {}
    for index, column in enumerate(columns):
        weights[column] = [x[index] for x in subsets]
    w_martrix = pd.DataFrame(weights)
    return w_martrix

# Function to find mean, variance, and std of each portfolio
def point_generator(w_matrix, cov_matrix, MVEF_Clean):
    points = {}
    for point in range(0, len(w_matrix)):
        point_var = (w_matrix.loc[point]*cov_matrix.dot(w_matrix.loc[point])).sum()
        point_std = point_var**(1/2)
        point_mean = (w_matrix.loc[point] * MVEF_Clean.mean(numeric_only=True)).sum()
        points[point] = (point_mean, point_std)
    MVEF = pd.DataFrame({'Point':points.keys(),
                         'Mean':[x[0] for x in points.values()],
                         'Std':[x[1] for x in points.values()],
                         })
    return MVEF

# Compare a MVEF Point performance at a certain point in time to actual performance
def evaluate_portfolio(point, w_matrix, raw_data, MVEF_Clean, perf_columns, mg, decision_date, delay):
    # Enforce data type
    decision_date = pd.to_datetime(decision_date, format='%Y-%m-%d')
    # Find actual performance
    mg_group = raw_data[raw_data['Material_Group'] == mg]
    mg_group['late_bin'] = np.where(mg_group['days_late'] > 0, 1, 0)
    mg_group['PO_Document_Date'] = pd.to_datetime(mg_group['PO_Document_Date'], format='%Y-%m-%d')
    grouped = mg_group.resample('M', on='PO_Document_Date')
    late_p = grouped.mean()['late_bin'].reset_index()
    number_POs = grouped.nunique()['PO_Number'].reset_index()
    number_POs = number_POs.rename(columns={'PO_Document_Date': 'Date'})
    useful_data = pd.merge(late_p, number_POs, how='left', left_on='PO_Document_Date', right_on='Date')[
        ['Date', 'late_bin', 'PO_Number']]
    useful_data['Actual_Late_POs'] = useful_data['PO_Number'] * useful_data['late_bin']
    # Now Find What Performance of the new portfolio would be
    portfolio = w_matrix.loc[point]
    portfolio_performance = pd.DataFrame({
        'port_Date': MVEF_Clean['Month'],
        'Portfolio Performance': (MVEF_Clean[perf_columns] * portfolio).sum(axis=1)
    })
    useful_data = pd.merge(useful_data, portfolio_performance, how='left', left_on='Date', right_on='port_Date')
    useful_data['Portolio_Late_POs'] = useful_data['PO_Number'] * (1 - useful_data['Portfolio Performance'])
    useful_data['Actual_Late_POs'] = useful_data['Actual_Late_POs'].apply(np.ceil)
    useful_data['Portolio_Late_POs'] = useful_data['Portolio_Late_POs'].apply(np.ceil)
    # Compute when decision would take effect
    effective_date = decision_date + MonthEnd(delay + 1)
    # Now find what performance what have been over time given data, portfolio, decision time, and delay
    useful_data['Implemented POs'] = np.where(useful_data['Date'] >= effective_date,
                                              useful_data['Portolio_Late_POs'],
                                              np.NAN
                                              )
    return useful_data