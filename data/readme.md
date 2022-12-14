# Data 

This directory contains data used to create the streamllit app, as well as relevant research into how to create many of the functions the app posesses, such as the recommendations.

- <ins> Auto_Arima_Forecasting.ipynb </ins>: A notebook that contains research on how to better forecast supplier performance.
- <ins> consts.py </ins>: Is a file containing lists of data to be used in the data object creation.
- data_prep.py Is a file that is used to clean and prep data for modeling purposed and for streamlit injection.
- forecastingcodev2.py Is another later iteration of the code in Auto_Arima_Forecasting.ipynb but in script form
- model_data, model_data1, model_data2, model_data3, Are all files containing iterations of data used to build the streamlit app.
    model_data3 is the data set which the app currently uses.
- MVEF.py is the python script housing the functions to generate and interpret the mean-variance results for suppliers in a material group. This is used in the streamlit app recommendations page.
- MVEF_test.ipynb is file that was used to test the accuracy of the MVEF.py file
- Price_and_Quantity_Correlation.ipynb is a python file that was used to research correlation between the price of a PO and that POs timeliness
- Report_viz.ipynb is a file that was used to generate images found in the final report.
- Test_data_prep.ipynb is a file that was used to test the data prep and injestion process.
- distribution_fitting is a directory used to generate further recommendations results. It was used for research purposes
