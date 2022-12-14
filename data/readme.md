# Data 

This directory contains data used to create the streamllit app, as well as relevant research into how to create many of the functions the app posesses, such as the recommendations.

- <ins> Auto_Arima_Forecasting.ipynb </ins>: A notebook that contains research on how to better forecast supplier performance.
- <ins> consts.py </ins>: Is a file containing lists of data to be used in the data object creation.
- <ins> data_prep.py </ins>: Is a file that is used to clean and prep data for modeling purposed and for streamlit injection.
- <ins> forecastingcodev2.py </ins>: Is another later iteration of the code in Auto_Arima_Forecasting.ipynb but in script form
- <ins> model_data, model_data1, model_data2, model_data3 </ins>: Are all files containing iterations of data used to build the streamlit app.
    model_data3 is the data set which the app currently uses.
- <ins> MVEF.py </ins>: Is the python script housing the functions to generate and interpret the mean-variance results for suppliers in a material group. This is used in the streamlit app recommendations page.
- <ins> MVEF_test.ipynb </ins>: Is file that was used to test the accuracy of the MVEF.py file
- <ins> Price_and_Quantity_Correlation.ipynb </ins>: Is a python file that was used to research correlation between the price of a PO and that POs timeliness
- <ins> Report_viz.ipynb </ins>: Is a file that was used to generate images found in the final report.
- <ins> Test_data_prep.ipynb </ins>: Is a file that was used to test the data prep and injestion process.
- <ins> distribution_fitting </ins>: Is a directory used to generate further recommendations results. It was used for research purposes
