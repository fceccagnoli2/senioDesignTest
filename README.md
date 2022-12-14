# Senior-Design Deliverable Master Directory
 This is the directory housing all of the files used in the creation of the Senior Design deliverable for Team 17.
 A breakdown of all sub-directories and files:

- <ins> Data IO</ins> : Directory used for code that enables users to upload new data to streamlit. This directory was used for research. The actual code responsible for updating data is located in the pages directory in the data_input.py file.
- <ins>DataSetup</ins>: Is a directory used for building a database for data given to us by Steelcase. This directory is no longer active and was only used for research.
- <ins>Deliverables</ins>: A directory used for housing deliverables to Steelcase, such as the Final Report, the Presentation, and a Poster Summary of the project.
- <ins>Jupyter_NB_Research</ins>: A directory containing very early scratch work. All relevant research can be found in the data directory or or in the  ML_Model_Dev directory.
- <ins>MIP_Model_Dev</ins>: A directory used for research efficacy of an MIP model to recommend suppliers. No longer active. 
- <ins>ML_Model_Dev</ins>: A directory that was used for very early attempts at predicting supplier performance with machine learning models. Results were poor. No longer active. 

*The following direcories all contain code that is <ins>live</ins> in runnning the Streamlit App*
- <ins>Assets</ins>: Contains non code assets used in the app, such as images.
- <ins>data</ins>: Contains data used and relevant research into recommendation algorithms.
- <ins>pages</ins>: Contains code used to build the pages of the app. All pages besides the homepage can be found here.
- <ins>homepage.py</ins>: A python file that containing code used to build the homepage of the app. References all files in the pages directory.
- <ins>gui.py</ins>: A python file used to help format the app.
- <ins>MVEF.py</ins>: A python file containing functions used to generate the recommendations for the recommendations page.
