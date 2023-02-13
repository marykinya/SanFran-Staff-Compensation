import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import zipfile

#Sidebar setup
st.sidebar.title("Quick Navigation")
app_mode = st.sidebar.selectbox( "Choose an action: ", ["Project Background",
                                                        "Data Analysis",
                                                        "Salary Prediction"
                                                        ])

#Import the dataset
zf = zipfile.ZipFile('Employee_Compensation.zip') 
data = pd.read_csv(zf.open('Employee_Compensation.csv'))

# data = pd.read_csv('Employee_Compensation.zip')
data.columns = [x.lower() for x in data.columns.str.replace(' ', '_')]
        
#Introduction
def show_about():
    global data 
    st.title("Compensation Prediction app")

    st.write('''
    #### Data Selection \n
    - The Data is sourced from [DataSF](https://data.sfgov.org/City-Management-and-Ethics/Employee-Compensation/88g8-5mnd) which is an open data platform that provides datasets shared by various institutions \n
        - The San Francisco Controller's Office maintains a database of the salary and benefits paid to City employees since fiscal year 2013
        - There's an API endpoint option but I opted to download the CSV instead 
    - The dataset is vast, running over a period of 9 years and the focus area is HR analytics
    #### Project Objective \n
    - This project is intended to predict the employee's overall compensation based on their unique variables \n
    - Steps followed:
        - Data Exploration
        - Data Cleaning
        - Feature engineering
        - Modelling
        - User test ; based on your unique selections, what is your compensation most likely to be ?
    ''')

    st.write("\n")

    st.write("**View data sample:**")
    st.write(data.head())
    
    st.write("\n")

    data_dictionary = st.checkbox("**View Data Glossary:**")
    if data_dictionary:
        st.write('''
        Only columns that are not self-explanatory are highlighted below: \n
        | Column | Description | 
        |---------|-------|
        |Year Type|`Fiscal` (July through June) or `Calendar` (January through December)|
        |Year|An accounting period of 12 months|
        |Organization Group|Group of Departments i.e. the `Public Protection` group includes departments such as `Police`, `Fire`, `Adult Probation` etc.|
        |Department|Departments are the primary organizational unit used by the City and County of San Francisco eg. `Police`|
        |Union|Unions represent employees in collective bargaining agreements. A job belongs to 1 union, although some jobs are unrepresented|
        |Job Family|Job Family combines similar Jobs into meaningful groups|
        |Employee ID|Each distinct number in the `Employee Identifier` column represents one employee|
        |Salaries|Normal salaries paid to permanent or temporary employees|
        |Overtime|Amounts paid to employees working in excess of 40 hours per week|
        |Other Salaries|Various irregular payments made to employees including premium pay, incentive pay, or other one-time payments|
        |Total Salary|The sum of all salaries paid to employees|
        |Retirement|City contributions to employee retirement plans|
        |Other Benefits|Mandatory benefits paid on behalf of employees, such as Social Security (FICA and Medicare) contributions, unemployment insurance premiums, and minor discretionary benefits not included in the above categories|
        |Total Benefits|The sum of all benefits paid to employees|
        |Total Compensation|The sum of all salaries and benefits paid to employees|
        ''')
    
    data['union'] = data['union'].replace({
                '':'Miscellaneous Unrepresented Employees',
                'Misc. Unrepresented Employees':'Miscellaneous Unrepresented Employees',
                'Mgt. Unrepresented Employees':'Management Unrepresented Employees',
                'Operating Engineers - Miscellaneous, Local 3':'Operating Engineers, Local 3',
                'Carpet, Linoleum & Soft Tile':'Carpet, Linoleum and Soft Tile Workers, Local 12',
                'Theatrical Stage Emp, Local 16':'Theatrical and Stage Employees, Local 16',
                'Prof & Tech Eng, Local 21':'International Federation of Professional & Technical Engineers, Local 21',
                'Prof & Tech Engineers - Miscellaneous, Local 21':'International Federation of Professional & Technical Engineers, Local 21',
                'Prof & Tech Engineers - Personnel, Local 21':'International Federation of Professional & Technical Engineers, Local 21',
                'Court-Local 21 Professional':'Prof & Tech Engineers - Court Employees, Local 21',
                'Prof & Tech Engineers - Court Employees, Local 21':'International Federation of Professional & Technical Engineers, Local 21',
                'Hod Carriers, LiUNA, Local 261':'Hod Carriers, Local 166',
                'Plumbers, Local 38':'Plumbers and Pipefitters, Local 38',
                'Roofers, Local 40':'Roofers and Waterproofers, Local 40',
                'Court Local 21 Staff Attorneys':'Prof & Tech Engineers - Court Employees, Local 21',
                'Prof & Tech Engineers - Court Attorneys, Local 21':'Prof & Tech Engineers - Court Employees, Local 21',
                'Auto Machinist, Local 1414':'Automotive Machinists, Local 1414',
                'Physician/Dentists 11-AA, UAPD':'Union of American Physicians and Dentists',
                'Physicians and Dentists - Spv Physician Specialist':'Union of American Physicians and Dentists',
                'Physician/Dentists 8-CC, UAPD':'Union of American Physicians and Dentists',
                'Physicians and Dentists - Miscellaneous':'Union of American Physicians and Dentists',
                'Court-Supr Court Interpreters':'Court Interpreters, Local 39521',
                'Court-Unrep Professional':'Court Unrepresented Professionals',
                'Court-Unrep Management':'Court Unrepresented Managers',
                'Court-Court Reporters':'Court Reporters',
                'Prof & Tech Engineers - Court Reporters, Local 21':'International Federation of Professional & Technical Engineers, Local 21',
                'TWU, Local 200':'Transport Workers Union, Local 200',
                'Transportation Workers, Local 200':'Transport Workers Union, Local 200',
                'SEIU - Health Workers, Local 1021':'SEIU, Local 1021',
                'TWU, Local 250-A, Misc':'Transport Workers Union, Local 250-A',
                'Transport Workers - Miscellaneous, Local 250-A':'Transport Workers Union, Local 250-A',
                'TWU, Local 250-A, AutoServ':'Transport Workers Union, Local 250-A',
                'Transport Workers - Auto Svc Workers, Local 250-A':'Transport Workers Union, Local 250-A',
                'TWU, Local 250-A, TransitOpr':'Transport Workers Union, Local 250-A',
                'Transport Workers - Transit Operators, Local 250-A':'Transport Workers Union, Local 250-A',
                'TWU, Local 250-A, TranFare':'Transport Workers Union, Local 250-A',
                'Transport Workers - Fare Inspectors, Local 250-A':'Transport Workers Union, Local 250-A',
                'Member, Board Of Sups':'Members of the Board of Supervisors',
                'Laborers Int, Local 261':'Laborers, Local 261',
                'Indv. Employment Contract-MTA':'Management Unrepresented Employees - MTA',
                "Municipal Attorneys Assoc":"Municipal Attorneys' Association",
                'Member, Board Or Commission':'Members of Boards and Commissions',
                'Commissioner No Benefits':'Members of Boards and Commissions - No Benefits',
                "Municipal Exec Assoc, Misc":"Municipal Executives' Association (MEA)",
                "Municipal Executive Association - Miscellaneous":"Municipal Executives' Association (MEA)",
                "Municipal Exec Assoc-Misc":"Municipal Executives' Association (MEA)",
                "Municipal Exec Assoc, Fire":"Municipal Executives' Association (MEA)",
                "Municipal Executive Association - Fire":"Municipal Executives' Association (MEA)",
                "Municipal Exec Assoc, Police":"Municipal Executives' Association (MEA)",
                "Municipal Executive Association - Police":"Municipal Executives' Association (MEA)",
                "Court-MEA":"Municipal Executives' Association (MEA)",
                "Municipal Executive Association - Court":"Municipal Executives' Association (MEA)",
                'SF Courts Commissioner Assoc':'Courts Commissioner Association',
                "SFDA Investigators Assn":"District Attorney Investigators' Association",
                "Deputy Sheriffs' Assoc (DSA)":"Deputy Sheriffs' Association",
                "Sheriffs' Mgrs and Supv (MSA)":"Sheriff's Managers and Supervisors Association",
                'SEIU - Human Services, Local 1021':'SEIU, Local 1021',
                'Cement Masons, Local 300 (580)':'Cement Masons, Local 300',
                "Probation Off Assoc (DPOA)":"Deputy Probation Officers' Association",
                'Glaziers, Metal, and Glass Workers, Local 718':'Glaziers, Local 718',
                'SEIU, Local 1021, Misc':'SEIU, Local 1021',
                'SEIU - Miscellaneous, Local 1021':'SEIU, Local 1021',
                'SEIU, Local 1021, RN':'SEIU, Local 1021',
                'SEIU - Staff and Per Diem Nurses, Local 1021':'SEIU, Local 1021',
                'Utd Pub EmpL790 SEIU-Crt Clrks':'SEIU, Local 1021',
                'SEIU - Court Employees, Local 1021':'SEIU, Local 1021',
                'SEIU, Local 1021, H-1':'SEIU, Local 1021',
                'SEIU - Firefighter Paramedics, Local 1021':'SEIU, Local 1021',
                'Firefighters,Local 798, Unit 1':'Firefighters, Local 798',
                'Firefighters - Miscellaneous, Local 798':'Firefighters, Local 798',
                'Firefighters Unit 1, Local 798':'Firefighters, Local 798',
                'Firefighters,Local 798, Unit 2':'Firefighters, Local 798',
                'Firefighters - Chiefs/Fire Boat Workers, Local 798':'Firefighters, Local 798',
                'Teamsters, Local 856, Multi':'Teamsters, Local 856',
                'Teamsters - Miscellaneous, Local 856':'Teamsters, Local 856',
                'Teamsters, Local 856, Spv RN':'Teamsters, Local 856',
                'Teamsters - Supervising Nurses, Local 856':'Teamsters, Local 856',
                'POA':"Police Officers' Association",
                "Building Inspects - 6332":"Building Inspectors' Association",
                "Building Inspectors' Association - Chiefs":"Building Inspectors' Association",
                "Building Inspects - 6331/33":"Building Inspectors' Association",
                "Building Inspectors' Association - Inspectors":"Building Inspectors' Association",
                'Court-Judge':'Court Unrepresented Bench Officers',
                'Sup Probation Ofcr, Op Eng 3':'Operating Engineers, Local 3 (OE3)',
                'Operating Engineers - Sup Probation Ofcrs, Local 3':'Operating Engineers, Local 3 (OE3)',
                'SFIPOA, Op Eng, Local 3':'Operating Engineers, Local 3 (OE3)',
                "Institutional Police Officers' Association":"Police Officers' Association",
                'Unrepresented Contract Rte FBP':'Executive Contract Employees'
                })
    
    data['department'] = data['department'].replace({
                            'AAM Asian Art Museum':'Asian Art Museum',
                            'ADM Gen Svcs Agency-City Admin':'General Services Agency - City Admin',
                            'ADP Adult Probation':'Adult Probation',
                            'AIR Airport Commission':'Airport Commission',
                            'ART Arts Commission':'Arts Commission',
                            'Assessor':'ASR Assessor / Recorder',
                            'BOA Board Of Appeals - PAB':'Board Of Appeals - Personnel Appeals Board',
                            'BOS Board Of Supervisors':'Board Of Supervisors',
                            'CAT City Attorney':'City Attorney',
                            'CFC Children & Families Commsn':'Children & Families Commission',
                            'CHF Children;Youth & Families':'Children - Youth & Families',
                            'Children Youth & Families':'Children - Youth & Families',
                            'CII Commty Invest & Infrstrctr':'Community Investments & Infrastructure',
                            'CON Controller':'Controller',
                            'CPC City Planning':'City Planning',
                            'CRT Superior Court':'Superior Court',
                            'CSC Civil Service Commission':'Civil Service Commission',
                            'CSS Child Support Services':'Child Support Services',
                            'DAT District Attorney':'District Attorney',
                            'DBI Building Inspection':'Building Inspection',
                            'DEM Emergency Management':'Emergency Management',
                            'Department Of Public Works':'Public Works',
                            'Department of Technology':'Technology',
                            'Dept of Emergency Management':'Emergency Management',
                            'Dept of Police Accountablility':'Police Accountability',
                            'Dept Status of Women':'Status of Women',
                            'DPH Public Health':'Public Health',
                            'DPW GSA - Public Works':'Public Works',
                            'DT GSA - Technology':'Technology',
                            'ECN Economic & Wrkfrce Dvlpmnt':'Economic Workforce Development',
                            'Emergency Communications Dept':'Emergency Communications ',
                            'ENV Environment':'Environment',
                            'ETH Ethics Commission':'Ethics Commission',
                            'FAM Fine Arts Museum':'Fine Arts Museum',
                            'FIR Fire Department':'Fire Department',
                            'GEN General City / Unallocated':'Unallocated',
                            'HHP Hetch Hetchy Water & Power':'Hetch Hetchy Water & Power',
                            'HRC Human Rights Commission':'Human Rights Commission',
                            'HRD Human Resources':'Human Resources',
                            'HSA Human Services Agency':'Human Services Agency',
                            'HSS Health Service System':'Health Service System',
                            'JUV Juvenile Probation':'Juvenile Probation',
                            'LIB Public Library':'Public Library',
                            'LLB Law Library':'Law Library',
                            'MTA Municipal Transprtn Agncy':'Municipal Transportation Agency',
                            'Municipal Transportation Agcy':'Municipal Transportation Agency',
                            'MYR Mayor':'Mayor',
                            'PDR Public Defender':'Public Defender',
                            'POL Police':'Police',
                            'PRT Port':'Port',
                            'PUB Public Utilities Bureaus':'Public Utilities Bureaus',
                            'REC Recreation & Park Commsn':'Recreation And Park Commission',
                            'REG Elections':'Elections',
                            'RET Retirement System':'Retirement System',
                            'RNT Rent Arbitration Board':'Rent Arbitration Board',
                            'SCI Academy Of Sciences':'Academy Of Sciences',
                            'Sheriff Accountability OIG':'Sheriff Accountability',
                            'SHF Sheriff':'Sheriff',
                            'TTX Treasurer/Tax Collector':'Treasurer/Tax Collector',
                            'WAR War Memorial':'War Memorial',
                            'WOM Status Of Women':'Status Of Women',
                            'WTR Water Enterprise':'Water Enterprise',
                            'WWE Wastewater Enterprise':'Wastewater Enterprise'
                            })
    
    data['job_family'] = data['job_family'].replace({
                        'Clerical, Secretarial & Steno':'Clerical, Secretarial & Stenographer',
                        'Untitled':'Unknown',
                        'Budget, Admn & Stats Analysis':'Budget, Admin & Statistical Analysis',
                        'Supervisory-Labor & Trade':'Supervisory - Labor & Trade',
                        'Construction Project Mgmt':'Construction Project Management',
                        'Pub Relations & Spec Assts':'Public Relations & Special Assets',
                        'Administrative-Labor & Trades':'Administrative - Labor & Trades',
                        'Administrative & Mgmt (Unrep)':'Administrative & Management (Unrep)',
                        'Administrative-DPW/PUC':'Administrative - DPW/PUC',
                        'Computer Operatns & Repro Svcs':'Computer Operations & Repro Services',
                        'Unassigned':'Unknown'
                        })
    
    data['job'] = data['job'].astype(str).apply(lambda x: x.upper())

    data = data[data.year_type == 'Fiscal']

#Data analysis
def data_analysis():
    global data 
    st.title("Data Cleaning & Exploration")
    st.write('''
    #### Data Understanding \n
    Let's explore the dataset to try gain some insights
    ''')

    ## year_type exploration -- fiscal vs calendar 
    st.write('''
    - **What is the proportion of employees paid by fiscal vs. calendar year?**
        - Would it make sense to limit the dataset to 1 pay period to ensure there are no overlaps ?
    ''')

    year_df = data[['year', 'year_type', 'employee_identifier']]
    year_df = year_df.groupby(['year','year_type'])['employee_identifier'].agg('count').reset_index()
    year_df.columns = ['year', 'year_type', 'employee_count']

    fig = px.line(year_df, x='year', y='employee_count', color='year_type', markers=True, height = 400)
    fig.update_layout(xaxis=dict(tickmode='linear',tick0=1,dtick=1),xaxis_title="Year", yaxis_title="Number of Employees")
    st.plotly_chart(fig, use_container_width=True)

    st.write('''
    `A calendar year helps estimate annual salary on a calendar basis (Jan to Dec).` \n 
    `Fiscal year is mostly for financial managers to ensure that all revenue and overhead costs are within the same tax return period, for simplified reconciliation.` \n
    `In our case, we'll filter our dataset by Fiscal year since the dataset includes fiscal data in 2022, regardless either option should be viable.`
    ''')
    
    st.write('''
    #### Text Columns Clean-up \n
    - The following columns have redundant entries written out in different variations:
        - `Union`
        - `Department` 
        - `Job Family`
        - `Job`
    - Here's a quick summary of the clean up: \n
    |Column|Entries pre-clean up|Entries post-clean up|
    |---------|---------|---------|
    |Union|128|54|
    |Department|108|68|
    |Job Family|59|58|
    |Job|1359|1312|
    ''')

    st.write("\n")
    #What are the unique organization types?
    st.write("**What are the unique organization types?**")

    og = data[['organization_group', 'employee_identifier']]
    og = og.groupby(['organization_group'])['employee_identifier'].agg('count').reset_index()
    og.columns = ['organization_group', 'employee_count']

    fig = px.bar(og.sort_values(by='employee_count',ascending = True), x='organization_group', y='employee_count', height = 600, text='employee_count')
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside').update_layout(xaxis_title="Organization types", yaxis_title="Number of Employees")
    st.plotly_chart(fig, use_container_width=True)

    st.write("`There are 7 unique organizational types`")

    st.write("**How do the organization types tie into various departments?**")
    og = data[['organization_group', 'department', 'employee_identifier']]
    og = og.groupby(['organization_group', 'department'])['employee_identifier'].agg('count').reset_index()
    og.columns = ['organization_group', 'department', 'employee_count']

    fig = px.bar(og, y='organization_group', x='employee_count', color='department', height = 600, width = 1000)
    fig.update_layout(showlegend=False,yaxis_title="Organization types", xaxis_title="Number of Employees")
    st.plotly_chart(fig, use_container_width=True)


    st.write("**What are the various Job Families categorized by union?**")
    og = data[['job_family', 'union', 'employee_identifier']]
    og = og.groupby(['job_family','union'])['employee_identifier'].agg('count').reset_index()
    og.columns = ['job_family', 'union','employee_count']

    fig = px.bar(og, y='job_family', x='employee_count', color='union',height = 1000)
    fig.update_layout(showlegend=False,yaxis_title="Jobs", xaxis_title="Number of Employees")
    st.plotly_chart(fig, use_container_width=True)

    st.write('''
        `Quick Summary:` \n
        - `We have 420302 rows and 22 columns in our dataset.` \n
        - `Both categorical and numerical variables.` \n
        - `Missing values, null values dropped` \n
        ''')
    
    st.write('''
    #### Data Cleaning on Numerical Columns
    - First step, drop redundant numerical columns: \n
        - `organization_group_code`
        - `job_family_code`
        - `job_code`
        - `department_code`
        - `union_code`
        - `total_salary`
        - `total_benefits`
        - `year_type`
    - Second, check numerical columns `max()` and `min()` to ensure values are non-negative or equal to zero \n
        - Negative Salaries: `7027`
        - Negative Overtime: `219398`
        - Negative Other Salaries: `135746`
        - Negative Retirement: `60369`
        - Negative Health and Dental: `83604`
        - Negative Other Benefits: `1174`
        - Negative Total Compensation: `244`
    - All values replaced by the mean value of the column. 
    
    ''')

    data = data.dropna()
    data.drop(['employee_identifier','organization_group_code','job_code','department_code','job_family_code','union_code','total_salary','total_benefits','year_type'], axis=1, inplace=True)

    clean_up_df = data[['salaries','overtime','other_salaries','retirement','health_and_dental','other_benefits','total_compensation']]
    for col in clean_up_df:
        data.loc[data[col] <= 0, col] = 0
        data[col]=data[col].replace(0,data[col].mean())
    
    st.write("\n")
    st.write("#### Correlation Matrix")
    st.write(data.corr()) 

    st.write('''
    The matrix shows strong +ve correlations between:
    - `salaries` and `total_compensation`
    - `overtime` and `other_salaries`
    - `retirement` and `salaries`
    ''')


def salary_prediction():
    global data 
    st.title("Let's predict your most likely compensation!") 
    
    data = data[data.year_type == 'Fiscal']
    data = data.dropna()
    data.drop(['employee_identifier','year_type'], axis=1, inplace=True)

    clean_up_df = data[['salaries','overtime','other_salaries','retirement','health_and_dental','other_benefits','total_compensation']]
    for col in clean_up_df:
        data.loc[data[col] <= 0, col] = 0
        data[col]=data[col].replace(0,data[col].mean())
        
        cols = ['organization_group_code','department_code','union_code','job_family_code','job_code']
        for col in cols:
            data[cols] = data[cols].astype('category')
            data[cols] = data[cols].apply(lambda x: x.cat.codes)

        ref_data = data.copy()

    def linear_regression_model():
        data.drop(['organization_group','department','union','job_family','job'], axis=1, inplace=True)

        X = data.drop('total_compensation',axis=1)
        y = data['total_compensation']
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2, random_state=45)

        #import your model
        model=LinearRegression()
        model.fit(X_train, y_train)
        errors = np.sqrt(mean_squared_error(y_test,model.predict(X_test)))
        predictions = model.predict(X_test)[0]

        return int(predictions)

    dept_options = data.department.unique()
    dept_select = st.multiselect('**Select 1 or more Departments**',dept_options)
    # st.write('You selected:', dept_select)
        
    if dept_select != None:
        df = data.query('department == @dept_select')
        job_options = df.job.unique()

        job_select = st.selectbox('**Select 1 Job Option**',job_options)
        # st.write('You selected:', job_select)

        data = data[(data.department.isin(dept_select)) & (data.job == job_select)]
    else:
        st.write('Kindly select an option')

    if st.button('Calculate the compensation'):
        output = linear_regression_model() 
        m_output = output / 12

        st.write("\n")

        st.write("##### As a",job_select,"working in either of the following departments : \n")
        st.write(dept_select,"##### You are most likely to earn an annual compensation of: $",output)
        st.write("##### This is roughly: $",int(m_output),"per month")

if app_mode == "Project Background":  
    show_about()
elif app_mode == "Data Analysis":   
    data_analysis()
elif app_mode == "Salary Prediction":  
    salary_prediction()  