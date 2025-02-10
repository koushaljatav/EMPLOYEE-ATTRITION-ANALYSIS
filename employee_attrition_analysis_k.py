#!/usr/bin/env python
# coding: utf-8

# # Load Data

# In[1]:


import pandas as pd
dataset=pd.read_csv("employee-attrition-dataset.csv")
dataset.head()
dataset.tail()


# # import liberaries

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, ttest_ind
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm


# In[3]:


# Load dataset
dataset=pd.read_csv("employee-attrition-dataset.csv")
dataset


# In[4]:


dataset.size


# In[5]:


dataset.shape


# In[6]:


dataset.index


# In[7]:


dataset.columns


# In[8]:


dataset.columns.to_list()


# In[9]:


dataset.head()


# In[10]:


dataset.tail()


# In[11]:


dataset.info(verbose=False,show_counts=True)


# In[12]:


dataset.info()


# In[13]:


dataset.describe()


# In[14]:


dataset.describe().columns


# In[15]:


list(set(dataset.columns)-set(dataset.describe().columns))


# In[16]:


dataset["Gender"]


# In[17]:


dataset.Gender


# In[18]:


dataset[["Gender","MonthlyIncome"]]


# In[19]:


dataset.nunique()


# In[20]:


dataset["Gender"].unique()


# In[21]:


dataset.isnull()


# In[22]:


pd.isna(dataset)


# In[23]:


dataset.isnull().sum()


# In[24]:


dataset.isnull().mean() 


# In[25]:


dataset.isnull().mean()*100


# In[26]:


def missing_value(dataframe):
    return round(dataframe.isnull().mean()*100,2).sort_values(ascending=False)


# In[27]:


missing_value(dataset)


# In[28]:


null=missing_value(dataset)[missing_value(dataset)>50]


# In[29]:


null


# In[30]:


null.index


# In[31]:


dataset.drop(columns=null.index,inplace=False)


# In[32]:


dataset[["ï»¿Age","Attrition","BusinessTravel","DailyRate","Department","DistanceFromHome","Education","EducationField","EmployeeCount","EmployeeNumber","RelationshipSatisfaction","StandardHours","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany",
         "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]].isnull().sum()


# In[33]:


sns.boxplot(dataset["TotalWorkingYears"],color="magenta")


# In[34]:


plt.hist(dataset["TotalWorkingYears"],color="cyan")


# In[35]:


dataset["TotalWorkingYears"].describe()


# In[36]:


q1=np.quantile(dataset["TotalWorkingYears"].dropna(),0.25)
q1


# In[37]:


q2=np.quantile(dataset["TotalWorkingYears"].dropna(),0.50)
q2


# In[38]:


q3=np.quantile(dataset["TotalWorkingYears"].dropna(),0.75)
q3


# In[39]:


IQR=q3-q1
IQR


# In[40]:


upper_fence=q3+1.5*(IQR)
lower_fence=1-1.5*(IQR)


# In[41]:


l=[]
for i in dataset["TotalWorkingYears"]:
    if i>upper_fence  or i<lower_fence:
        l.append(i)
print(l)


# In[42]:


missing_values = dataset.isnull().sum()
print("\nMissing Values Per Column:")
print(missing_values)


# In[43]:


dataset['MonthlyIncome'].fillna(dataset['MonthlyIncome'].median(), inplace=True)
dataset['NumCompaniesWorked'].fillna(dataset['NumCompaniesWorked'].mode(), inplace=True)


# In[44]:


dataset['MonthlyIncome']


# In[45]:


plt.hist(dataset["MonthlyIncome"],color="hotpink")


# In[46]:


dataset['NumCompaniesWorked']


# In[47]:


sns.boxplot(dataset["NumCompaniesWorked"],color="cyan")


# In[48]:


dataset.dropna(subset=['Department'], inplace=True)
dataset['Department']


# In[49]:


plt.hist(dataset["Department"],color="pink")


# # Drop columns that are irrelevant

# In[50]:


# List of columns to drop
columns_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeID']

# Drop columns if they exist in the dataset
dataset.drop(columns=[col for col in columns_to_drop if col in dataset.columns], axis=1, inplace=True)

# Display remaining columns
print("\nColumns after dropping irrelevant ones:")
print(dataset.columns)


# In[51]:


# Map categorical variables to numeric
# dataset['Attrition'] = dataset['Attrition'].map({'Yes': 1, 'No': 0})
# dataset['OverTime'] = dataset['OverTime'].map({'Yes': 1, 'No': 0})
# dataset['Department'] = dataset['Department'].map({ 'Sales': 1, 'Research & Development': 2, 'Human Resources': 3})


# In[52]:


print(dataset.head())


# In[53]:


dataset['AgeGroup'] = pd.cut(dataset['ï»¿Age'], bins=[18, 30, 40, 50, 60], labels=['18-30', '30-40', '40-50', '50-60'])
dataset['YearsAtCompanyBins'] = pd.cut(dataset['YearsAtCompany'], bins=[0, 1, 5, 10, 20, 40], 
                                     labels=['<1', '1-5', '5-10', '10-20', '>20'])
print("Data cleaning and preprocessing complete.")


# In[54]:


dataset['AgeGroup']


# In[55]:


dataset['YearsAtCompanyBins']


# # Re-importing all liberaries for perform analysis
# 

# In[56]:


import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# # introduction of project 

# ATTRITION --> Attrition of employees from organization for any reason or in simple words we can say that emplpyees who wants to 
# leave the company
# 
# ATTRITION "YES"-->Employee wants to leave the company.
# 
# ATTRITON "NO"-->Eemployee don't want to leave the company

# # domain analysis

# 1.AGE
# 
# Age of an employee
# 
# 2.BUSSINESS TRAVEL
# 
# That employee is travelling or not for bussines purpose
# 
# 3.DEPARTMENT
# 
# In which department of company employee is working
# 
# 4.DISTANCE FROM HOME
# 
# that how much distance is there of an employee from his/her house to office
# 
# 5.HOURLY/DAILY/MONTHLY RATING
# 
# Rating given to employee on daily,hourly and montly basis
# 
# 6.EDUCATION FIELD
# 
# from which education background employee belongs to
# 
# 7.JOB AND ENVIRONMENT SATISFACTION
# 
# It is that employee is satisfiedd with their job as well as (given in 1-5 rating)
# 
# 8.JOB INVOLVEMENT
# 
# How much employee is involvment in his work(given in rating 1-5)
# 
# 9.JOB LEVEL
# 
# Level of job,higher rating==higher job level
# 
# 10.PERFORMANCE RATING
# 
# How employee is performing according to his job(given rating 1-5)
# 
# 11.MONTHLY INCOME
# 
# Monthly salary of an employee in doller
# 
# 12.PERCEMTAGE SALARY HIKE
# 
# How many percent salary in incresing anually
# 
# 

# In[57]:


dataset.describe(include='O')


# In[58]:


dataset.describe()


# # Analysis on categorical data

# TARGET COLUMN==ATTRITION

# In[59]:


sns.countplot(x=dataset.Attrition)
plt.show()


# -->Data  of attrition says that it has more number of"No" values and less number of "Yes"
# 
# -->It can be seen that there is big diffrence in counts of the values so we can says that it is IMBALANCE DATA

# # 1  IMPACT OF BUSSINES TRAVEL ON ATTRITION

# In[60]:


sns.countplot(hue=dataset.Attrition,x=dataset.BusinessTravel)
plt.show()


# -->The graph tell us that company has more count or more no of employees travel rearly
# 
# -->There are more employees travel rearly and not satisfied with their jobs
# 
# -->Non-traveller have least count as well as least attrition

# # 2 IMPACT OF DEPARTMENT ON ATTRITION 

# In[61]:


plt.figure(figsize=(10,8))
sns.countplot(hue=dataset.Attrition,x=dataset.Department)
plt.show()


# -->There are 3 no. of department are there = 1.Sales,2.Research & Development,3.Human Resources
# 
# -->"Research & Development" department have more number of department of Attrition(150 employees)as compare to other two department
# 
# -->"HR Department" have least Attrition with 5 to 10 employees

# # 3 IMPACT OF EDUCATION ON ATTRITION

# In[62]:


plt.figure(figsize=(14,10))
sns.countplot(hue=dataset.Attrition,x=dataset.EducationField)
plt.show()


# -->First and foremost things is that Employees are from "life science" and "Medical" background are more as comapre to other education field
# 
# --> Nearly 100 number of employee are there who are from Lifes Sciences education background will leave the company and follow the Medical education employees
# 
# -->As we conclude from analysis of Department and Attrition .here also HR education background employees have least Attrition 

# # 4. IMPACT GENDER ON ATTRITION  

# In[63]:


plt.figure(figsize=(8,6))
sns.countplot(x=dataset.Attrition,hue=dataset.Gender)
plt.show()


# --> Male employees are more as compared to Female employees
# 
# --> Males are more likely to quit rather than Females

# # 5 OVERTIME AND ATTRITION 

# In[64]:


plt.figure(figsize=(8,6))
sns.countplot(hue=dataset.Attrition,x=dataset.OverTime)
plt.show()


# --> As for "Attrition yes" there is the minor diffrents between Employees who are doing OverTime and who are not doing OverTime
# 
# --> So we can say overtime feature is not mutch effecting Attrition 
# 
# --> But we can conclude that most of employees are not doing overtime

# # 6 IMPACT OF JOBOLE ON ATTRITION 

# In[65]:


plt.figure(figsize=(22,10),facecolor='white')
sns.countplot(x=dataset.JobRole,hue=dataset.Attrition)
plt.show()


# --> There is less no of Research director who leaves the company
# 
# -->Laboratory technician , sales executive and reasearch scientist are the top 3job roles in which employees have their "Attrition yes"
# 
# -->It can also seen that more number of employees in sales executive job role

# # ANALYSIS ON CONTINUOS DATA WITH RESPECT TO TARGET COLUMN

# In[66]:


numerical_col=[]
for column in dataset.columns:
    if dataset[column].dtype=="int64" and len(dataset[column].unique())>=10:
        numerical_col.append(column)


# In[67]:


numerical_col


# # graphical repesentation of continuos data

# In[68]:


dataset2=dataset[
    ['ï»¿Age',
 'DailyRate',
 'DistanceFromHome',
 'EmployeeNumber',
 'HourlyRate',
 'MonthlyIncome',
 'MonthlyRate',
 'NumCompaniesWorked',
 'PercentSalaryHike',
 'TotalWorkingYears',
 'YearsAtCompany',
 'YearsInCurrentRole',
 'YearsSinceLastPromotion',
 'YearsWithCurrManager']]


# # Another method of visualization 

# In[69]:


plt.figure(figsize=(40,35),facecolor='white')
plotnumber = 1

for column in dataset2:
    if plotnumber<=16:
        ax=plt.subplot(4,4,plotnumber)
        sns.histplot(x=dataset2[column].dropna(axis=0),hue=dataset.Attrition)
        
        plt.xlabel(column,fontsize=40)
        plt.ylabel('Attrition',fontsize=40)
    plotnumber+=1
plt.tight_layout()


# In[70]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_in_pairs(dataset2, hue_column):
    columns = dataset2.columns
    num_columns = len(columns)
    pair_count = 2 

    for i in range(0, num_columns, pair_count):
        plt.figure(figsize=(20, 10), facecolor='white') 

        for j in range(pair_count):
            if i + j < num_columns:  
                ax = plt.subplot(1, pair_count, j + 1) 
                
                sns.histplot(
                    x=dataset2[columns[i + j]].dropna(), 
                    hue=hue_column, 
                    kde=False, 
                    ax=ax
                )
                
                plt.xlabel(columns[i + j], fontsize=20)
                plt.ylabel('Count', fontsize=20)
                ax.set_title(f'Distribution of {columns[i + j]}', fontsize=25)
                ax.tick_params(axis='both', which='major', labelsize=15)
        
        plt.tight_layout()
        plt.show()

plot_in_pairs(dataset2, hue_column=dataset.Attrition)


# In[ ]:





# # 1 IMPACT OF AGE ON ATTRITION

# In[71]:


plt.figure(figsize=(8,6),facecolor='white')
sns.histplot(x=dataset['ï»¿Age'],hue=dataset.Attrition)
plt.show()


# -->Employees in age 25 to 35 are more likely to  leave their job
# 
# -->After the age 40. the distribution tell us that "Higher the age lesser will we Attrition"

# In[72]:


plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(hue=dataset['ï»¿Age'],x=dataset.Attrition)
plt.show()


# # 2 DISTANCE FROM HOME AND ATTRITION 

# In[73]:


plt.figure(figsize=(8,6),facecolor='white')
sns.histplot(x=dataset.DistanceFromHome,hue=dataset.Attrition)
plt.show()


# In[74]:


plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(hue=dataset.DistanceFromHome,x=dataset.Attrition)
plt.show()


# --> Employees who has distance range "0-100" km are more likely to leave their job.
# 
# --> we can also conclude that lesser the distance more number of employees working.

# # 3 how monthly income give trends with respect to attrition 

# In[75]:


plt.figure(figsize=(8,6),facecolor='white')
sns.histplot(x=dataset.MonthlyIncome,hue=dataset.Attrition)
plt.show()


# --> Higher the monthly income give rise to less Attrition (means Attrition "no")
# 
# --> Employees who have their income 2500 aprox are more likely to quit their job because 2500 is the least range of income

# # 4.IMPACT OF NO. OF COMPANIES WORKED

# In[76]:


plt.figure(figsize=(8,6),facecolor='white')
sns.histplot(x=dataset.NumCompaniesWorked,hue=dataset.Attrition)
plt.show()


# In[77]:


plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(hue=dataset.NumCompaniesWorked,x=dataset.Attrition)
plt.show()


# --> Olny the employees(no. of employees=100) who work with one company before have more (Attrition yes) rest have similar data

# # 5 HOW SALARY HIKE IMPACT  THE ATTRITION 

# In[78]:


plt.figure(figsize=(8,6),facecolor='white')
sns.histplot(x=dataset.PercentSalaryHike,hue=dataset.Attrition)
plt.show()


# --> Higher the salary percentage hike lesser the Attrition ("no")

# # 6 YEAR AT  THE COMPANY

# In[79]:


plt.figure(figsize=(12,6),facecolor='white')
sns.histplot(x=dataset.	YearsAtCompany,hue=dataset.Attrition)
plt.show()


# --> Fresher have higher Attrition "yes" that is of 75 no. of workers are more than half of  feshers
# 
# --> Apart from employees who ranges fron 1 to 10 years working on company are less likely to quit their job.

# # ANALYSIS OF DISCRETE  DATA WITH RESPWCT TO TARGET COLUMN

# In[80]:


discrete_col=[]
for column in dataset.columns:
    if dataset[column].dtype=="int64" and len(dataset[column].unique())>=10:
        discrete_col.append(column)


# In[81]:


discrete_col


# In[82]:


dataset.describe()


# In[83]:


dataset3=dataset[[
    'Education',
    'EmployeeNumber',
    'EnvironmentSatisfaction',
    'JobInvolvement',
    'JobLevel',
    'JobSatisfaction',
     'NumCompaniesWorked',
    'PerformanceRating',
    'RelationshipSatisfaction',
    'StockOptionLevel',
    'TrainingTimesLastYear',
    'WorkLifeBalance'
]]


# In[84]:


dataset3


# In[85]:


plt.figure(figsize=(40,35),facecolor='white')
plotnumber = 1

for column in dataset3:
    if plotnumber<=16:
        ax=plt.subplot(4,4,plotnumber)
        sns.histplot(x=dataset3[column].dropna(axis=0),hue=dataset.Attrition)
        
        plt.xlabel(column,fontsize=40)
        plt.ylabel('Attrition',fontsize=40)
    plotnumber+=1
plt.tight_layout()


# # GRAPHICAL REPSENTATION

# In[86]:


import matplotlib.pyplot as plt
import seaborn as sns

def plot_in_pairs(dataset3, hue_column):
    columns = dataset3.columns
    num_columns = len(columns)
    pair_count = 2 

    for i in range(0, num_columns, pair_count):
        plt.figure(figsize=(20, 10), facecolor='white') 

        for j in range(pair_count):
            if i + j < num_columns:  
                ax = plt.subplot(1, pair_count, j + 1) 
                
                sns.histplot(
                    x=dataset3[columns[i + j]].dropna(), 
                    hue=hue_column, 
                    kde=False, 
                    ax=ax
                )
                
                plt.xlabel(columns[i + j], fontsize=20)
                plt.ylabel('Count', fontsize=20)
                ax.set_title(f'Distribution of {columns[i + j]}', fontsize=25)
                ax.tick_params(axis='both', which='major', labelsize=15)
        
        plt.tight_layout()
        plt.show()

plot_in_pairs(dataset3, hue_column=dataset.Attrition)


# In[ ]:





# # 1 IMPACT OF ENVIRNMENT AND JOB SATISFACTION ON ATTRITION 

# In[87]:


#plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(x=dataset.EnvironmentSatisfaction,hue=dataset.Attrition)
plt.show()


# --> Increase in rate of environment as well as job satisfaction give rise to increase in Attrtion "no"(means not willing to quit)

# # 2 HOW JOB LEVEL IMPACTING ON ATTRITION 

# In[88]:


#plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(x=dataset.JobLevel,hue=dataset.Attrition)
plt.show()


# --> Increase in job level .Decrease in chances of leaving the company for Employees

# # 3 JOBINVOLVEMENT IMPACT ON ATTRITION 

# In[89]:


#plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(x=dataset.JobInvolvement,hue=dataset.Attrition)
plt.show()


# --> The employees who involved in job more than sufficient are more likely to quit or we can say that they have mmore pressure of work
# 
# --> Somehow. there are some emp. who not involvd fully in there job but still they are likely to quit

# #  4 IMPACT OF STOCK OPTION LEVEL ON ATTRITION 

# In[90]:


#plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(x=dataset.StockOptionLevel,hue=dataset.Attrition)
plt.show()


# -->For the employees who're not having stock option are likely to quit

# # 5 PERFORMANCE RATING AND ATTRITION 

# In[91]:


#plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(x=dataset.PerformanceRating,hue=dataset.Attrition)
plt.show()


# --> On an average,most of the employees are moderately performed(because performance rating lies in 3-4).
# 
# --> However employees having less performance rating are more likely to quit we can say that company wants to fire that employees

# # 6 WORK LIFE BALANCE IMPACT ON ATTRITION 

# In[92]:


#plt.figure(figsize=(8,6),facecolor='white')
sns.countplot(x=dataset.WorkLifeBalance,hue=dataset.Attrition)
plt.show()


# --> More the employees life is balance ,lesser the Attrition.

# In[ ]:




