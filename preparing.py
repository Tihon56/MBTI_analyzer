import pandas as pd
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data.csv')

#размерность 128061*9

#Мужчина = 0, Женцина = 1
data['Gender'] = data['Gender'].map({'Male':0,'Female':1})

#Отсутствие интересов=0 Спорт=1 Другие=2 Технологии=3 Исскуство=4
data['Interest'] = data['Interest'].map({
    'Unknown':0,
    'Sports':1,
    'Others':2,
    'Technology':3, 
    'Arts':4
})

data['Personality'] = data['Personality'].map({
    'ENFP':1,
    'ESFP':2,
    'INTP':3,
    'INFP':4,
    'ENFJ':5,
    'ENTP':6,
    'ESTP':7,
    'ISTP':8,
    'INTJ':9,
    'INFJ':10,
    'ISFP':11,
    'ENTJ':12,
    'ESFJ':13,
    'ISFJ':14,
    'ISTJ':15,
    'ESTJ':16,
    })

scaler = StandardScaler()

#data[['Age','Gender','Education','Introversion Score','Sensing Score','Thinking Score','Judging Score','Interest','Personality']] = scaler.fit_transform(data[['Age','Gender','Education','Introversion Score','Sensing Score','Thinking Score','Judging Score','Interest','Personality']])

data.to_csv("prepared_data", index=True)