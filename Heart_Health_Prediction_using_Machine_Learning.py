import pandas as pd
df=pd.read_csv("/content/heart.csv")
df.head()
df.isnull().sum()
import matplotlib.pyplot as plt
import seaborn as sns 

# Assuming df is your DataFrame containing the data
sns.countplot(x="sex", data=df, palette="Set2")

# Set labels and title
plt.xlabel("Sex (0: Female, 1: Male)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.title("Distribution of Sex in the Dataset", fontsize=14)

# Show the plot
plt.show()
x=df.drop("output",axis=1)
y=df["output"]
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x=scaler.fit_transform(x)
from sklearn.model_selection import train_test_split
a,b,c,d=train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.linear_model import LogisticRegression
obj=LogisticRegression()
obj.fit(a,c)
ycap=obj.predict(b)
from sklearn.metrics import accuracy_score
print(accuracy_score(d,ycap))


new=[[20,1,0,120,100,0,0,100,0,0.8,0,0,0]]
print(obj.predict(new))