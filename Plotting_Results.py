import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv("results_table.csv")
print(df)

df['Model'] = df['Model'].astype('category')

#set seaborn plotting aesthetics
sns.set(style='white')

#create grouped bar chart
sns.barplot(x='Antibiotic', y='Accuracy', hue='Model', data=df,
            palette=['purple', 'steelblue'])

#add overall title
plt.title('Accuracy', fontsize=16)

#add axis titles
plt.xlabel('Antibiotic', fontsize=14)
plt.ylabel('Accuracy\n(%)', fontsize=14)

#rotate x-axis labels
plt.xticks(rotation=45)

plt.show()



#set seaborn plotting aesthetics
sns.set(style='white')

#create grouped bar chart
sns.barplot(x='Antibiotic', y='Sensitivity', hue='Model', data=df,
            palette=['purple', 'steelblue'])

#add overall title
plt.title('Sensitivity', fontsize=16)

#add axis titles
plt.xlabel('Antibiotic', fontsize=14)
plt.ylabel('Sensitivity\n(%)', fontsize=14)

#rotate x-axis labels
plt.xticks(rotation=45)

plt.show()



#set seaborn plotting aesthetics
sns.set(style='white')

#create grouped bar chart
sns.barplot(x='Antibiotic', y='Specificity', hue='Model', data=df,
            palette=['purple', 'steelblue'])

#add overall title
plt.title('Specificity', fontsize=16)

#add axis titles
plt.xlabel('Antibiotic', fontsize=14)
plt.ylabel('Specificity\n(%)', fontsize=14)

#rotate x-axis labels
plt.xticks(rotation=45)

plt.show()

