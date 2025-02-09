import pandas as pd
import matplotlib.pyplot as pyplot

try:
    data_frame = pd.read_excel("lab02\oscars.xlsx")
except FileNotFoundError:
    print("File is not found.")
    exit()
except ValueError:
    print("The file format is incorrect.")
    exit()
except Exception as e:
    print(f"Error: {e}")
    exit()

if 'gender' not in data_frame.columns:
    print("Missing 'gender' column.")
    exit()

gender_counter = data_frame['gender'].value_counts()
pyplot.figure(figsize=(7,7))
gender_counter.plot.pie(autopct='%1.1f%%', colors=['#75a154','#FFA33F'], startangle=90)
pyplot.title("Gender Distribution of Oscar Winners")
pyplot.ylabel('')
pyplot.savefig('lab02\gender_distribution.png')

if 'Category' not in data_frame.columns:
    print("Missing 'Category' column.")
    exit()
category_counter = data_frame['Category'].value_counts()
pyplot.figure(figsize=(10, 6))
category_counter.plot.bar(color=['#8FBF83'])
pyplot.title("Number of Oscar Winners by Category")
pyplot.xlabel('')
pyplot.ylabel('Number of Winners')
pyplot.xticks(rotation=45, ha = 'right')
pyplot.savefig('lab02\winners_by_category.png')

n = int(input("Enter n: "))
max_categories = len(data_frame['Category'].unique())
if n > max_categories: 
    print(f"Only {max_categories} categories are available.")
    exit()
    
top_n_cat = category_counter.head(n)
pyplot.figure(figsize=(10, 6))
top_n_cat.plot.barh(color='lightgreen')
pyplot.title(f"Top {n} Categories with Most Oscar Winners")
pyplot.xlabel('Category')
pyplot.ylabel('Number of Winners')
pyplot.savefig(f'lab02\\top_{n}_winners_by_category.png')
