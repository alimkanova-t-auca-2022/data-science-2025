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

data_frame['gender'] = data_frame['gender'].replace("female", "Female")
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
pyplot.figure(figsize=(18, 10))
category_counter.plot.bar(color=['#8FBF83'])
pyplot.title("Number of Oscar Winners by Category")
pyplot.xlabel('')
pyplot.ylabel('Number of Winners',loc='top',rotation=0)
xticks_categories = list(category_counter.index[:5]) + list(category_counter.index[-5:])
xticks_positions = list(range(5)) + list(range(len(category_counter) - 5, len(category_counter)))
pyplot.xticks([])
pyplot.subplots_adjust(bottom=0.2) 
pyplot.savefig('lab02\winners_by_category.png')

n = int(input("Enter n: "))
max_categories = len(data_frame['Category'].unique())
if n > max_categories: 
    print(f"Only {max_categories} categories are available.")
    exit()
    
top_n_cat = category_counter.head(n)
pyplot.figure(figsize=(18, 8))

top_n_cat.plot.barh(color='lightgreen')
pyplot.title(f"Top {n} Categories with Most Oscar Winners")

pyplot.xlabel('Number of Winners')

pyplot.ylabel('Category',loc='top', rotation=0, labelpad=2)
pyplot.gca().yaxis.set_label_coords(-0.005, 1.02)  
pyplot.tight_layout()
pyplot.savefig(f'lab02\\top_{n}_winners_by_category.png')
