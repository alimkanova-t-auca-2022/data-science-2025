import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("group-project\love,arranged,kidnapped.xlsx")

years = df['Love_Marriage year']
love_amount = df['Love_Amount']
arranged_amount = df['Arranged_Amount']
kidnapped_amount = df['Kidnapped_Amount']

plt.figure(figsize=(12, 6))
plt.plot(years, love_amount, label='Love Marriage', marker='o', linestyle='-')
plt.plot(years, arranged_amount, label='Arranged Marriage', marker='s', linestyle='--')
plt.plot(years, kidnapped_amount, label='Kidnapped Marriage', marker='^', linestyle=':')

plt.xlabel('Marriage Year', fontsize=11, fontweight='bold')
plt.text(1960, 75, '   Number \nof Marriages', fontsize=11, fontweight='bold')
plt.title('Trends in Love, Arranged, and Kidnapped Marriages (1965–2010)', fontweight='bold')
plt.legend(loc='upper left') 
plt.grid(True)
plt.xticks(range(1965, 2011, 5), rotation=45)
plt.yticks(range(0, 80, 10))
plt.ylim(0, 80)
plt.xlim(1965, 2010)
plt.show()