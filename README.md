# 4416-lab-4

1- Find association rules in the dataset using Apriori algorithm with minimum support 25% and minimum confidence 70%.



For question 2, use the Belgium retail market dataset that is available at Frequent Itemset Mining Dataset Repository. More details can be found here.

2- Download apriori.exe from web site http://www.borgelt.net/apriori.html 

Run the algorithm in command prompt according to given instructions http://www.borgelt.net/doc/apriori/apriori.html

Find top-5 1-itemset  2-itemsets  3-itemsets  and 4-itemsets. 

•	Example Command
apriori.exe retail.dat -s2m3n3 -q-2 –

3- Execute the following Python code. Paste and examine the results.

#%pip install mlxtend
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
itemsets = apriori(df, min_support=0.6, use_colnames=True)

print(itemsets)

association_rules(itemsets, metric='confidence', min_threshold=0.5, support_only=False)


