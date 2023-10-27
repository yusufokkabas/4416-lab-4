
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#first task
dataset = [
    ['Y', 'P', 'G'],
    ['C'],
    ['C', 'Y', 'K', 'R', 'G'],
    ['K', 'G', 'R'],
    ['C', 'A', 'L'],
    ['Y'],
    ['Y', 'R', 'G', 'K'],
    ['C', 'P']
]
df = pd.DataFrame(0, columns=set(item for transaction in dataset for item in transaction), index=range(len(dataset)))
for i, transaction in enumerate(dataset):
    df.loc[i, transaction] = 1

frequent_itemsets = apriori(df, min_support=0.25, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)


print("Frequent Itemsets:")
print(frequent_itemsets)

print("Association Rules:")
print(rules)
#this is the output of the first task

# Frequent Itemsets:
#     support      itemsets
# 0     0.500           (G)
# 1     0.375           (K)
# 2     0.500           (C)
# 3     0.250           (P)
# 4     0.375           (R)
# 5     0.500           (Y)
# 6     0.375        (G, K)
# 7     0.375        (G, R)
# 8     0.375        (G, Y)
# 9     0.375        (K, R)
# 10    0.250        (K, Y)
# 11    0.250        (R, Y)
# 12    0.375     (G, K, R)
# 13    0.250     (G, K, Y)
# 14    0.250     (G, R, Y)
# 15    0.250     (K, R, Y)
# 16    0.250  (G, K, R, Y)
# Association Rules:
#    antecedents consequents  antecedent support  consequent support  support  confidence      lift  leverage  conviction  zhangs_metric
# 0          (G)         (K)               0.500               0.375    0.375        0.75  2.000000  0.187500         2.5       1.000000
# 1          (K)         (G)               0.375               0.500    0.375        1.00  2.000000  0.187500         inf       0.800000
# 2          (G)         (R)               0.500               0.375    0.375        0.75  2.000000  0.187500         2.5       1.000000
# 3          (R)         (G)               0.375               0.500    0.375        1.00  2.000000  0.187500         inf       0.800000
# 4          (G)         (Y)               0.500               0.500    0.375        0.75  1.500000  0.125000         2.0       0.666667
# 5          (Y)         (G)               0.500               0.500    0.375        0.75  1.500000  0.125000         2.0       0.666667
# 6          (K)         (R)               0.375               0.375    0.375        1.00  2.666667  0.234375         inf       1.000000
# 7          (R)         (K)               0.375               0.375    0.375        1.00  2.666667  0.234375         inf       1.000000
# 8       (G, K)         (R)               0.375               0.375    0.375        1.00  2.666667  0.234375         inf       1.000000
# 9       (G, R)         (K)               0.375               0.375    0.375        1.00  2.666667  0.234375         inf       1.000000
# 10      (K, R)         (G)               0.375               0.500    0.375        1.00  2.000000  0.187500         inf       0.800000
# 11         (G)      (K, R)               0.500               0.375    0.375        0.75  2.000000  0.187500         2.5       1.000000
# 12         (K)      (G, R)               0.375               0.375    0.375        1.00  2.666667  0.234375         inf       1.000000
# 13         (R)      (G, K)               0.375               0.375    0.375        1.00  2.666667  0.234375         inf       1.000000
# 14      (K, Y)         (G)               0.250               0.500    0.250        1.00  2.000000  0.125000         inf       0.666667
# 15      (R, Y)         (G)               0.250               0.500    0.250        1.00  2.000000  0.125000         inf       0.666667
# 16      (K, Y)         (R)               0.250               0.375    0.250        1.00  2.666667  0.156250         inf       0.833333
# 17      (R, Y)         (K)               0.250               0.375    0.250        1.00  2.666667  0.156250         inf       0.833333
# 18   (G, K, Y)         (R)               0.250               0.375    0.250        1.00  2.666667  0.156250         inf       0.833333
# 19   (G, R, Y)         (K)               0.250               0.375    0.250        1.00  2.666667  0.156250         inf       0.833333
# 20   (K, R, Y)         (G)               0.250               0.500    0.250        1.00  2.000000  0.125000         inf       0.666667
# 21      (K, Y)      (G, R)               0.250               0.375    0.250        1.00  2.666667  0.156250         inf       0.833333
# 22      (R, Y)      (G, K)               0.250               0.375    0.250        1.00  2.666667  0.156250         inf       0.833333

#I can not perform second task because I use macOS and you given me the file for Windows. I tried to find the same file for macOS but I couldn't.
#I hope you will understand me.

#third task
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

#this code demonstrates that the most popular product is Kidney Beans and the most popular combination of products is Kidney Beans and Eggs.
# this code piece use the aprirori algorithm to find the most popular products and combinations of products with minimum support of 60%.
# association rules functoin to generate the rules with minimum confidence of 50%.
