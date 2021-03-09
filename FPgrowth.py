#CSDS 435- HW3  03/09/2021
#Maryam Ghasemian (mxg708)

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules

# load dataset
transaction = [['Bread', 'Milk'],['Bread', 'Diaper', 'Beer', 'Eggs'],['Milk', 'Diaper', 'Beer', 'Coke'],['Bread', 'Milk', 'Diaper', 'Beer'],['Bread','Milk', 'Diaper', 'Coke']]


te = TransactionEncoder()
te_array = te.fit(transaction).transform(transaction)
df = pd.DataFrame(te_array, columns=te.columns_)
print("dataset: ")
print(df)
print(" ")
# finding frequent itemsets using FP growth
frequent_itemsets_fp=fpgrowth(df, min_support=0.4, use_colnames=True)
# print frequent item sets
print("frequent itemsets founded by FP growth: ")
print(frequent_itemsets_fp)
print(" ")
# find association rules
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.6)
# print association rules:
print(" association Rules:")
print(rules_fp)
