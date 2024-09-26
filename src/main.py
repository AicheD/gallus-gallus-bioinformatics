#main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#load data into python
df = pd.read_csv('data\SRP092257\data_with_gene_names.tsv', sep='\t')

#Check size of expression matrix
num_genes, num_samples = df.shape
print(f"The expression matrix has {num_genes} genes and {num_samples - 1} samples.")

#get number of genes included
num_unique_genes = df['Gene'].nunique()
print(f"The dataset includes {num_unique_genes} unique genes.")

#how much variation in the data
expression_data = df.drop('Gene', axis=1)
log_expression_data = np.log2(expression_data + 1)

median_expression = log_expression_data.median(axis=1)

#style choice
sns.set(style='whitegrid')

#density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(median_expression, shade=True)

plt.title('Density Plot of Per-Gene Median Log2 Expression Values')
plt.xlabel('Median Log2 Expression')
plt.ylabel('Density')

plt.show()