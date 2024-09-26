#main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#1
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
sns.set_theme(style='whitegrid')

#density plot
plt.figure(figsize=(10, 6))
sns.kdeplot(median_expression, fill=True)

plt.title('Density Plot of Per-Gene Median Log2 Expression Values')
plt.xlabel('Median Log2 Expression')
plt.ylabel('Density')

plt.show()


#2
#affected by heat vs thermoneutral
#load metadata
metadata = pd.read_csv('data\SRP092257\metadata_SRP092257.tsv', sep='\t')

#func to assign groups based on 'refinebio_title'
def assign_group(title):
    if 'Control' in title:
        return 'Thermoneutral'
    elif 'Heat Stress' in title:
        return 'Affected by Heat'
    else:
        return 'Unknown'

#create 'Group' col
metadata['Group'] = metadata['refinebio_title'].apply(assign_group)

#ensure sample names match between expression data and metadata
expression_samples = df.columns.tolist()[1:]
metadata_samples = metadata['refinebio_accession_code'].tolist()

#check for missing samples
missing_samples_in_metadata = set(expression_samples) - set(metadata_samples)
if missing_samples_in_metadata:
    print(f"Warning: The following samples are missing in metadata: {missing_samples_in_metadata}")

missing_samples_in_expression = set(metadata_samples) - set(expression_samples)
if missing_samples_in_expression:
    print(f"Warning: The following samples are missing in expression data: {missing_samples_in_expression}")

#transpose the expression data
df.set_index('Gene', inplace=True)
log_expression_data = np.log2(df + 1)
transposed_data = log_expression_data.T.reset_index()
transposed_data.rename(columns={'index': 'Sample'}, inplace=True)

#merge metadata with expression data
merged_data = pd.merge(metadata, transposed_data, left_on='refinebio_accession_code', right_on='Sample')

#set 'Sample' as index
merged_data.set_index('Sample', inplace=True)

#prep data for dimensionality reduction
expression_columns = df.index.tolist()
expression_values = merged_data[expression_columns].values

#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(expression_values)
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=merged_data.index)
pca_df['Group'] = merged_data['Group']

#Plot PCA
sns.set_theme(style='whitegrid')
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1', y='PC2',
    hue='Group',
    palette={'Thermoneutral': 'blue', 'Affected by Heat': 'red'},
    data=pca_df,
    s=100,
    alpha=0.7
)
plt.title('PCA of Gene Expression Data')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% Variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% Variance)')
plt.legend(title='Group')
plt.show()

#t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
tsne_components = tsne.fit_transform(expression_values)
tsne_df = pd.DataFrame(data=tsne_components, columns=['t-SNE1', 't-SNE2'], index=merged_data.index)
tsne_df['Group'] = merged_data['Group']

#Plot t-SNE
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='t-SNE1', y='t-SNE2',
    hue='Group',
    palette={'Thermoneutral': 'blue', 'Affected by Heat': 'red'},
    data=tsne_df,
    s=100,
    alpha=0.7
)
plt.title('t-SNE of Gene Expression Data')
plt.xlabel('t-SNE1')
plt.ylabel('t-SNE2')
plt.legend(title='Group')
plt.show()

#UMAP
import umap
umap_reducer = umap.UMAP(n_components=2, random_state=42)
umap_components = umap_reducer.fit_transform(expression_values)
umap_df = pd.DataFrame(data=umap_components, columns=['UMAP1', 'UMAP2'], index=merged_data.index)
umap_df['Group'] = merged_data['Group']

#Plot UMAP
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='UMAP1', y='UMAP2',
    hue='Group',
    palette={'Thermoneutral': 'blue', 'Affected by Heat': 'red'},
    data=umap_df,
    s=100,
    alpha=0.7
)
plt.title('UMAP of Gene Expression Data')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.legend(title='Group')
plt.show()