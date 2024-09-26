#main.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import pandas as pd
from statsmodels.stats.multitest import multipletests

#1------------------------------------------------------------------------------------------
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


#2---------------------------------------------------------------------------------------------
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

#3--------------------------------------------------------------------------------------------------
#prep the data for differential expression analysis
# Load the gene expression data and metadata
df = pd.read_csv('data/SRP092257/data_with_gene_names.tsv', sep='\t')
metadata = pd.read_csv('data/SRP092257/metadata_SRP092257.tsv', sep='\t')
print("Data loaded.")

# Function to assign groups (Thermoneutral or Affected by Heat)
def assign_group(title):
    if 'Control' in title:
        return 'Thermoneutral'
    elif 'Heat Stress' in title:
        return 'Affected by Heat'
    else:
        return 'Unknown'

metadata['Group'] = metadata['refinebio_title'].apply(assign_group)

# Ensure sample names match between expression data and metadata
expression_samples = df.columns.tolist()[1:]
metadata_samples = metadata['refinebio_accession_code'].tolist()

# Filter out missing samples
common_samples = set(expression_samples).intersection(metadata_samples)

# Filter data to include only common samples
df = df[['Gene'] + list(common_samples)]
metadata = metadata[metadata['refinebio_accession_code'].isin(common_samples)]

# Transpose the expression data for easier analysis
df.set_index('Gene', inplace=True)
log_expression_data = np.log2(df + 1)
transposed_data = log_expression_data.T

# Merge metadata with expression data
merged_data = pd.merge(metadata, transposed_data, left_on='refinebio_accession_code', right_index=True)
print("Data merged with metadata.")

# Perform t-tests for each gene
p_values = []
log_fold_changes = []

# Set the batch size
batch_size = 1000

# Get the total number of genes
total_genes = len(df.index)

# Process only the first 3 batches for debugging (batch 1-3)
for batch_start in range(0, min(2 * batch_size, total_genes), batch_size):
    batch_end = min(batch_start + batch_size, total_genes)
    batch_genes = df.index[batch_start:batch_end]

    print(f"Processing genes {batch_start + 1} to {batch_end} of {total_genes}...")

    # Iterate over the genes in the current batch
    for gene in batch_genes:
        group1 = merged_data.loc[merged_data['Group'] == 'Thermoneutral', gene]
        group2 = merged_data.loc[merged_data['Group'] == 'Affected by Heat', gene]

        # Perform t-test
        try:
            t_stat, p_val = ttest_ind(group1, group2, equal_var=False)
        except ValueError as e:
            print(f"Skipping gene {gene} due to error: {e}")
            continue

        # Log fold change
        log_fc = np.mean(group2) - np.mean(group1)

        p_values.append(p_val)
        log_fold_changes.append(log_fc)

    # Print progress for each batch
    print(f"Completed processing batch {batch_start // batch_size + 1}.")

# Create a DataFrame to store results
volcano_df = pd.DataFrame({
    'logFC': log_fold_changes,
    'p-value': p_values
})
# Remove any rows where p-values are invalid (NaN, inf)
volcano_df = volcano_df.replace([np.inf, -np.inf], np.nan).dropna(subset=['p-value'])

# Adjust p-values for multiple testing using Benjamini-Hochberg (False Discovery Rate)
volcano_df['adj_p-value'] = multipletests(volcano_df['p-value'], method='fdr_bh')[1]

# Plot Volcano Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='logFC', y=-np.log10(volcano_df['adj_p-value']), data=volcano_df)
plt.title('Volcano Plot of Differential Expression')
plt.xlabel('Log Fold Change')
plt.ylabel('-Log10(adjusted p-value)')
plt.axhline(y=-np.log10(0.05), color='red', linestyle='--')
plt.savefig('results/volcano_plot_batches.png')  # Save plot as PNG in results folder
plt.show()
plt.close()

print("Differential expression analysis for batches 1-3 is complete.")

'''
group_labels = merged_data['Group']

#getlist of genes
genes = expression_columns


mask_thermoneutral = group_labels == 'Thermoneutral'
mask_heat_stress = group_labels == 'Affected by Heat'

expression_values = merged_data[expression_columns]


expression_thermoneutral = expression_values[mask_thermoneutral]
expression_heat_stress = expression_values[mask_heat_stress]

print(f"Number of Thermoneutral samples: {expression_thermoneutral.shape[0]}")
print(f"Number of Affected by Heat samples: {expression_heat_stress.shape[0]}")

#calculate mean expression for each gene in both groups
mean_expression_thermoneutral = expression_thermoneutral.mean(axis=0)
mean_expression_heat_stress = expression_heat_stress.mean(axis=0)

#init a list to store p-values
p_values = []

#t-test's
for gene in genes:
    gene_expression_thermoneutral = expression_thermoneutral[gene]
    gene_expression_heat_stress = expression_heat_stress[gene]
    
    t_stat, p_val = ttest_ind(gene_expression_heat_stress, gene_expression_thermoneutral, equal_var=False)
    p_values.append(p_val)

#convert p-values to a NumPy arr
p_values = np.array(p_values)

#adjust p-values for multiple testing using FDR bh
adjusted_p_values = multipletests(p_values, method='fdr_bh')[1]

#calculate log2 fold change
log2_fold_changes_series = np.log2(mean_expression_heat_stress + 1) - np.log2(mean_expression_thermoneutral + 1)
log2_fold_changes = log2_fold_changes_series.values

#Debug shape issues*
print(f"Length of genes: {len(genes)}")
print(f"Length of log2_fold_changes: {len(log2_fold_changes)}")
print(f"Length of p_values: {len(p_values)}")
print(f"Length of adjusted_p_values: {len(adjusted_p_values)}")

#df to store results
results_df = pd.DataFrame({
    'Log2FoldChange': log2_fold_changes,
    'PValue': p_values,
    'AdjustedPValue': adjusted_p_values
}, index=genes)

results_df.index.name = 'Gene'

#Volcano plot
results_df['Significant'] = 'Not Significant'
results_df.loc[(results_df['AdjustedPValue'] < 0.05) & (abs(results_df['Log2FoldChange']) > 1), 'Significant'] = 'Significant'

#create the volcano plot
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='Log2FoldChange',
    y=-np.log10(results_df['AdjustedPValue']),
    hue='Significant',
    data=results_df,
    palette={'Not Significant': 'grey', 'Significant': 'red'},
    alpha=0.7
)

#threshold lines
plt.axhline(-np.log10(0.05), color='blue', linestyle='--')
plt.axvline(1, color='blue', linestyle='--')
plt.axvline(-1, color='blue', linestyle='--')

plt.title('Volcano Plot of Differential Expression')
plt.xlabel('Log2 Fold Change')
plt.ylabel('-Log10 Adjusted P-value')
plt.legend(title='Significance')
plt.show()

sorted_results = results_df.sort_values('AdjustedPValue')

top50_genes = sorted_results.head(50)

top50_genes.to_csv('results/top50_differentially_expressed_genes_python.csv')

sorted_results.to_csv('results/differential_expression_results_python.csv')

num_sig_genes = (sorted_results['AdjustedPValue'] < 0.05).sum()
print(f"Number of significantly differentially expressed genes (adjusted p-value < 0.05): {num_sig_genes}")

print("Top 10 differentially expressed genes:")
print(top50_genes[['Log2FoldChange', 'AdjustedPValue']].head(10))
'''

#4--------------------------------------------------------------------------------------------------



