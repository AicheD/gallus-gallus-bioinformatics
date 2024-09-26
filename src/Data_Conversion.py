import pandas as pd
import requests
import time

df = pd.read_csv('data\SRP092257\SRP092257.tsv', sep='\t')

#format string data
df['Gene'] = df['Gene'].astype(str).str.split('.').str[0]

#store list of unique Ensembl Gene IDs
ensembl_ids = df['Gene'].unique().tolist()

#Ensembl REST API query func
def query_ensembl(ids):
    server = "https://rest.ensembl.org"
    ext = "/lookup/id"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    #Prep data for POST request
    data = {"ids": ids}

    #Send the POST request
    response = requests.post(server + ext, headers=headers, json=data)

    #Handle rate limiting with time lib
    while response.status_code == 429:
        print("Rate limited, sleeping for 1 second...")
        time.sleep(1)
        response = requests.post(server + ext, headers=headers, json=data)

    if not response.ok:
        response.raise_for_status()

    return response.json()

#Map Ensembl IDs to Gene Names
id_to_name = {}

#process in batches
chunk_size = 1000  

#Process IDs in chunks
for i in range(0, len(ensembl_ids), chunk_size):
    chunk = ensembl_ids[i:i + chunk_size]
    print(f"Processing IDs {i + 1} to {i + len(chunk)} of {len(ensembl_ids)}...")
    result = query_ensembl(chunk)
    for ensembl_id in chunk:
        entry = result.get(ensembl_id)
        if entry:
            gene_name = entry.get('display_name')
            id_to_name[ensembl_id] = gene_name
        else:
            id_to_name[ensembl_id] = None
    time.sleep(1)

#Create a df from the mapping
mapping_df = pd.DataFrame.from_dict(id_to_name, orient='index', columns=['Gene name']).reset_index()
mapping_df.rename(columns={'index': 'Gene'}, inplace=True)

# Merge dataset with the mapping df
merged_df = pd.merge(df, mapping_df, on='Gene', how='left')

#Replace Ensembl IDs w/ Gene Names
merged_df['Gene'] = merged_df['Gene name'].combine_first(merged_df['Gene'])

#Drop the 'Gene name' col
merged_df = merged_df.drop(columns=['Gene name'])

#export resulting df to tsv file
merged_df.to_csv('data_with_gene_names.tsv', sep='\t', index=False)

print("Mapping complete. Output saved to 'data_with_gene_names.tsv'.")