import pandas as pd
import os
import wget
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

def download_script():
    fname_csv = '../data/info/BIL_doi.csv'
    folder_output = '../data/raw/seu_nature/swc'
    request_url='https://doi.brainimagelibrary.org/doi/10.35077/g.73'

    try:
        os.makedirs(folder_output)
    except:
        pass
    
    page = requests.get(request_url)
    soup = BeautifulSoup(page.content, 'html.parser')
    links = soup.find_all("a")
    
    df = pd.DataFrame()
    for link in links: 
        url = link["href"]
        if url[-4:] != '.swc':
            continue
        print(url)
        fname = wget.download(url, out=folder_output)
        fname = os.path.basename(fname)
        df = df.append({'link':url,'fname':fname},ignore_index=True)
    
    df.to_csv(fname_csv)

def link_neuron_to_info():
    fname_info = '../data/info/41586_2021_3941_MOESM4_ESM.xlsx'
    fname_doi = '../data/info/BIL_doi.csv'

    df_info = pd.read_excel(fname_info)
    df_swc = pd.read_csv(fname_doi)
    dict_swc = defaultdict(dict)
    for idx, row in df_swc.iterrows():
        tmp = row['fname']
        tmp = tmp[11:]
        if '_reg.swc' in tmp:
            key = 'reg'
            tmp = tmp.replace('__reg.swc','')
            tmp = tmp.replace('_reg.swc','')
        else:
            key = 'raw'
            tmp = tmp.replace('.swc','')
        dict_swc[tmp]['formatted name'] = tmp
        dict_swc[tmp]['swc__%s__fname'%key] = row['fname']
        dict_swc[tmp]['swc__%s__url'%key] = row['link']
    df_merged = pd.DataFrame()
    for key in dict_swc:
        df_merged = df_merged.append(dict_swc[key], ignore_index=True)
    df_merged = df_merged.merge(df_info, how='outer', on='formatted name')
    df_merged.to_csv('../data/info/BIL_recon_infos.csv')

if __name__=='__main__':
    download_script()
    link_neuron_to_info()
