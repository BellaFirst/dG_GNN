import urllib3
import os
import json

molfile_dir = 'data/KEGG_MOLS/'
if not os.path.exists(molfile_dir): os.mkdir(molfile_dir)

compound_file = 'data/KEGG_COMPOUNDS_NAMES/compounds.json'
with open(compound_file, 'r') as f:
    compounds = json.load(f)

for cname in list(compounds.keys())[14800:]:

    if os.path.exists(molfile_dir + cname + '.mol'): 
        print("----already downloaded %s ----"%(cname))
        continue

    http = urllib3.PoolManager()
    head = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.109 Safari/537.36"}

    url = "https://www.kegg.jp/entry/-f+m+" + cname

    mol = http.request('GET', url, headers=head)

    moldata = mol.data.decode('UTF-8')

    if moldata == '':
        continue

    else:
        print("----download %s ----"%(cname)) 
        with open(molfile_dir + cname + '.mol', 'w', encoding='utf-8') as f:
            f.write(moldata)


print("===== Done! =====")





