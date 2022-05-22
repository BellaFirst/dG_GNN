from turtle import pu
import pubchempy as pcp
from rdkit import Chem
import urllib3, os, json
import numpy as np

def initialize_compound():

    compound = {'name': [],
                'identifiers': {'KEGG': []},
                'cpd_code': '',
                'glycan_code': '',
                'pubchem_cid': None,
                'pubchem_sid': None,
                'smile': ''
                }

    return compound

save_dir = 'data/KEGG_COMPOUNDS_SMILES_2/'
if not os.path.exists(save_dir): os.mkdir(save_dir)

query = urllib3.PoolManager()

pubchems = query.request('GET', 'https://rest.kegg.jp/conv/pubchem/compound')
pubchems = pubchems.data.decode('UTF-8')
pubchems = pubchems.strip().split('\n')


pubchem_cids = dict()
pubchem_sids = dict()
count = 0

for pubchem in pubchems:

    count += 1 

    keggs, pubs = pubchem.split('\t')

    kegg_id = keggs.split(':')[1]

    pub_sid = pubs.split(':')[1]

    pubchem_sids[kegg_id] = pub_sid 

    try:
        # from sid to cid
        substance = pcp.Substance.from_sid(pub_sid)
        cid = substance.standardized_cid

        if cid is not None:

            pubchem_cids[kegg_id] = substance.standardized_cid

            print("pubchem CID of %s is %s" %(kegg_id, pubchem_cids[kegg_id]))

            if count%500==0:
                cid_file = save_dir + 'compounds_cids_' + str(count) + '.json'
                with open(cid_file, 'w') as f:
                    json.dump(pubchem_cids, f)

    except:

        continue   

cid_file = save_dir + 'compounds_cids_all.json'
with open(cid_file, 'w') as f:
    json.dump(pubchem_cids, f)



compounds = query.request('GET', 'http://rest.kegg.jp/list/compound')
compounds = compounds.data.decode('UTF-8')
compounds = compounds.strip().split('\n')

cpd = dict()
smiles = dict()
non_smile = []
count = 0

for compound in compounds: # [12646:]:

    count += 1

    code, names = compound.replace('cpd:', '').split('\t')

    names = names.split(';')  # ['H2O','Water']

    cpd[code] = initialize_compound()
    cpd[code]['cpd_code'] = code
    cpd[code]['identifiers']['KEGG'].append(code)

    for name in names:
        cpd[code]['name'].append(name.strip())

    
    if code in pubchem_sids.keys():
        
        cpd[code]['pubchem_sid'] = pubchem_sids[code]


    # get smiles
    if code in pubchem_cids.keys():

        cpd[code]['pubchem_cid'] = pubchem_cids[code]

        try:

            c = pcp.Compound.from_cid(cpd[code]['pubchem_cid'])
            cpd[code]['smile'] = c.isomeric_smiles

        except:
            non_smile.append(code)
            continue

    elif os.path.exists('data/KEGG_MOLS/' + code + '.mol'):

        try:
            m = Chem.MolFromMolFile('data/KEGG_MOLS/' + code + '.mol')
            cpd[code]['smile'] = Chem.MolToSmiles(m)

        except:
            non_smile.append(code)
            continue

    else:
        non_smile.append(code)
        continue

    print("smile of %s is : %s"%(code, cpd[code]['smile']))

    if count%500==0:
        cpd_file = save_dir + 'compounds_' + str(count) + '.json'
        with open(cpd_file, 'w') as f:
            json.dump(cpd, f)



cpd_file = save_dir + 'compounds_all.json'
with open(cpd_file, 'w') as f:
    json.dump(cpd, f)

non_smile_file = save_dir + 'compound_non_smile.npy'
np.save(non_smile_file, non_smile)

# smile_file = 'data/KEGG_COMPOUNDS_SMILES/compounds_smiles.json'
# with open(smile_file, 'w') as f:
#     json.dump(smiles, f)




