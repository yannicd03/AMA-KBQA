# AMA KBQA 


## Setup
- Clone the repo
- Download the KQAPro Dataset from [here](https://huggingface.co/datasets/drt/kqa_pro/blob/main/kb.json)
- place the kb.json file into this folder: "db/datasets/kqapro"
- Run `convert_kb_to_nt.py` in order to transform the .json file to .nt, a format which can be read by qlever
- Start the Databases by using
```bash
docker compose up -d
```
- Populate the Database by using
```bash
docker exec -i virtuoso_db isql 1111 dba kit_ama_kbqa "EXEC=ld_dir('/usr/share/proj', 'kb.nt', 'http://kqapro.org/kb'); rdf_loader_run(); checkpoint;"
``` 
- (WIP) Setup Qdrant



You can test Virtuoso by opening http://localhost:8890/sparql and running this query: 
```sparql
PREFIX ex:   <http://kqapro.org/entity/>
PREFIX prop: <http://kqapro.org/property/>
PREFIX attr: <http://kqapro.org/attribute/>
PREFIX qual: <http://kqapro.org/qualifier/>
PREFIX unit: <http://kqapro.org/unit/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>

SELECT ?date
WHERE {
  ?person rdfs:label "Barack Obama" .
  ?person attr:date_of_birth ?date .
}
```
