# AMA KBQA 


## Setup
- Clone the repo
- Download the KQAPro Dataset from here
- place the kb.json file into this folder: "db/datasets/kqapro"
- Run `convert_kb_to_nt.py` in order to transform the .json file to .nt, a format which can be read by qlever
- Run `setup_qlever.bat`or `setup_qlever.sh` depending on your os.
- (WIP) Setup Qdrant
- Start the Databases by using `docker compose up -d` 


