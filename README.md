# MLP Project
This is our Workspace for MLP project. Check the [google docs](https://docs.google.com/document/d/1E2p1h7s7PhCX-CSWkcq4TnxyZOLhmIXH6-iZFe6VzN4/edit?usp=sharing) for our weekly TODO list. Cheers. 🍺

## How to use
1. Download the tarball of data from [Google Drive](https://drive.google.com/file/d/161iKccsFBqHAiuvRU1AP-6x89z07xEqN/view?usp=sharing). I have processed the data already.
2. unzip the tarball `tar -xzf data.tar.gz`, then get a folder named `data`.
3. move the `data` folder into the ROOT path of our porject.
3. run the SHELL scripts in the folder `scripts`.

## Set pyrouge properly.
1. install python wrapper for ROUGE
```
pip install pyrouge
```
2. set rouge path
```
pyrouge_set_rouge_path /absolute/path/to/mlp_proj/pyrouge/tools/ROUGE-1.5.5
```
3. run evaluation in dir scripts

### see scripts and comments for detailed explaination
