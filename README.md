# Feature Rich Pointer Generator
This is a [MLP](https://www.inf.ed.ac.uk/teaching/courses/mlp/index-2018.html) group project work in University of Edinburgh. Our work is an extension for the pointer-generator network proposed by [See (2017)](https://github.com/abisee/pointer-generator). See the report in the folder `docs` for detailed information.

## Usage
### Step 1. install pacakge
```
git clone https://github.com/JZ95/feature-rich-pointer-generator
cd feature-rich-pointer-generator
# you may create a new virutal env for that
python setup.py develop
# run the command to see the help info
frpg_run --help
```
### Step 2. get CNN/Daily Mail Data Set
you may see the instructions [here](https://github.com/abisee/cnn-dailymail) and preprocess the data on you own, or just use the data processed by us by going through the following steps.
1. Download the tarball of data from [Google Drive](https://drive.google.com/file/d/161iKccsFBqHAiuvRU1AP-6x89z07xEqN/view?usp=sharing).
2. decompress the tarball `tar -xzf data.tar.gz`, then get a folder named `data`.
3. move the `data` folder into the ROOT path of this repo.

### Step 3. run your experiment
check the SHELL scripts in the folder `scripts`, see the comments for detailed explaination.

------------------
## TIPS for Setting pyrouge.
1. install python wrapper for ROUGE
```
pip install pyrouge
```
2. set rouge path
clone the ROUGE PERL repo
```
git clone https://github.com/andersjo/pyrouge
pyrouge_set_rouge_path /absolute/path/to/pyrouge/tools/ROUGE-1.5.5
```
3. run evaluation
```
frpg_eval_rouge --dir=/path/to/your/result
```

## Docker
Refer the file `docs/docker_usage.md` for guidelines to deploy this repo using docker.

------------------
### Collaborators: 
[@Shihao Liu](https://github.com/HrBlack), [@Christos Drou](https://github.com/cdroutsas)