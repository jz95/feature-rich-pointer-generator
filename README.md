# MLP Project
This is our Workspace for MLP project. Check the [google docs](https://docs.google.com/document/d/1E2p1h7s7PhCX-CSWkcq4TnxyZOLhmIXH6-iZFe6VzN4/edit?usp=sharing) for our weekly TODO list. Cheers. 🍺

## UPDATE
To train the model with pos embeddings, move the file `vocab_pos.txt` under the path `/data/finished_files/vocab_pos.txt`.

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
clone the pyrouge PERL repo
```
git clone https://github.com/andersjo/pyrouge
pyrouge_set_rouge_path /absolute/path/to/pyrouge/tools/ROUGE-1.5.5
```
3. run evaluation in dir scripts

### see scripts and comments for detailed explaination

----------------------------

# How to use TensorBoard remotely on cluster
## step 1.
On you pc run:
```
ssh -L localhost:6006:localhost:6006 s1234567@student.ssh.inf.ed.ac.uk
```
which maps port 6006 on your pc to port 6006 on DICE.
## step 2.
after login on DICE, type
```
ssh -L localhost:6006:localhost:6006 mlp
```
which maps 6006 on DICE to 6006 on cluster head node.
## step 3.
After logining to the cluster
```
tensorboard --logdir=/path/to/your/dir --port=6006
```
## step 4.
Go to [localhost:6006](http://localhost:6006) in the web browser on your pc.

## TIPS
We use default port 6006 for tensorboard, you are allowed to use other port number if you like, but we don't recommend so.
