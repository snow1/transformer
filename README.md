﻿# EEG_trans

For reference only
The first time run setup.py, get all the command line tools ready and run that in the dtu server with the package.
 
module swap python3/3.9.11


Copy all this and put it in the dtu's server terminal:
cd ~/Desktop
mkdir transformer
cd transformer
module load python3/3.9.6
python3 -m venv project-env
source project-env/bin/activate
python -m pip install git+https://github.com/FredslundMagnus/dtu-package.git
python -m pip install torch torchvision matplotlib einops pandas scipy torchsummary
git config --global credential.helper store
git clone https://github.com/snow1/transformer.git
yes | cp project-env/bin/dtu_server ~/bin/dtu
cd transformer
deactivate
dtu


# copy files from local to remote 
scp -r  * ssh dtu:~/Desktop/transformer/transformer/data/14-Subjects-Dataset/

# test1  
transformer
# test2
with pretrained model in models folder
# test3
transformer with 22 channel
# test4
transformer with 22 channel and pretrained model
# test5
LSTM
# test6
linear regression
