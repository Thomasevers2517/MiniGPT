# MiniGPT
This is a proper codebase implementation of a language model. Several factors of the project can be changed to test the performance effects.


## Installation
First run the following command to clone this repository on your local machine.

```git clone git@github.com:Thomasevers2517/MiniGPT.git ```

You can install all the dependecies needed to run this project in a Conda environment:

```conda env create -f environment.yml```

Afterwards, you can activate the newly created environment with the command:
 
```conda activate gpt-env```


## Running The Code

First, download the pre-trained models [here](https://drive.google.com/file/d/1FbW1f4KNVMJA5CtWMF59sN1bbDOTwa6V/view?usp=sharing) and unzip the file
in the root of this project. After completing this step, you can generate text from the command line using the pre-trained models as follows:

```python src/generate.py --model simple --length 30 --prompt "ROMEO:"```

Or more concisely:

```python src/generate.py -m openai -l 50 -p "JULIETA:"```