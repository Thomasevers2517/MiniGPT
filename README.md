# MiniGPT
This is a proper codebase implementation of a language model. Several factors of the project can be changed to test the performance effects.


## Installation
First run the following command to clone this repository on your local machine.

```git clone git@github.com:Thomasevers2517/MiniGPT.git ```

You can install all the dependecies needed to run this project in a Conda environment as follows:

```conda env create -f environment.yml```

Afterwards, you can activate the newly created environment wuth the command:
 
```conda activate gpt-env```


## Running The Code

You can generate text from the command line using the pre-trained models as follows:

```python src/generate.py --model simple --length 30 --prompt "ROMEO:"```

Or more concisely:

```python src/generate.py -m openai -l 50 -p "JULIETA:"```