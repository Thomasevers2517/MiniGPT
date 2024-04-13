# miniGPT
This is the code base used to build and evaluate a miniGPT model for the course EE4685 Machine Learning, a Bayesian Perspective at Delft University of Technology. The code follows the traditional Transformer architecture and it is based on the [minGPT](https://github.com/karpathy/minGPT) repository by Andrej Karpathy.

Modifications from the original code include:

- Code rewritten to use [Pytorch Lighting](https://lightning.ai/docs/pytorch/stable/) for more readability and easier training.
- Implemented [Memorization-free Decoding](https://arxiv.org/pdf/2210.17546.pdf) to reduce data memorization.
- Hyperparameter tuning for two different tokenizers: a simple character-level tokenizer and a pre-trained byte pair tokenizer from OpenAI called [tiktoken](https://github.com/openai/tiktoken).
- Extended the tokenizer with special tokens that indicate when a different character is speaking. The hope is that a GPT trained with these special tokens will give better predictions. **Results for this are not shown in the report but you can find the code in [src/utils/preprocessing.py](src/utils/preprocessing.py).**


## Installation
First, clone the repository and install the requiered dependecies.

```bash
git clone https://github.com/Thomasevers2517/MiniGPT.git
cd MiniGPT
conda env create -f environment.yml
conda activate gpt-env
```

## Usage

You can train a GPT with the settings provided in the [src/config/run.py](src/config/run.py) config file:

```bash
$ python src/train.py
```

The default settings train a GPT with a context size of 256 characters (`tokenizer=simple` specifies that the prediction is done on a character level), in a 6-layer Transformer Decoder with 6 heads per layer. Some additional parameteres, the embedding dimesion has a size of 128, learning rate is 1e-4 and dropout is set to 0.2. We use 16-bit precision for faster training. 


## Hyperparameter tuning

For the hyperparameter tuning, we used [WandB Sweeps](https://docs.wandb.ai/guides/sweeps). The first sweep was for the simple, character-level, tokenizer.



| run | block\_size | n\_embd | n\_head | n\_layer | lr (1e3) | test\_loss      | runtime (s) | params |
|-----|-------------|---------|---------|----------|----------|-----------------|-------------|--------|
| 1   | 64          | 512     | 8       | 4        | 0.2110   | **0.3013** | 4368        | 12.7 M |
| 2   | 64          | 512     | 4       | 2        | 0.2110   | 0.4832          | 3286        | 6.4 M  |
| 3   | 16          | 512     | 2       | 8        | 0.2222   | 1.0530          | 2457        | 25.3 M |
| 4   | 64          | 32      | 4       | 8        | 2.9895   | 1.5110          | 439         | 107 K  |
| 5   | 16          | 32      | 16      | 8        | 4.4603   | 1.5996          | 1047        | 105 K  |
| 6   | 16          | 32      | 8       | 16       | 6.2460   | 1.6095          | 676         | 206 K  |
| 7   | 16          | 32      | 4       | 4        | 8.0924   | 1.6780          | 255         | 55.2 K |
| 8   | 64          | 32      | 4       | 16       | 0.0345   | 1.7611          | 3986        | 208 K  |
| 9   | 64          | 32      | 2       | 2        | 0.3401   | 1.7772          | 484         | 31.6 K |
| 10  | 64          | 32      | 2       | 4        | 0.3097   | 1.7934          | 319         | 56.8 K |
| 11  | 4           | 128     | 4       | 4        | 5.4815   | 1.9559          | 212         | 809 K  |
| 12  | 4           | 128     | 4       | 4        | 0.4984   | 1.9693          | 251         | 809 K  |
| 13  | 4           | 128     | 4       | 8        | 5.3010   | 1.9885          | 416         | 1.6 M  |
| 14  | 4           | 32      | 8       | 4        | 8.5951   | 2.0202          | 260         | 54.8 K |
| 15  | 4           | 32      | 2       | 8        | 0.3183   | 2.0284          | 315         | 105 K  |
| 16  | 4           | 32      | 2       | 4        | 0.4156   | 2.1775          | 135         | 54.8 K |
| 17  | 16          | 8       | 4       | 4        | 7.4257   | 2.2230          | 226         | 4.6 K  |
| 18  | 16          | 8       | 4       | 4        | 6.0856   | 2.2396          | 251         | 4.6 K  |
| 19  | 4           | 128     | 2       | 8        | 8.1141   | 2.3242          | 171         | 1.6 M  |
| 20  | 16          | 8       | 8       | 4        | 0.6784   | 2.3906          | 394         | 4.6 K  |

The second sweep was for the OpenAI tokenizer:

| run | block\_size | n\_embd | n\_head | n\_layer | lr (1e3) | test\_loss      | runtime (s) | params |
|-----|-------------|---------|---------|----------|----------|-----------------|-------------|--------|
| 1   | 16          | 512     | 2       | 4        | 0.0454   | **0.7286** | 15345       | 64.1 M |
| 2   | 16          | 128     | 4       | 8        | 0.4436   | 1.0057          | 10121       | 14.6 M |
| 3   | 4           | 1024    | 8       | 2        | 0.2093   | 2.6754          | 3032        | 128 M  |
| 4   | 4           | 512     | 4       | 4        | 0.0929   | 2.7086          | 2511        | 64.1 M |
| 5   | 4           | 512     | 2       | 2        | 0.3627   | 2.8178          | 1042        | 57.8 M |
| 6   | 4           | 128     | 8       | 8        | 0.4841   | 3.3266          | 2006        | 14.6 M |
| 7   | 4           | 128     | 4       | 2        | 0.1497   | 3.6489          | 977         | 13.3 M |
| 8   | 4           | 32      | 4       | 8        | 0.4366   | 4.1076          | 1016        | 3.4 M  |
| 9   | 4           | 32      | 2       | 8        | 0.3970   | 4.1622          | 835         | 3.4 M  |
| 10  | 4           | 32      | 8       | 4        | 0.3629   | 4.1727          | 944         | 3.3 M  |
| 11  | 4           | 32      | 8       | 4        | 0.3324   | 4.2061          | 839         | 3.3 M  |
| 12  | 4           | 32      | 8       | 2        | 0.3968   | 4.2095          | 753         | 3.3 M  |
| 13  | 4           | 32      | 4       | 8        | 0.1600   | 4.2331          | 2276        | 3.3 M  |
| 14  | 16          | 8       | 4       | 4        | 0.3930   | 4.5503          | 62981       | 857 K  |
| 15  | 16          | 8       | 2       | 8        | 0.4298   | 4.5528          | 2348        | 861 K  |
| 16  | 4           | 8       | 8       | 4        | 0.3394   | 4.8153          | 1238        | 857 K  |
| 17  | 4           | 8       | 8       | 8        | 0.2698   | 4.8364          | 2174        | 866 K  |
| 18  | 4           | 8       | 8       | 4        | 0.4992   | 4.8508          | 1174        | 857 K  |
| 19  | 4           | 8       | 8       | 4        | 0.4281   | 4.8550          | 924         | 857 K  |
| 20  | 16          | 8       | 2       | 4        | 0.1014   | 4.9006          | 4363        | 857 K  |
| 21  | 4           | 8       | 2       | 4        | 0.2952   | 10.0787         | 187         | 857 K  |



## Pre-trained 

If you do not have the patience to train the models from scratch, you can download the checkpoints of our best performing models [here](https://drive.google.com/file/d/1FbW1f4KNVMJA5CtWMF59sN1bbDOTwa6V/view?usp=sharing). Make sure to unzip the downloaded file in the root directory so that it follows the structure:

```bash
MiniGPT
├── checkpoints
│   ├── best_openai.ckpt
│   └── best_simple.ckpt
```

You can then generate text using the pre-trained models:

```bash
$ python src/generate.py --model simple --length 30 --prompt "ROMEO:"
# writen more concisely and using memfree decoding.
$ python src/generate.py -m openai -l 50 -p "JULIETA:" -mf True
```

## Results

Here is a sample of generated text using the best performing simple token model:

```
ROMEO:
No matter, uncle? speak you?

JULIET:
No, madam; we have cuose as ,
Where you may hand openly, to shorten his eyes:
He has the old father charity nor shame to me:
Uncharitably with me have you dead?

SEBASTIAN:
What, art thou waking?

AUTOLYCUS:
No, good-faced sir; no, sweet sir.

CAPULET:
O, God ye god-den, I had rather lie.
```

And here is a sample using the best performing `tiktoken` model:

```
ROMEO:
Then plainly know my heart's dear love is set
On the fair daughter of rich Capulet;
As who should say, if I should sleep or eat,
'Twere heavier grief shows still some want of wit.

Messenger:
My gracious sovereign, now in Devonshire,
As I before unparted to your worship,
I am to get a man,--whate'er he be,
It skills not much. we'll fit him to our turn,--
And he that lies andant may answer a letter.

GLOUCESTER:
Your eyes drop millstones, when heointed king,
As was the covert'st shelter'd traitor
That us wretched by the death of thee,
Than I know how much company to hoar.
```

## Discussion

- More parameters generally imply better model performance but also longer training time. 
  
-  The average loss for the `tiktoken` model is higher than the simple token model, since its  vocabulary size is much larger and it thus have more options to choose from when predicting the next token.
  
- The next token predictions tend to be more meaningful in the `tiktoken` model.

- Multiple heads are only meaningful if the embedding size is sufficiently large. This is expected since
the head size is defined as the embedding size divided by the number of heads.

- For both tokenizers, the top models seem to have relatively large block sizes which indicates that the models are sufficiently complex to use a longer text history.
  
- The generated text looks quite convincing: both models can learn a dialogue structure between different speakers, and the language feels like old English. 


- The simple tokenizer text, at first glance, looks like proper Shakespeare; however,
when looking at a sequence of more than a few words, the text makes little sense. 

- The tiktoken model, however,
generates text impressively similar to Shakespeare. However, looking deeper, we realize some sub-strings are memorized from the training data. 