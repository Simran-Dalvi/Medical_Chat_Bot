# RAG Medical_Chat_Bot

## project template
using a bash script for project structure
all stored in template.sh

called this file on git terminall using 
```bash
sh template.sh
```



## now we create a virtual environment
```bash 
conda create -n medbot python=3.10
```

to check if the env exists
```bash
conda env list
```

now to activate this virtual env
```bash
conda activate medbot
```

if the activate function dosent work
```bash 
conda init bash
```
then try again after reopenong shell


## install all dependencies from requirements.txt
set up the setup.py file

and

pip install -r requirements.txt

add 
```bash 
*.egg-info/
```
to .gitignore file?

 ## we did all the experiments for the project in trials.ipynb in research file
 (save your api keys in the .env file)

 1. document loading (pdf with 637 pages)
 2. text splitting/ chunking (5859 text chunks)
 3. embedding-  done through hugging face sentence-transformers model
 4. storing it in a vector database - pinecone
    1. connect to pinecone api
    2. create an index in pinecone 
5. use langchain to convert our text into vectors in our vectorDB
    1. at the created index add these 5859 vectors
    2. retrieve the top 5 most relevant vectors for our quesy using similarity score
6. connect to our chat model- OPENAI
    1. use langchain to connect to chatgpt
    2. make a prompt template
    3. make a chain that connects our llm to prompt and then to our retrieval model
    4. get response answers

## now we do this whole in modelar form


 