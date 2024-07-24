# NLM
Family of Nepali Language Models
### Start

#### Clone the repo
```bash
git clone https://github.com/mridul3301/NLM
```
#### Navigate to the directory & make python virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
#### Install requrements
```bash
pip install -r requirements.txt
```

#### Create a new directory "data" and make two directories inside it "csv" and "text"
```bash
mkdir data
cd data
mkdir csv
mkdir text
cd ..
```

### Tokenize the text data

#### Make the script executable
```bash
chmod +x run_tokenization.sh
```

#### Execute the script
```bash
./scripts/run_tokenization.sh
```