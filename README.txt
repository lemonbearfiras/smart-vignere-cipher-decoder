🔐 Hybrid Vigenère Cipher Cracker

A powerful Python tool for breaking **Vigenère ciphers** using a mix of:
- 📖 Dictionary-based key search  
- 📊 Frequency analysis (chi-squared scoring)  
- 🔤 N-gram scoring (bigrams & trigrams)  
- 🧠 Grammar and word detection heuristics  

It ranks possible decryptions and returns the most **English-like plaintext**.

 ⚙️ Features
- Supports **known and unknown key lengths**
- Smart scoring using **English frequency + grammar rules**
- Handles **short ciphertexts** with fallback frequency attacks
- Multiple output formats: **text, JSON, CSV**
- Interactive mode for easy usage

 🚀 Usage

 ▶️ Basic (unknown key length)
```bash
python script.py "CIPHERTEXT"

 ⚙️ Features

* Supports **known and unknown key lengths**
* Smart scoring using **English frequency + grammar rules**
* Handles **short ciphertexts** with fallback frequency attacks
* Multiple output formats: **text, JSON, CSV**
* Interactive mode for easy usage

 🚀 Usage

 ▶️ Basic (unknown key length)

python script.py "CIPHERTEXT"

 🎯 Specify key length

python script.py "CIPHERTEXT" --key-length 5

 ⚡ Limit key length range

python script.py "CIPHERTEXT" --min-key-length 3 --max-key-length 10

 🔍 Show top results

python script.py "CIPHERTEXT" --top 10

 🧠 More accurate (recommended)

python script.py "CIPHERTEXT" \
--min-key-length 3 \
--max-key-length 10 \
--prefilter-limit 400 \
--top 10 \
--explain-top 3

 📊 JSON output

python script.py "CIPHERTEXT" --json

 📄 Save results to file

python script.py "CIPHERTEXT" --output results.txt

 📈 Export to CSV

python script.py "CIPHERTEXT" --csv results.csv

 🤫 Quiet mode (best plaintext only)

python script.py "CIPHERTEXT" --quiet

 🧪 Interactive mode

python script.py --interactive

 🛠️ Command Options

--key-length = exact key length to use  
--min-key-length = minimum key length to try  
--max-key-length = maximum key length to try  
--dictionary = path to dictionary file  
--top = number of results to display  
--prefilter-limit = candidates kept after first filtering  
--max-words = limit dictionary words tested  
--output = save output to file  
--interactive = run in interactive mode  
--short-cipher-mode = on/off for short text optimization  
--json = output in JSON format  
--best-only-json = output only best result (JSON)  
--json-pretty = pretty JSON formatting  
--csv = export results to CSV  
--quiet = print only best plaintext  
--explain-top = explain top N results  
--summary-only = show only summary  
--explain-all = explain all results  

 📌 Example

python script.py "LXFOPVEFRNHR"

Output:
Best key: LEMON  
Decrypted text: ATTACK AT DAWN

 🧠 How it Works

1. Generates candidate keys from a dictionary
2. Uses frequency analysis to rank likely decryptions
3. Scores plaintext using:

   * English letter frequency
   * word matching
   * grammar heuristics
   * n-gram patterns
4. Returns the highest scoring result

 📂 Requirements

* Python 3.x
* No external libraries required

 🏁 Notes

* Works best with **English plaintext**
* Short ciphertexts use **extra heuristics**
* Results are ranked — always check top candidates

 ⭐

If you find this useful, consider starring the repo!

