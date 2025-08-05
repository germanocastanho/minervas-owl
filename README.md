# 🦉 Minerva's Owl

Designed for researchers who want to optimize their time and deepen their investigations, **Minerva's Owl** is an intelligent virtual assistant capable of analyzing, summarizing, and dialoguing with **Markdown files** from **Obsidian** vaults. A true research support tool! 🔬

<div align="center">
  <img style="max-width: 100%; height: auto;" src="assets/owl.jpg" />
  <p>
    <i>Icon of knowledge that emerges at the twilight of reason, the figure of <b>Minerva's Owl</b> refers to <b>Hegel's philosophy,</b> symbolizing philosophical reflection, which understands the world only "at twilight"; that is, only after the consummation of historical processes.</i>
  </p>
</div>

# 🚀 Main Features

- **Interactive Chatbot:** 🤖 Powerful tool to stimulate research and scientific development.
- **Intelligent Model:** 🧠 Uses OpenAI's GPT-4o model, leading to more coherent responses.
- **User-Friendly Interface:** 🎨 Interaction through a minimalist and visually appealing GUI.

# ✅ Prerequisites

- **Python 3.12+**, available through the [**official website**](https://www.python.org/downloads/).
- **OpenAI API key**, accessible through the [**platform**](https://platform.openai.com/login).
- `.env` file with the **`OPENAI_API_KEY` variable** configured.

# 🛠️ Local Installation

```bash
# Clone the repository
git clone https://github.com/germanocastanho/coruja-minerva.git

# Navigate to the directory
cd coruja-minerva

# Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
echo "OPENAI_API_KEY=YOUR_API_KEY" > .env
```

# 🗒️ Usage Instructions

- Install the chatbot following the instructions above.
- Move your Obsidian files to the **`data/`** directory.
- Finally, run the `demo.py` script to start the chatbot.

# 📜 Free Software

Distributed under the [**GNU GPL v3**](LICENSE), ensuring freedom - as in "free speech" - to use, modify, and redistribute the software, as long as these freedoms are preserved in any derivative versions. By using or contributing, you support the **free software** philosophy and help build a libertarian technological environment! ✊

