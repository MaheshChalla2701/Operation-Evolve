# Google Colab Training Guide for Operation Evolve

This guide covers how to move your `Operation-Evolve` local training pipeline into the cloud using Google Colab. By running in Colab, you get access to powerful GPUs (like a T4 or A100) which will drastically speed up how quickly your model learns. 

Because we map the output directly to Google Drive, your `best_model.pt` weights and `config.json` will automatically sync to your PC without you needing to do manual downloads.

---

## Step 1: Prepare Google Drive
1. Go to [Google Drive](https://drive.google.com/) in your browser.
2. Create a brand new folder called `AI_Models`.
3. Inside `AI_Models`, you don't need to do anything yet. We will clone your GitHub repository directly into this folder using Python.

## Step 2: Open a New Colab Notebook
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook** (blue button at the bottom right of the popup).
3. Name the notebook something like `Operation_Evolve_Trainer.ipynb` at the top left.

> [!IMPORTANT]
> **Turn on the GPU!**
> 1. Go to the top menu bar and click **Runtime** > **Change runtime type**.
> 2. Under "Hardware accelerator", select **T4 GPU** (or any higher GPU if you have Colab Pro).
> 3. Click **Save**. If you skip this, the training won't be any faster than your Windows PC!

## Step 3: Mount Your Google Drive
In your very first code cell, paste the following code:
```python
from google.colab import drive
drive.mount('/content/drive')
```
1. Click the "Play" button next to the cell (or press `Shift + Enter`).
2. A popup will ask you to connect to Google Drive. Click **Connect to Google Drive**, select your Google account, and click **Allow**.

## Step 4: Clone the Repository into Drive
Add a new code cell (hover over the bottom of a cell and click **+ Code**). Paste this code to jump into the folder you made and download your latest code:

```bash
# Go to the folder you created in Step 1
%cd /content/drive/MyDrive/AI_Models

# Download your code directly from GitHub
!git clone https://github.com/MaheshChalla2701/Operation-Evolve.git

# Move inside the exact folder where the training scripts live
%cd Operation-Evolve/EVOLVE
```
*Run the cell. It will download the codebase into your Google Drive permanently.*

## Step 5: Install Requirements
Add another code cell and run this to install PyTorch, Tiktoken, Groq, etc:
```bash
!pip install -r requirements.txt
```

## Step 6: Set Your API Keys
Since you are in the cloud, the notebook doesn't have your Windows `.env` file! We need to inject your Groq API key into the environment so `evolve.py` can still reach the LLM agent.

Create a new cell and add this:
```python
import os
# Replace the Xs below with your ACTUAL API KEY
os.environ["GROQ_API_KEY"] = "gsk_XXXXXXXXXXXXXXXXXXXXXXXXX"
```
*Run that cell.*

## Step 7: Start the Infinite Trainer
Now for the exciting part. Create one final code cell:
```bash
!python evolve.py
```
*Run it!*

> [!NOTE]
> Because you are running from the `/content/drive/...` directory, every time `evolve.py` says `[AutoSave] best_model.pt updated`, that massive model file is physically being saved to your Google Drive servers.

---

### Step 8: Resume Locally on your Laptop
When you are done training (or if Colab disconnects):
1. Wait a moment for Google Drive to finish syncing.
2. Go to your local Windows PC.
3. Open your Google Drive folder and navigate to `AI_Models/Operation-Evolve/EVOLVE`.
4. Copy the `best_model.pt`, `config.json`, and `evolution_log.json` files.
5. Paste and overwrite them into your local `C:\Users\Mahesh\OneDrive\Documents\GitHub\Operation-Evolve\EVOLVE` directory.
6. Open your Windows terminal and run `python ask.py "your prompt here"`.
7. You are now communicating with the brain you just trained in the cloud!
