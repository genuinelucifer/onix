# Walkthrough: Training LLaMA 1B on TinyStories

This guide provides a complete, step-by-step walkthrough to pretrain a LLaMA 1B parameter model from scratch using the **TinyStories** dataset. We will use a high-performance configuration optimized for ROCm/AMD hardware.

---

## 🛠️ Step 1: Download & Pre-Tokenize the Dataset

The very first step is to acquire and prepare our dataset. We use the unified Hugging Face downloader script to download, tokenize, and shard the TinyStories dataset. This streams the text directly, tokenizes it on the fly using GPT-2 encoding, and stores the processed sequences as memory-mapped binary arrays (`.npy`). This avoids system memory bottlenecks during training.

Run the following command in your workspace:

```bash
# Activate the virtual environment
source onix_env/bin/activate

# Stream, tokenize, and shard TinyStories
python download_hf.py --dataset tiny-stories
```

*This will create the tokenized shards and a `metadata.json` progress file in `datasets/tiny-stories/`.*

---

## 📐 Step 2: Configure the Model Architecture

For this walkthrough, we will use the optimized LLaMA 1B model configuration located at `configs/llama1b_512_opt.json`. This configuration is tailored to maximize hardware training throughput while keeping memory consumption low:

- **Context Length**: `512` (perfect for short story datasets).
- **Scaled Dot-Product Attention (SDPA)**: `"use_sdpa": true` (utilizes PyTorch's optimized attention kernels).
- **Gradient Checkpointing**: `"grad_checkpointing": false` (Faster training at the cost of some RAM).

---

## 🚀 Step 3: Launch the Pre-training Job

We run the training job in the background using the `./run_train.sh` Modality Dispatcher. We will pass some flags:
- `--bf16`: Enables BFloat16 mixed-precision training.
- `--compile`: Uses `torch.compile()` to compile the model graph and optimize training speed.
- `--batch-size 32`: Sets the training batch size to 32.
- `--epochs 1`: Trains the model for 1 epoch.
- `--log-freq 10`: Logs progress every 10 steps (defaults to 5).
- `--save-iters 1000`: Saves a checkpoint every 1000 steps (defaults to 0/disabled).
- `--save-limit 1`: Limits checkpoint storage to the latest checkpoint to conserve disk space (defaults to 3).
- `--eval-freq 1000`: Evaluates the model every 1000 steps (defaults to 50).
- `--num-workers 2`: Uses 2 background worker processes for data loading (defaults to 0).
- `--prefetch-factor 4`: Prefetches 4 batches per worker (defaults to 2).

Run the launch script:

```bash
./run_train.sh my-llama \
    --mode llm \
    --config configs/llama1b_512_opt.json \
    --data-dir datasets/tiny-stories/ \
    --bf16 \
    --compile \
    --batch-size 32 \
    --epochs 1 \
    --log-freq 10 \
    --save-iters 10 \
    --save-limit 1 \
    --eval-freq 10 \
    --num-workers 2 \
    --prefetch-factor 4
```

---

## 📊 Step 4: Monitor Training Progress

The launch script starts the training process in the background using `nohup`. A status file is updated in real-time with training loss, validation loss, tokens processed, and story generation samples.

### ⚡ Expected Resource Utilization & Speed
* **Memory & VRAM**: The pre-training job will consume about **61 GB of VRAM** and **10 GB of system RAM**.
* **Step Speed**: Each step with a batch size of 32 takes about **5.5 seconds**.
* **Log Interval**: With `--log-freq 10` configured, you should see status updates in the log files every **~55 seconds**.
* **Total Time**: The full training will take about `24414 * 5.5 seconds` + time for validation steps + time for saving checkpoints = $\approx$ **40 hours**.

To monitor your training run:

```bash
./train_status.sh my-llama
```

---

## 🛑 Step 5: Stop or Resume Training

### Stopping the Training Run
If you need to stop the training run at any point, you can safely terminate it:

```bash
./stop_train.sh my-llama
```

### Resuming the Training Run
The training engine automatically saves checkpoints periodically. If you stop the run or if the system reboots, you can resume training exactly where you left off (the engine will reload the model configuration, the saved optimizer states, and fast-forward the dataset loader to the exact step):

```bash
./run_train.sh my-llama --resume
```

---

## 🧪 Step 6: Test the Model via Model Runner

Once you have trained the model or want to test an intermediate checkpoint, you can interact with it using the visual model runner application:

1. Launch the model runner app from the terminal:
```bash
python model_runner/app.py
```
2. In the "Checkpoint path", select the latest checkpoint file:
   `models/my-llama/checkpoint_latest.pt`
3. Click on **Load** to load the model onto the GPU.
4. Enter the start of a story in the "Messages" input field at the bottom of the page and click **Send**. 
5. Watch the model attempt to complete the story in the chat window above.
6. When finished, click **Unload** to release the GPU memory, then press `Ctrl+C` in your terminal to shut down the model runner.

---

## 🎯 Step 7: Fine-tune the Model on the Instruct Dataset

After pre-training your model to generate stories, you can fine-tune it on instructions to follow prompts (e.g., "Tell me a story about a king"). 

### 1. Download the Raw Dataset
The Hugging Face dataset for instructions must be downloaded as raw, un-tokenized JSON:

```bash
python download_hf.py --dataset tiny-stories-instruct
```

### 2. Preprocess, Transform, and Tokenize the Dataset
Run the unified preprocessing script to convert the raw prompt format to natural language prompts, and pre-tokenize the dataset into memory-efficient binary shards (`.npy` files) in one step:

```bash
python utils/preprocess_tinystories_instruct_data.py \
    --input datasets/tiny-stories-instruct/train.json \
    --output datasets/tiny-stories-instruct/natural-instruction-data \
    --to-natural \
    --to-tokenized
```

### 3. Launch the Supervised Fine-Tuning (SFT) Job
Run the SFT script using the `./run_finetune.sh` background launcher script:

```bash
./run_finetune.sh my-llama-sft \
    --base-model my-llama \
    --data datasets/tiny-stories-instruct/natural-instruction-data \
    --bf16 \
    --compile \
    --batch-size 32 \
    --epochs 1 \
    --lr 5e-5 \
    --log-freq 10 \
    --save-iters 100 \
    --save-limit 1 \
    --eval-freq 500
```

#### New SFT-Specific Flags:
- `--base-model my-llama`: The pre-trained model directory whose weights are loaded to initialize SFT training.
- `--data datasets/tiny-stories-instruct/natural-instruction-data`: Path prefix to the pre-tokenized binary dataset files.
- `--lr 5e-5`: Learning rate for SFT optimization (usually lower than pre-training).

*This will save the fine-tuned checkpoints under `models/my-llama-sft/`.*

### 4. Monitor Fine-Tuning Progress
The SFT job runs in the background using `nohup`. You can monitor training progress, loss, validation results, and VRAM utilization in real-time.

To monitor your fine-tuning run:
```bash
./train_status.sh my-llama-sft
```

### ⚡ Expected Resource Utilization & Speed
* **Memory & VRAM**: The fine-tuning job will consume about **66 GB of VRAM**.
* **Step Speed**: Each step with a batch size of 32 takes about **6 seconds**.
* **Log Interval**: With `--log-freq 10` configured, you will see status updates in the logs every **~60 seconds**.
* **Total Time**: Running SFT for the full 1 epoch will take approximately **5 to 6 days**. However, you do not need to complete the full epoch to get usable instruction-following results; checkpoints from earlier steps will already showcase good alignment.

### 5. Stopping or Resuming the Fine-Tuning Run

* **Stopping the Run**: Like pre-training, SFT training can be safely terminated at any point using the stop script:
```bash
./stop_train.sh my-llama-sft
```

* **Resuming the Run**: To resume fine-tuning from the latest checkpoint, run the launcher with the `--resume` flag:
```bash
./run_finetune.sh my-llama-sft --resume
```

### 6. Test the Fine-Tuned Model

Once the instruction fine-tuning is complete, follow the same steps described in **Step 6** of the pre-training guide to launch the model runner, select and load the `models/my-llama-sft/checkpoint_latest.pt` checkpoint, and test it with a natural language prompt like:
`Tell me a story about a boy.`
