import json
import random
import os
import asyncio
import time
import io
import contextlib
import sys
from datetime import datetime
from pathlib import Path

# Import the Agent
from agent import KQAProAgent

# --- Configuration ---
VALID_FILE_PATH = "valid.json"
RESULTS_BASE_DIR = "batch_results"


def load_data(filepath):
    """Loads the valid.json dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_next_batch_dir(base_dir):
    """Creates and returns the next available batch directory (e.g., batch_001, batch_002)."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return os.path.join(base_dir, "batch_001")

    existing_batches = [d for d in os.listdir(base_dir) if os.path.isdir(
        os.path.join(base_dir, d)) and d.startswith("batch_")]

    if not existing_batches:
        new_batch_name = "batch_001"
    else:
        # Extract numbers, find max, increment
        batch_nums = []
        for d in existing_batches:
            try:
                num = int(d.split('_')[1])
                batch_nums.append(num)
            except (IndexError, ValueError):
                continue

        next_num = max(batch_nums) + 1 if batch_nums else 1
        new_batch_name = f"batch_{next_num:03d}"

    full_path = os.path.join(base_dir, new_batch_name)
    os.makedirs(full_path, exist_ok=True)
    return full_path


async def run_single_benchmark(agent: KQAProAgent, item: dict, batch_dir: str, index: int):
    """Runs a single question benchmark and saves the result."""
    question = item.get('question', '')
    golden_answer = item.get('answer', '')
    sparql_query = item.get('sparql', '')

    print(f"\n--- Running Question {index + 1} ---")
    print(f"Q: {question}")

    # Capture Console Output
    # We use io.StringIO to capture stdout (print statements)
    output_capture = io.StringIO()

    start_time = time.time()
    predicted_answer = ""
    error_msg = None

    try:
        # Redirect stdout to capture buffer
        with contextlib.redirect_stdout(output_capture):
            predicted_answer = await agent.ask(question)
    except Exception as e:
        error_msg = str(e)
        # Capture the error in the logs too if needed
        output_capture.write(f"\n[ERROR ENCOUNTERED]: {e}\n")
        print(f"Error processing question: {e}")
    finally:
        end_time = time.time()

    duration = end_time - start_time
    console_logs = output_capture.getvalue()

    # Construct Result Object
    result_data = {
        "metadata": {
            "question_index": index,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(duration, 4),
            "total_tokens_used": agent.token_usage.get("total_tokens", 0),
            "prompt_tokens": agent.token_usage.get("prompt_tokens", 0),
            "completion_tokens": agent.token_usage.get("completion_tokens", 0),
            "status": "success" if not error_msg else "error",
            "error_message": error_msg
        },
        "input": {
            "question": question,
            "golden_answer": golden_answer,
            "golden_sparql": sparql_query,
            "choices": item.get('choices', []),
            "program": item.get('program', [])
        },
        "output": {
            "predicted_answer": predicted_answer,
            "conversation_history": agent._messages
        },
        "console_logs": console_logs
    }

    # Write to file
    filename = f"question_{index:03d}.json"
    file_path = os.path.join(batch_dir, filename)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(result_data, f, indent=4, ensure_ascii=False)

    print(f"Saved result to {file_path}")


async def main():
    # 1. Configuration Inputs
    try:
        n_questions = int(input("How many questions to benchmark? (default 5): ") or 5)
        seed_val = input("Enter random seed (default 42): ")
        seed_val = int(seed_val) if seed_val else 42
    except ValueError:
        print("Invalid input. Using defaults.")
        n_questions = 5
        seed_val = 42

    print(f"Starting Benchmark: N={n_questions}, Seed={seed_val}")

    # 2. Load and Sample Data
    try:
        data = load_data(VALID_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: Could not find {VALID_FILE_PATH}. Please ensure it is in the same directory.")
        return

    random.seed(seed_val)
    if len(data) < n_questions:
        print(f"Warning: Requested {n_questions} questions, but file only has {len(data)}. Using all.")
        selected_items = data
    else:
        selected_items = random.sample(data, n_questions)

    # 3. Create Batch Directory
    batch_dir = get_next_batch_dir(RESULTS_BASE_DIR)
    print(f"Results will be saved in: {batch_dir}")

    # 4. Save Batch Metadata
    batch_meta = {
        "seed": seed_val,
        "n_requested": n_questions,
        "n_actual": len(selected_items),
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(batch_dir, "batch_metadata.json"), 'w') as f:
        json.dump(batch_meta, f, indent=4)

    # 5. Run Benchmark Loop
    for i, item in enumerate(selected_items):
        # Instantiate a fresh agent for each question to ensure clean state
        agent = KQAProAgent(name=f"BenchAgent_{i}", session_id=f"bench_{i}")

        try:
            await run_single_benchmark(agent, item, batch_dir, i)
        except Exception as e:
            print(f"Critical error running benchmark for item {i}: {e}")
        finally:
            # Ensure resources (like MCP client) are cleaned up
            if agent.mcp:
                await agent.mcp.close()

    print(f"\nBatch processing complete. Check folder: {batch_dir}")

if __name__ == "__main__":
    asyncio.run(main())
