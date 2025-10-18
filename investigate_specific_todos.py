#!/usr/bin/env python3

import subprocess
import time
from pathlib import Path

# Specific TODOs/FIXMEs from the chat context attachments
SPECIFIC_TODOS = [
    {
        "file": "python/sglang/srt/model_executor/model_runner.py",
        "line": 299,
        "text": "# FIXME: hacky set `use_mla_backend`",
        "type": "FIXME"
    },
    {
        "file": "python/sglang/srt/managers/tp_worker.py",
        "line": 350,
        "text": "# FIXME(lsyin): unify the interface of forward_batch",
        "type": "FIXME"
    },
    {
        "file": "python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py",
        "line": 256,
        "text": "# TODO(yuwei): support return logprob",
        "type": "TODO"
    },
    {
        "file": "python/sglang/srt/managers/scheduler.py",
        "line": 258,
        "text": "# TODO(lsyin): refactor PP and avoid using dict",
        "type": "TODO"
    },
    {
        "file": "python/sglang/srt/managers/scheduler.py",
        "line": 999,
        "text": "# FIXME(lsyin): hacky way to keep a reference to avoid GPU tensors being freed by torch GC",
        "type": "FIXME"
    },
    {
        "file": "python/sglang/srt/managers/scheduler.py",
        "line": 2191,
        "text": "# FIXME: remove this assert",
        "type": "FIXME"
    },
    {
        "file": "python/sglang/srt/managers/scheduler.py",
        "line": 2210,
        "text": "# FIXME(lsyin): maybe move this to forward_batch_generation",
        "type": "FIXME"
    },
    {
        "file": "python/sglang/srt/managers/scheduler.py",
        "line": 2220,
        "text": "# FIXME(lsyin): move this assignment elsewhere",
        "type": "FIXME"
    },
    {
        "file": "python/sglang/srt/managers/scheduler.py",
        "line": 2247,
        "text": "#       which can probably be replaced by future_indices later [TODO(lsyin)].",
        "type": "TODO"
    },
    {
        "file": "python/sglang/srt/speculative/ngram_info.py",
        "line": 202,
        "text": "# TODO: boolean array index leads to a device sync. Remove it.",
        "type": "TODO"
    },
    {
        "file": "python/sglang/srt/speculative/ngram_worker.py",
        "line": 200,
        "text": "# FIXME: Whether to insert 'extend' into the cache or not, after testing,",
        "type": "FIXME"
    },
    {
        "file": "python/sglang/srt/layers/moe/fused_moe_triton/layer.py",
        "line": 830,
        "text": "# TODO: consider using symmetric memory",
        "type": "TODO"
    },
    {
        "file": "python/sglang/srt/entrypoints/context.py",
        "line": 88,
        "text": "# TODO: REMOVE here:",
        "type": "TODO"
    },
    {
        "file": "python/sglang/srt/entrypoints/context.py",
        "line": 203,
        "text": "# TODO: REMOVE here:",
        "type": "TODO"
    },
    {
        "file": "python/sglang/srt/layers/attention/trtllm_mha_backend.py",
        "line": 30,
        "text": "512  # Memory workspace size in MB, todo(Yingyi): read from config",
        "type": "TODO"
    },
    {
        "file": "python/sglang/srt/layers/attention/flashinfer_backend.py",
        "line": 1151,
        "text": "# TODO: remove this device sync, we can use forward_batch.extend_prefix_lens_cpu",
        "type": "TODO"
    }
]

def create_investigation_prompt(todo_item):
    """Create a detailed investigation prompt for a specific TODO/FIXME."""
    
    prompt = f"""Investigate this {todo_item['type']} in the sglang codebase:

File: {todo_item['file']}
Line: {todo_item['line']}
Issue: {todo_item['text']}

Please analyze:
1. Read the code context around line {todo_item['line']} (at least 20 lines before and after)
2. Understand what the current implementation does
3. Identify why this was marked as {todo_item['type']}
4. Propose a concrete solution with code examples
5. Assess the impact of fixing this issue
6. Write a detailed markdown report to: sglang_todo_investigation_{todo_item['line']}_{Path(todo_item['file']).stem}.md

The report should include:
- Current implementation analysis
- Problem description
- Proposed solution with code
- Dependencies and impact analysis
- Testing recommendations
"""
    
    return prompt

def dispatch_todo_investigation(todo_item, index):
    """Dispatch a single TODO investigation using codex."""
    
    prompt = create_investigation_prompt(todo_item)
    
    # Create output filename
    output_file = f"todo_investigation_{index+1}_{Path(todo_item['file']).stem}_line{todo_item['line']}.md"
    
    print(f"\n[{index+1}/{len(SPECIFIC_TODOS)}] Dispatching investigation for:")
    print(f"  File: {todo_item['file']}")
    print(f"  Line: {todo_item['line']}")
    print(f"  Issue: {todo_item['text']}")
    print(f"  Output: {output_file}")
    
    # Run codex exec command with proper syntax
    cmd = [
        "codex", "exec",
        "--full-auto",  # Allow file edits
        "-o", output_file,  # Output to file
        prompt
    ]
    
    # Execute and capture output for verification
    proc = subprocess.Popen(cmd, 
                            stdout=subprocess.PIPE, 
                            stderr=subprocess.PIPE,
                            text=True)
    
    # Check immediate output
    try:
        stdout, stderr = proc.communicate(timeout=2)
        if "ERROR: Missing environment variable: `OPENAI_API_KEY`" in stderr:
            print(f"  ❌ Error: OPENAI_API_KEY not set")
            # Create a placeholder file with the prompt
            with open(output_file, 'w') as f:
                f.write(f"# TODO Investigation (Manual Required)\n\n")
                f.write(f"**File**: {todo_item['file']}\n")
                f.write(f"**Line**: {todo_item['line']}\n")
                f.write(f"**Issue**: {todo_item['text']}\n\n")
                f.write("## Investigation Prompt\n\n")
                f.write(prompt)
                f.write("\n\n---\n")
                f.write("*Note: OPENAI_API_KEY not set. Manual investigation required.*\n")
            return output_file, None
        elif stderr:
            print(f"  Started: {stderr.strip()[:80]}...")
    except subprocess.TimeoutExpired:
        print(f"  Process started (PID: {proc.pid})")
    
    return output_file, proc.pid

def main():
    """Main function to dispatch all specific TODO investigations."""
    
    print(f"Dispatching {len(SPECIFIC_TODOS)} specific TODO/FIXME investigations using codex exec...")
    
    output_files = []
    pids = []
    
    # Dispatch all investigations
    for i, todo in enumerate(SPECIFIC_TODOS):
        output_file, pid = dispatch_todo_investigation(todo, i)
        output_files.append(output_file)
        pids.append(pid)
        time.sleep(1)  # Small delay between dispatches
    
    print(f"\n✓ All {len(SPECIFIC_TODOS)} investigations dispatched!")
    print(f"Active processes: {len([p for p in pids if p])}")
    
    print("\nOutput files will be generated in the current directory:")
    for f in output_files:
        print(f"  - {f}")
    
    print("\nTo check progress:")
    print("  ls -la todo_investigation_*.md")
    print("  ps aux | grep 'codex exec'")
    print("\nTo wait for completion and combine all reports:")
    print("  while ps aux | grep -q '[c]odex exec'; do sleep 5; done")
    print("  cat todo_investigation_*.md > ALL_TODO_INVESTIGATIONS.md")

if __name__ == "__main__":
    main()
