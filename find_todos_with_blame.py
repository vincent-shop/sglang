#!/usr/bin/env python3

import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def get_git_blame_date(file_path, line_number):
    """Get the commit date for a specific line using git blame."""
    try:
        cmd = ['git', 'blame', '-L', f'{line_number},{line_number}', '--porcelain', file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Parse porcelain output to get commit timestamp
        lines = result.stdout.strip().split('\n')
        for line in lines:
            if line.startswith('author-time '):
                timestamp = int(line.split()[1])
                date = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
                return date
                
    except subprocess.CalledProcessError:
        pass
    
    return None


def find_todos_and_fixmes(root_dir='.'):
    """Find all TODO and FIXME comments in code files."""
    todo_pattern = re.compile(r'(?i)\b(TODO|FIXME)\b.*', re.IGNORECASE)
    
    # Common code file extensions
    code_extensions = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.cpp', '.cc', '.h', '.hpp',
        '.cs', '.rb', '.go', '.rs', '.php', '.swift', '.kt', '.scala', '.r', '.m', '.mm',
        '.sh', '.bash', '.zsh', '.ps1', '.bat', '.cmd', '.lua', '.pl', '.pm', '.dart'
    }
    
    results_by_date = defaultdict(list)
    
    for root, dirs, files in os.walk(root_dir):
        # Skip hidden directories and common non-code directories
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'dist', 'build', 'target']]
        
        for file in files:
            file_path = Path(root) / file
            
            # Check if it's a code file
            if file_path.suffix not in code_extensions:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        match = todo_pattern.search(line)
                        if match:
                            # Get git blame date
                            blame_date = get_git_blame_date(str(file_path), line_num)
                            
                            if blame_date:
                                todo_info = {
                                    'file': str(file_path),
                                    'line': line_num,
                                    'text': line.strip(),
                                    'type': match.group(1).upper()
                                }
                                results_by_date[blame_date].append(todo_info)
                                
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return dict(results_by_date)


def main():
    """Main function to find TODOs/FIXMEs and output as JSON."""
    print("Searching for TODO and FIXME comments...")
    
    # Check if we're in a git repository
    try:
        subprocess.run(['git', 'status'], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: This script must be run in a git repository.")
        return
    
    # Find all TODOs and FIXMEs
    results = find_todos_and_fixmes()
    
    # Sort dates
    sorted_results = dict(sorted(results.items()))
    
    # Output as JSON
    output_json = json.dumps(sorted_results, indent=2)
    print("\nResults:")
    print(output_json)
    
    # Also save to file
    output_file = 'todos_by_date.json'
    with open(output_file, 'w') as f:
        f.write(output_json)
    
    print(f"\nResults also saved to: {output_file}")
    
    # Print summary
    total_items = sum(len(items) for items in results.values())
    print(f"\nSummary:")
    print(f"- Total TODO/FIXME items: {total_items}")
    print(f"- Unique dates: {len(results)}")
    
    if results:
        print(f"- Oldest: {min(results.keys())}")
        print(f"- Newest: {max(results.keys())}")


if __name__ == '__main__':
    main()
