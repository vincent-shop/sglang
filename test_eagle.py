#!/usr/bin/env python3
"""
Test script for EAGLE speculative decoding in sglang.
Tests the WIP eagle implementation with various configurations.

Usage:
    python test_eagle.py
    python test_eagle.py --target-model meta-llama/Llama-2-7b-hf
    python test_eagle.py --page-size 16 --topk 8
"""

import argparse
import json
import os
import subprocess
import sys
import time
import requests
from typing import Dict, List, Optional, Tuple
import signal

# Default models for testing EAGLE3
DEFAULT_TARGET_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"

# Test prompts
TEST_PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a short story about a robot learning to love.",
    "List 5 benefits of regular exercise.",
]


class SGLangServer:
    """Helper class to manage sglang server lifecycle"""
    
    def __init__(self, port: int = 30000):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
        
    def start(self, args: List[str]):
        """Start the sglang server with given arguments"""
        cmd = [sys.executable, "-m", "sglang.launch_server"] + args
        print(f"Starting server with command: {' '.join(cmd)}")
        
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # Create new process group
        )
        
        # Wait for server to be ready
        for _ in range(600):
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    print("Server is ready!")
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(1)
            
        raise RuntimeError("Server failed to start within 600 seconds")
        
    def stop(self):
        """Stop the sglang server"""
        if self.process:
            print("Stopping server...")
            # Kill the entire process group
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait()
            self.process = None
            print("Server stopped")
            

def send_request(base_url: str, prompt: str, max_new_tokens: int = 100, 
                 temperature: float = 0.0) -> Dict:
    """Send a generation request to the server"""
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        "stream": False,
    }
    
    response = requests.post(f"{base_url}/generate", json=payload)
    if response.status_code != 200:
        raise RuntimeError(f"Request failed: {response.text}")
        
    return response.json()


def run_performance_test(server: SGLangServer, prompts: List[str], 
                        max_new_tokens: int = 100) -> Tuple[float, float]:
    """Run performance test and return average speed and acceptance length"""
    total_speed = 0
    total_acc_length = 0
    num_requests = len(prompts)
    
    for prompt in prompts:
        result = send_request(server.base_url, prompt, max_new_tokens)
        
        meta_info = result["meta_info"]
        latency = meta_info["e2e_latency"]
        completion_tokens = meta_info["completion_tokens"]
        
        speed = completion_tokens / latency
        
        # Calculate acceptance length for EAGLE
        if "spec_verify_ct" in meta_info:
            acc_length = completion_tokens / meta_info["spec_verify_ct"]
        else:
            acc_length = 1.0
            
        total_speed += speed
        total_acc_length += acc_length
        
        print(f"  Prompt: {prompt[:50]}...")
        print(f"  Speed: {speed:.2f} tokens/s, Acc Length: {acc_length:.2f}")
        
    avg_speed = total_speed / num_requests
    avg_acc_length = total_acc_length / num_requests
    
    return avg_speed, avg_acc_length


def test_eagle_basic(args):
    """Test basic EAGLE3 functionality"""
    print("\n" + "="*60)
    print("TEST 1: Basic EAGLE3 Functionality")
    print("="*60)
    
    server = SGLangServer(port=args.port)
    
    server_args = [
        "--model-path", args.target_model,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", args.draft_model,
        "--speculative-num-steps", str(args.num_steps),
        "--speculative-eagle-topk", str(args.topk),
        "--speculative-num-draft-tokens", str(args.num_draft_tokens),
        "--port", str(args.port),
        "--dtype", "float16",
        "--log-level", "info",
    ]
    
    if args.enable_beta_spec:
        server_args.append("--enable-beta-spec")
    
    try:
        server.start(server_args)
        
        # Test single request
        result = send_request(server.base_url, TEST_PROMPTS[0])
        print(f"Response: {result['text'][:100]}...")
        
        meta_info = result["meta_info"]
        if "spec_verify_ct" in meta_info:
            print(f"✓ EAGLE is working! Spec verify count: {meta_info['spec_verify_ct']}")
        else:
            print("✗ EAGLE spec_verify_ct not found in response")
            
    finally:
        server.stop()
        

def test_eagle_page_sizes(args):
    """Test EAGLE3 with different page sizes"""
    print("\n" + "="*60)
    print("TEST 2: EAGLE3 with Different Page Sizes")
    print("="*60)
    
    page_sizes = [1, 8, 16, 32] if args.test_all_page_sizes else [args.page_size]
    
    for page_size in page_sizes:
        print(f"\nTesting page_size={page_size}")
        
        server = SGLangServer(port=args.port)
        server_args = [
            "--model-path", args.target_model,
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", args.draft_model,
            "--speculative-num-steps", str(args.num_steps),
            "--speculative-eagle-topk", str(args.topk),
            "--speculative-num-draft-tokens", str(args.num_draft_tokens),
            "--page-size", str(page_size),
            "--port", str(args.port),
            "--dtype", "float16",
            "--log-level", "info",
        ]
        
        if args.enable_beta_spec:
            server_args.append("--enable-beta-spec")
        
        try:
            server.start(server_args)
            
            # Run performance test
            avg_speed, avg_acc_length = run_performance_test(
                server, TEST_PROMPTS[:2], max_new_tokens=50
            )
            
            print(f"  Average speed: {avg_speed:.2f} tokens/s")
            print(f"  Average acceptance length: {avg_acc_length:.2f}")
            
        except Exception as e:
            print(f"  ✗ Failed with page_size={page_size}: {e}")
        finally:
            server.stop()
            

def test_eagle_topk_variations(args):
    """Test EAGLE with different topk values"""
    print("\n" + "="*60)
    print("TEST 3: EAGLE with Different TopK Values")
    print("="*60)
    
    topk_values = [1, 4, 8, 16] if args.test_all_topk else [args.topk]
    
    for topk in topk_values:
        print(f"\nTesting topk={topk}")
        
        server = SGLangServer(port=args.port)
        server_args = [
            "--model-path", args.target_model,
            "--speculative-algorithm", "EAGLE3",
            "--speculative-draft-model-path", args.draft_model,
            "--speculative-num-steps", str(args.num_steps),
            "--speculative-eagle-topk", str(topk),
            "--speculative-num-draft-tokens", str(args.num_draft_tokens),
            "--page-size", str(args.page_size),
            "--port", str(args.port),
            "--dtype", "float16",
            "--log-level", "info",
        ]
        
        if args.enable_beta_spec:
            server_args.append("--enable-beta-spec")
        
        try:
            server.start(server_args)
            
            # Run performance test
            avg_speed, avg_acc_length = run_performance_test(
                server, TEST_PROMPTS[:2], max_new_tokens=50
            )
            
            print(f"  Average speed: {avg_speed:.2f} tokens/s")
            print(f"  Average acceptance length: {avg_acc_length:.2f}")
            
        except Exception as e:
            print(f"  ✗ Failed with topk={topk}: {e}")
        finally:
            server.stop()


def test_eagle_vs_baseline(args):
    """Compare EAGLE performance vs baseline (no speculative decoding)"""
    print("\n" + "="*60)
    print("TEST 4: EAGLE vs Baseline Performance")
    print("="*60)
    
    results = {}
    
    # Test baseline (no speculative decoding)
    print("\nTesting baseline (no EAGLE)...")
    server = SGLangServer(port=args.port)
    server_args = [
        "--model-path", args.target_model,
        "--port", str(args.port),
        "--dtype", "float16",
        "--log-level", "info",
    ]
    
    try:
        server.start(server_args)
        avg_speed, _ = run_performance_test(server, TEST_PROMPTS, max_new_tokens=100)
        results["baseline"] = avg_speed
    finally:
        server.stop()
    
    # Test with EAGLE3
    print("\nTesting with EAGLE3...")
    server = SGLangServer(port=args.port)
    server_args = [
        "--model-path", args.target_model,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", args.draft_model,
        "--speculative-num-steps", str(args.num_steps),
        "--speculative-eagle-topk", str(args.topk),
        "--speculative-num-draft-tokens", str(args.num_draft_tokens),
        "--page-size", str(args.page_size),
        "--port", str(args.port),
        "--dtype", "float16",
        "--log-level", "info",
    ]
    
    if args.enable_beta_spec:
        server_args.append("--enable-beta-spec")
    
    try:
        server.start(server_args)
        avg_speed, avg_acc_length = run_performance_test(
            server, TEST_PROMPTS, max_new_tokens=100
        )
        results["eagle"] = avg_speed
        results["eagle_acc_length"] = avg_acc_length
    finally:
        server.stop()
    
    # Print comparison
    print("\n" + "-"*40)
    print("Performance Comparison:")
    print(f"  Baseline: {results['baseline']:.2f} tokens/s")
    print(f"  EAGLE3: {results['eagle']:.2f} tokens/s")
    print(f"  Speedup: {results['eagle'] / results['baseline']:.2f}x")
    print(f"  EAGLE3 Acceptance Length: {results.get('eagle_acc_length', 0):.2f}")
    

def test_eagle_batch_requests(args):
    """Test EAGLE3 with batch requests"""
    print("\n" + "="*60)
    print("TEST 5: EAGLE3 with Batch Requests")
    print("="*60)
    
    server = SGLangServer(port=args.port)
    server_args = [
        "--model-path", args.target_model,
        "--speculative-algorithm", "EAGLE3",
        "--speculative-draft-model-path", args.draft_model,
        "--speculative-num-steps", str(args.num_steps),
        "--speculative-eagle-topk", str(args.topk),
        "--speculative-num-draft-tokens", str(args.num_draft_tokens),
        "--page-size", str(args.page_size),
        "--port", str(args.port),
        "--dtype", "float16",
        "--log-level", "info",
    ]
    
    if args.enable_beta_spec:
        server_args.append("--enable-beta-spec")
    
    try:
        server.start(server_args)
        
        # Send batch request
        batch_size = 4
        payload = {
            "text": TEST_PROMPTS[:batch_size],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": 50,
            },
            "stream": False,
        }
        
        start_time = time.time()
        response = requests.post(f"{server.base_url}/generate", json=payload)
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            print(f"✓ Batch request successful!")
            print(f"  Batch size: {batch_size}")
            print(f"  Total time: {end_time - start_time:.2f}s")
            
            # Check if all requests have spec_verify_ct
            all_have_spec = all("spec_verify_ct" in r["meta_info"] for r in results)
            if all_have_spec:
                print("  ✓ All requests used EAGLE speculation")
            else:
                print("  ✗ Some requests didn't use EAGLE speculation")
        else:
            print(f"✗ Batch request failed: {response.text}")
            
    finally:
        server.stop()


def main():
    parser = argparse.ArgumentParser(description="Test EAGLE speculative decoding in sglang")
    parser.add_argument("--target-model", default=DEFAULT_TARGET_MODEL,
                        help="Target model path")
    parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL,
                        help="EAGLE draft model path")
    parser.add_argument("--page-size", type=int, default=16,
                        help="Page size for memory allocation")
    parser.add_argument("--topk", type=int, default=1,
                        help="EAGLE top-k value")
    parser.add_argument("--num-steps", type=int, default=3,
                        help="Number of speculative steps")
    parser.add_argument("--num-draft-tokens", type=int, default=4,
                        help="Number of draft tokens")
    parser.add_argument("--port", type=int, default=30000,
                        help="Server port")
    parser.add_argument("--test-all-page-sizes", action="store_true",
                        help="Test all page sizes [1, 8, 16, 32]")
    parser.add_argument("--test-all-topk", action="store_true",
                        help="Test all topk values [1, 4, 8, 16]")
    parser.add_argument("--skip-basic", action="store_true",
                        help="Skip basic functionality test")
    parser.add_argument("--skip-page-sizes", action="store_true",
                        help="Skip page size variation test")
    parser.add_argument("--skip-topk", action="store_true",
                        help="Skip topk variation test")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip baseline comparison test")
    parser.add_argument("--skip-batch", action="store_true",
                        help="Skip batch request test")
    parser.add_argument("--enable-beta-spec", action="store_true",
                        help="Enable beta spec for EAGLE (enables overlap schedule)")
    
    args = parser.parse_args()
    
    print("EAGLE Speculative Decoding Test Suite")
    print(f"Target Model: {args.target_model}")
    print(f"Draft Model: {args.draft_model}")
    print(f"Default Page Size: {args.page_size}")
    print(f"Default TopK: {args.topk}")
    print(f"Default Num Steps: {args.num_steps}")
    print(f"Beta Spec Enabled: {args.enable_beta_spec}")
    
    # Run tests
    try:
        if not args.skip_basic:
            test_eagle_basic(args)
            
        if not args.skip_page_sizes:
            test_eagle_page_sizes(args)
            
        if not args.skip_topk:
            test_eagle_topk_variations(args)
            
        if not args.skip_comparison:
            test_eagle_vs_baseline(args)
            
        if not args.skip_batch:
            test_eagle_batch_requests(args)
            
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
