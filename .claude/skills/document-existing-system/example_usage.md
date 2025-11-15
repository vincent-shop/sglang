# Example Usage of Document Existing System Skill

## Basic Usage

To use this skill, provide Claude with a list of file paths and ask it to document the system:

```
"Use the 'Document Existing System' skill to analyze and document the following files:
- python/sglang/srt/layers/attention/triton_ops/decode_attention.py
- python/sglang/srt/layers/attention/fla/fused_recurrent.py
- python/sglang/srt/layers/attention/mamba/ops/ssd_state_passing.py

Follow the CS161 documentation structure and trace all dependencies."
```

## Advanced Usage with Specific Focus

```
"Use the 'Document Existing System' skill to document the attention mechanism subsystem:
- Start with files in python/sglang/srt/layers/attention/
- Focus on the triton_ops implementation
- Pay special attention to memory management and optimization techniques
- Include performance implications in your analysis"
```

## Output Example Structure

The skill will produce documentation like:

```markdown
# Attention Mechanism Documentation

## 1. Introduction

The attention mechanism subsystem consists of three major components:
- Triton-based operations for GPU acceleration
- FLA (Fused Linear Attention) implementations
- Mamba state-passing operations

## 2. Architecture Overview

### APIs and Interfaces
[Details about module interfaces]

### Core Data Structures
[Key data structures with purposes]

### Synchronization Mechanisms
[How concurrent operations are handled]

## 3. Component Analysis

### 3.1 Triton Operations (decode_attention.py)
[Detailed analysis with code examples]

### 3.2 FLA Implementation
[Function-by-function breakdown]

[etc...]
```

## Tips for Best Results

1. **Provide context**: Tell Claude what aspect of the system you're most interested in
2. **List core files**: Start with the most important files, Claude will trace dependencies
3. **Request specific analysis**: Ask for performance implications, security considerations, etc.
4. **Iterative refinement**: After initial documentation, ask for deeper analysis of specific areas
