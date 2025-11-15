---
name: document-existing-system
description: Analyze and document an existing codebase following CS161 documentation standards for comprehensive system understanding.
---

# Instructions

Given a list of file paths, create comprehensive documentation for the existing system:

- **Read and analyze** all provided files plus any additional files they import/reference to understand the complete system architecture, tracing execution paths and identifying how components interact

- **Document the system** by creating sections for: Introduction (major subsystems overview), Architecture Overview (APIs, data structures, synchronization), and detailed Component Analysis (key functions, algorithms, edge cases) for each subsystem

- **Focus on clarity** by explaining not just what the code does but why it works that way, including concrete code examples, calling relationships, and any complex behaviors or potential issues you discover

## Checklist

- [ ] All provided files have been read and analyzed
- [ ] Import dependencies and references have been traced
- [ ] Major subsystems have been identified
- [ ] Architecture overview includes APIs and data structures
- [ ] Component analysis covers key functions and algorithms
- [ ] Edge cases and potential issues are documented
- [ ] Code examples illustrate complex behaviors
- [ ] Documentation follows CS161 structure guidelines

## Output Structure

The documentation should follow this structure:

1. **Introduction**: System decomposition and major components
2. **Overview**: Architecture, APIs, data structures, algorithms, synchronization
3. **Topics**: Detailed subsystem analysis with:
   - Key files and their purposes
   - Function documentation with call relationships
   - Error handling and edge cases
   - Code excerpts demonstrating behavior

## Tips

- Start by reading all files to get the full picture before documenting
- Use parallel file reads for efficiency
- Trace execution paths for complete understanding
- Include simplified pseudocode for complex algorithms
- Document both the "what" and the "why" of the code
