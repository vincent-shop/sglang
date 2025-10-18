<context>
  <li>I will handle committing changes, do not offer to commit branch changes</li>
  <li>Follow error handling patterns in existing code</li>
  <li>Follow existing code patterns and style</li>
  <li>Prefer using clear variable and method naming. Do not write comments.</li>
  <li>Run any tasks from your task list in parallel.</li>
  <li> Never write in flowerly, non-straightforward language.</li>
  <li> When answering a piece of question about code especially in the codebase, trace all upstream and downstream methods / invocations to make sure you fully understand the system - it's easy to get to a conclusion that is not accurate / doesn't reflect the real state of the system.
  <li> Unless asked otherwise always cite and reference the source of your answers from your code as well as whatever tools you invoked.
</context>

<async-exec-pattern>
  <li>When the user asks to "exec this to an agent" or delegate a small task asynchronously:</li>
  <li>Use `codex exec --model gpt-5-codex -c model_reasoning_effort=medium --full-auto "task description" &` to run in the background</li>
  <li>Capture the session ID or use `--last` to resume later</li>
  <li>Check progress after X time with `codex exec resume --last "check status"` or poll the output file</li>
  <li>Example workflow:
    ```bash
    # Start async task with optimal settings
    codex exec --model gpt-5-codex -c model_reasoning_effort=medium --full-auto "analyze test coverage and write report" > /tmp/task_output.txt 2>&1 &
    TASK_PID=$!

    # Check if still running
    ps -p $TASK_PID

    # Or check output
    tail -f /tmp/task_output.txt

    # Resume/follow-up later
    codex exec --model gpt-5-codex resume --last "summarize your findings"
    ```
  </li>
  <li>Use `--json` mode for programmatic monitoring of long-running tasks</li>
  <li>Prefer `--output-schema` with `-o output.json` for structured results that need processing</li>
</async-exec-pattern>

