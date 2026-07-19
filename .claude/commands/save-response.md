---
description: Save your previous response verbatim to a text file
argument-hint: <file-path>
---

Take your immediately preceding assistant response in this conversation (the one before this command was invoked) and write it verbatim, in full, to the file path given as the argument: $1

If no argument was given, ask the user for a file path before writing anything — do not guess one.

Create any missing parent directories. After writing, report back the file path and how many lines were written. Do not summarize or alter the response content — write it exactly as it was said.