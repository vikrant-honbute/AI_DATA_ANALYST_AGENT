# Prompt Templates

This directory contains reusable prompt templates used by graph nodes.

## Why This Exists

Moving prompts out of node code makes the project easier to maintain:

- Prompt edits do not require touching business logic.
- Prompt versions can be reviewed independently in pull requests.
- Team members can iterate prompts without changing tool execution code.

## Files

- router_prompt.txt: Routes a user query to one source (postgres, csv, mongo).
- planner_prompt.txt: Generates strict JSON plan steps with schema and memory context.
- critic_prompt.txt: Validates execution and emits actionable corrected steps.
- insight_prompt.txt: Produces final analytical insights from final_result.
- loader.py: Shared prompt loading/rendering helpers.

## Rendering

Templates are rendered with Python string.Template.
Variables use the $variable_name syntax.

Current variables by template:

- router_prompt.txt:
  - $query
  - $has_uploaded_csv
- planner_prompt.txt:
  - $query
  - $routed_data_source
  - $schema_text
  - $csv_columns_text
  - $memory_context_text
  - $use_memory_context
  - $format_instructions
- critic_prompt.txt:
  - $data_source
  - $csv_columns
  - $serialized_plan
  - $serialized_results
  - $format_instructions
- insight_prompt.txt:
  - $final_result
