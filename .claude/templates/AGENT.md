# Agent Template

All agents use `<Agent_Prompt>` XML structure.

```yaml
---
name: <agent-name>
description: "<description>. Use when [trigger]. NOT for [exclusion]."
model: <opus|sonnet>
---
```

```xml
<Agent_Prompt>
  <Role>
    You are [role]. Your mission is [mission].
    You are responsible for: [responsibilities].
    You are NOT responsible for: [exclusions with agent names].

    | Situation | Priority |
    |-----------|----------|
    | [trigger condition] | MANDATORY / RECOMMENDED / OPTIONAL |
  </Role>
  <Why_This_Matters>
    [What fails without this agent. Why manual/naive approach breaks.]
  </Why_This_Matters>
  <Success_Criteria>
    - [Measurable criterion 1]
    - [Measurable criterion 2]
  </Success_Criteria>
  <Constraints>
    [ONE-LINE IRON LAW IN CAPS]

    | DO | DON'T |
    |----|-------|
    | [correct behavior] | [incorrect behavior] |
  </Constraints>
  <Investigation_Protocol>
    1) [Step with sub-steps a, b, c]
    2) [Step]
  </Investigation_Protocol>
  <Tool_Usage>
    [Which tools and why. MCP tool names if delegating.]
  </Tool_Usage>
  <Output_Format>
    ## Report Title
    ### Section
    | Column | Column |
    |--------|--------|
  </Output_Format>
  <Failure_Modes_To_Avoid>
    - [Mode]: [What goes wrong]. Instead: [correction].
  </Failure_Modes_To_Avoid>
</Agent_Prompt>
```

### Required Sections

| Section | Description |
|---------|-------------|
| `Role` | Core responsibility + explicit NOT-responsible boundaries + When to Invoke table |
| `Success_Criteria` | Measurable completion criteria |
| `Constraints` | Iron law + DO/DON'T table |
| `Investigation_Protocol` or `Protocol` | Numbered execution steps |
| `Output_Format` | Structured output template with tables |
| `Failure_Modes_To_Avoid` | Common mistakes with "Instead:" corrections |

### Optional Sections

| Section | When to Include |
|---------|-----------------|
| `Why_This_Matters` | Tier 0-1 required |
| `Tool_Usage` | Agent uses specific tools |
| `Examples` | Good/Bad execution pairs |
| `Final_Checklist` | Pre-completion self-check |
