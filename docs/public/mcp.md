# Model Context Protocol (MCP) Setup

Version: 1.0.0  
Last Updated: 2026-10-12  
Canonical: http://localhost:8501/docs/mcp.md  

Sorrel provides a remote Model Context Protocol (MCP) server over Streamable HTTP, enabling external AI agents (such as Claude Desktop, ChatGPT, and Codex) to orchestrate data analysis workflows as first-class tools.

---

## 1. Remote Server Configuration

Connect your MCP client to the Sorrel remote endpoint:

```json
{
  "mcpServers": {
    "sorrel": {
      "url": "http://localhost:8501/mcp",
      "headers": {
        "Authorization": "Bearer YOUR_SORREL_API_KEY"
      }
    }
  }
}
```

---

## 2. Exposed Agent Tools

The MCP server exposes narrow, governed tools that wrap the full analysis lifecycle:

| Tool Name | Purpose | Side Effects |
| :--- | :--- | :--- |
| `get_sorrel_capabilities` | Inspect supported file formats, row caps, and active features | Read-only |
| `create_dataset_upload` | Allocate a secure, short-lived presigned upload slot for a file | Creates upload resource |
| `start_data_analysis` | Initiate the 7-stage governed analysis for a `dataset_id` | Enqueues analysis run |
| `get_analysis_status` | Query current stage, percentage progress, and safe event trace | Read-only |
| `get_analysis_result` | Retrieve structured findings, evidence metrics, and caveats | Read-only |
| `list_analysis_artifacts` | Enumerate available reports, charts, and model files for a run | Read-only |
| `cancel_data_analysis` | Cooperatively halt an ongoing analysis run | Modifies run state |

---

## 3. Read-Only MCP Resources

Completed run artifacts and public documentation are accessible as URI resources:

```text
sorrel://docs/overview
sorrel://runs/{run_id}/result
sorrel://runs/{run_id}/report.md
sorrel://runs/{run_id}/dashboard.json
```

---

## 4. Guardrails and Agent Boundaries

To ensure security and analytical integrity, the MCP interface enforces the following boundaries:

- **No Raw Tool Injection**: External agents cannot bypass `AgentController` to invoke internal tools (e.g. `CorrelationAnalysisTool`) or inject arbitrary SQL/Python scripts directly.
- **Asynchronous Run Handles**: Analysis operations return a `run_id` handle immediately. Agents are expected to inspect progress via `get_analysis_status` rather than blocking tool calls.
- **Data Privacy**: Tool outputs contain only aggregated statistics and structured findings. Individual raw rows are never returned to the agent context.
