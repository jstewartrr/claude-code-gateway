# Claude Code Gateway - Response-Limited MCP Server

## Overview

This is a response-limited version of the Sovereign Mind MCP Gateway, specifically designed for use with Claude Code on local machines. It prevents memory overload by automatically truncating large responses.

## Key Features

- **Response Size Limiting**: Max 50KB per response (configurable)
- **Row Limiting**: Database queries limited to 100 rows by default
- **Pagination Hints**: Automatically adds hints when data is truncated
- **Memory Safety**: JSON serialization with overflow protection

## Response Limits (Configurable)

| Parameter | Default | Description |
|-----------|---------|-------------|
| MAX_RESPONSE_BYTES | 51200 | Maximum response size in bytes |
| MAX_ROWS | 100 | Maximum rows from database queries |
| MAX_EMAIL_COUNT | 25 | Maximum emails per request |
| MAX_TASK_COUNT | 50 | Maximum Asana tasks per request |
| MAX_FILE_LIST | 100 | Maximum files in directory listings |

## Installation

### Option 1: Local Python (Recommended for Claude Code)

```bash
# Clone or copy files
cd /path/to/claude-code-gateway

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.template .env
# Edit .env with your credentials
```

### Option 2: Docker (For Remote Access)

```bash
# Build image
docker build -t claude-code-gateway:latest .

# Run locally
docker run -p 8000:8000 --env-file .env claude-code-gateway:latest
```

### Option 3: Azure Container Apps

```bash
# Build and push to ACR
az acr build --registry sovereignmindacr \
  --image claude-code-gateway:v1.0.0 \
  --file Dockerfile .

# Deploy to Container Apps
az containerapp update \
  --name claude-code-gateway \
  --resource-group SovereignMind-RG \
  --image sovereignmindacr.azurecr.io/claude-code-gateway:v1.0.0
```

## Claude Code Configuration

Add to your Claude Code MCP configuration (`~/.config/claude-code/mcp.json` or equivalent):

```json
{
  "mcpServers": {
    "sovereign-mind": {
      "command": "python",
      "args": ["/path/to/claude-code-gateway/gateway.py"],
      "env": {
        "SNOWFLAKE_PASSWORD": "your-password",
        "ASANA_TOKEN": "your-token",
        "GITHUB_TOKEN": "your-token",
        "M365_TENANT_ID": "your-tenant",
        "M365_CLIENT_ID": "your-client-id",
        "M365_CLIENT_SECRET": "your-secret"
      }
    }
  }
}
```

## Available Tools

### Gateway
- `gateway_status` - Get current limits and configuration

### Snowflake
- `sm_query_snowflake` - Execute SQL with automatic row limiting

### Hive Mind
- `hivemind_write` - Write to shared memory
- `hivemind_read` - Read from shared memory (limited)

### Asana
- `asana_list_tasks` - List tasks (limited to 50)
- `asana_create_task` - Create a task
- `asana_get_task` - Get task details
- `asana_complete_task` - Mark task complete
- `asana_add_comment` - Add comment to task

### M365 Email
- `m365_read_emails` - Read emails (limited to 25)
- `m365_send_email` - Send email

### GitHub
- `github_list_repos` - List repositories
- `github_get_file` - Get file contents (truncated if large)

## Truncation Behavior

When responses exceed limits, the gateway:

1. Truncates the data to fit within limits
2. Adds `_truncated: true` to the response
3. Includes `_original_count` and `_returned_count`
4. Adds `_message` with pagination hints

Example truncated response:
```json
{
  "success": true,
  "row_count": 100,
  "data": [...],
  "_truncated": true,
  "_original_count": 5000,
  "_returned_count": 100,
  "_message": "Results limited to 100 rows. Add LIMIT/OFFSET to query for pagination."
}
```

## Version History

- **v1.0.0** - Initial release with response limiting
  - Based on SM-MCP-Gateway v1.8.0
  - Added ResponseLimits configuration
  - Added truncate_response utility
  - Snowflake, Asana, M365, GitHub tools

## Support

For issues, contact the Sovereign Mind team or raise an issue in the repository.
