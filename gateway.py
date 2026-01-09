"""
Sovereign Mind MCP Gateway - Claude Code Edition v1.0.0
========================================================
Optimized gateway for Claude Code with response size limiting to prevent
memory overload on local machines.

Key Features:
- Response truncation at configurable limits (default 50KB)
- Row limiting for database queries (default 100 rows)
- Pagination hints for large datasets
- Memory-safe JSON serialization

Based on SM-MCP-Gateway v1.8.0
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime

import httpx
import uvicorn
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("claude_code_gateway")

# ============================================================================
# RESPONSE LIMITING CONFIGURATION
# ============================================================================

class ResponseLimits:
    """Response size limits for Claude Code safety."""
    MAX_RESPONSE_BYTES = int(os.getenv("MAX_RESPONSE_BYTES", "51200"))  # 50KB default
    MAX_ROWS = int(os.getenv("MAX_ROWS", "100"))  # Max rows from DB queries
    MAX_EMAIL_COUNT = int(os.getenv("MAX_EMAIL_COUNT", "25"))
    MAX_TASK_COUNT = int(os.getenv("MAX_TASK_COUNT", "50"))
    MAX_FILE_LIST = int(os.getenv("MAX_FILE_LIST", "100"))
    TRUNCATION_WARNING = True

limits = ResponseLimits()

def truncate_response(data: Any, max_bytes: int = None) -> tuple[Any, bool]:
    """
    Truncate response data to fit within size limits.
    Returns (truncated_data, was_truncated).
    """
    max_bytes = max_bytes or limits.MAX_RESPONSE_BYTES
    
    json_str = json.dumps(data, default=str)
    
    if len(json_str.encode('utf-8')) <= max_bytes:
        return data, False
    
    # Data exceeds limit - need to truncate
    if isinstance(data, dict):
        if "data" in data and isinstance(data["data"], list):
            # Truncate the data array
            original_count = len(data["data"])
            data["data"] = data["data"][:limits.MAX_ROWS]
            data["_truncated"] = True
            data["_original_count"] = original_count
            data["_returned_count"] = len(data["data"])
            data["_message"] = f"Response truncated to {limits.MAX_ROWS} rows. Use LIMIT/OFFSET for pagination."
            return data, True
        elif "emails" in data and isinstance(data["emails"], list):
            original_count = len(data["emails"])
            data["emails"] = data["emails"][:limits.MAX_EMAIL_COUNT]
            data["_truncated"] = True
            data["_original_count"] = original_count
            data["_returned_count"] = len(data["emails"])
            return data, True
        elif "tasks" in data and isinstance(data["tasks"], list):
            original_count = len(data["tasks"])
            data["tasks"] = data["tasks"][:limits.MAX_TASK_COUNT]
            data["_truncated"] = True
            data["_original_count"] = original_count
            data["_returned_count"] = len(data["tasks"])
            return data, True
    elif isinstance(data, list):
        original_count = len(data)
        data = data[:limits.MAX_ROWS]
        return {
            "data": data,
            "_truncated": True,
            "_original_count": original_count,
            "_returned_count": len(data)
        }, True
    elif isinstance(data, str) and len(data.encode('utf-8')) > max_bytes:
        # Truncate string response
        truncated = data[:max_bytes - 100]  # Leave room for truncation message
        return f"{truncated}\n\n[RESPONSE TRUNCATED - exceeded {max_bytes} bytes]", True
    
    return data, False

def safe_json_response(data: Any, indent: int = 2) -> str:
    """Create a JSON response with automatic truncation."""
    truncated_data, was_truncated = truncate_response(data)
    return json.dumps(truncated_data, indent=indent, default=str)

# ============================================================================
# CONFIGURATION
# ============================================================================

class GatewayConfig:
    """Central configuration loaded from environment variables."""
    
    # Snowflake
    SNOWFLAKE_ACCOUNT = os.getenv("SNOWFLAKE_ACCOUNT", "jga82554.east-us-2.azure")
    SNOWFLAKE_USER = os.getenv("SNOWFLAKE_USER", "JOHN_CLAUDE")
    SNOWFLAKE_PASSWORD = os.getenv("SNOWFLAKE_PASSWORD", "")
    SNOWFLAKE_WAREHOUSE = os.getenv("SNOWFLAKE_WAREHOUSE", "SOVEREIGN_MIND_WH")
    SNOWFLAKE_DATABASE = os.getenv("SNOWFLAKE_DATABASE", "SOVEREIGN_MIND")
    SNOWFLAKE_ROLE = os.getenv("SNOWFLAKE_ROLE", "ACCOUNTADMIN")
    
    # Asana
    ASANA_ACCESS_TOKEN = os.getenv("ASANA_TOKEN") or os.getenv("ASANA_ACCESS_TOKEN", "")
    ASANA_WORKSPACE_ID = os.getenv("ASANA_WORKSPACE_ID", "373563495855656")
    
    # Make.com
    MAKE_API_KEY = os.getenv("MAKE_API_KEY", "")
    MAKE_ORGANIZATION_ID = os.getenv("MAKE_ORGANIZATION_ID", "5726294")
    MAKE_TEAM_ID = os.getenv("MAKE_TEAM_ID", "1576120")
    
    # GitHub
    GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
    
    # ElevenLabs
    ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    
    # Google Cloud
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    
    # M365 / Microsoft Graph
    M365_TENANT_ID = os.getenv("M365_TENANT_ID", "")
    M365_CLIENT_ID = os.getenv("M365_CLIENT_ID", "")
    M365_CLIENT_SECRET = os.getenv("M365_CLIENT_SECRET", "")
    M365_DEFAULT_USER = os.getenv("M365_DEFAULT_USER", "john@middlegroundcapital.com")

config = GatewayConfig()

# ============================================================================
# SHARED UTILITIES
# ============================================================================

async def make_api_request(
    method: str,
    url: str,
    headers: Dict[str, str],
    json_data: Optional[Dict] = None,
    params: Optional[Dict] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Generic async HTTP request handler with error handling."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=json_data,
                params=params
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            return {
                "error": True,
                "status_code": e.response.status_code,
                "message": f"HTTP {e.response.status_code}: {e.response.text[:500]}"
            }
        except httpx.TimeoutException:
            return {"error": True, "message": "Request timed out"}
        except Exception as e:
            return {"error": True, "message": str(e)}

# ============================================================================
# M365 TOKEN MANAGEMENT
# ============================================================================

_m365_token_cache = {"token": None, "expires_at": 0}

async def get_m365_token() -> str:
    """Get M365 access token using client credentials flow."""
    import time
    
    if _m365_token_cache["token"] and time.time() < _m365_token_cache["expires_at"] - 60:
        return _m365_token_cache["token"]
    
    token_url = f"https://login.microsoftonline.com/{config.M365_TENANT_ID}/oauth2/v2.0/token"
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            token_url,
            data={
                "client_id": config.M365_CLIENT_ID,
                "client_secret": config.M365_CLIENT_SECRET,
                "scope": "https://graph.microsoft.com/.default",
                "grant_type": "client_credentials"
            }
        )
        response.raise_for_status()
        data = response.json()
        
        _m365_token_cache["token"] = data["access_token"]
        _m365_token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
        
        return data["access_token"]

async def m365_graph_request(
    method: str,
    endpoint: str,
    json_data: Optional[Dict] = None,
    params: Optional[Dict] = None,
    timeout: float = 30.0
) -> Dict[str, Any]:
    """Make authenticated request to Microsoft Graph API."""
    try:
        token = await get_m365_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        url = f"https://graph.microsoft.com/v1.0{endpoint}"
        
        return await make_api_request(
            method=method,
            url=url,
            headers=headers,
            json_data=json_data,
            params=params,
            timeout=timeout
        )
    except Exception as e:
        return {"error": True, "message": str(e)}

# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def gateway_lifespan(app):
    """Initialize shared resources for the gateway."""
    logger.info(f"Claude Code Gateway starting - Response limit: {limits.MAX_RESPONSE_BYTES} bytes, Max rows: {limits.MAX_ROWS}")
    
    snowflake_conn = None
    try:
        import snowflake.connector
        snowflake_conn = snowflake.connector.connect(
            user=config.SNOWFLAKE_USER,
            password=config.SNOWFLAKE_PASSWORD,
            account=config.SNOWFLAKE_ACCOUNT,
            warehouse=config.SNOWFLAKE_WAREHOUSE,
            database=config.SNOWFLAKE_DATABASE,
            role=config.SNOWFLAKE_ROLE
        )
        logger.info("Snowflake connection established")
    except Exception as e:
        logger.warning(f"Snowflake connection failed: {e}")
    
    yield {"snowflake_conn": snowflake_conn}
    
    if snowflake_conn:
        snowflake_conn.close()
        logger.info("Snowflake connection closed")

# ============================================================================
# INITIALIZE MCP SERVER
# ============================================================================

mcp = FastMCP(
    "claude_code_gateway",
    lifespan=gateway_lifespan
)

# ============================================================================
# GATEWAY STATUS TOOL
# ============================================================================

@mcp.tool(
    name="gateway_status",
    annotations={
        "title": "[GATEWAY] Get gateway status and limits",
        "readOnlyHint": True
    }
)
async def gateway_status() -> str:
    """Get the current gateway configuration and response limits."""
    return json.dumps({
        "gateway": "Claude Code Edition",
        "version": "1.0.0",
        "limits": {
            "max_response_bytes": limits.MAX_RESPONSE_BYTES,
            "max_rows": limits.MAX_ROWS,
            "max_email_count": limits.MAX_EMAIL_COUNT,
            "max_task_count": limits.MAX_TASK_COUNT,
            "max_file_list": limits.MAX_FILE_LIST
        },
        "status": "healthy"
    }, indent=2)

# ============================================================================
# SNOWFLAKE TOOLS (WITH ROW LIMITING)
# ============================================================================

class SnowflakeQueryInput(BaseModel):
    """Input for Snowflake SQL queries."""
    model_config = ConfigDict(str_strip_whitespace=True)
    
    sql: str = Field(
        ..., 
        description="SQL query to execute against Snowflake",
        min_length=1,
        max_length=50000
    )
    database: Optional[str] = Field(
        default=None,
        description="Database to use (defaults to SOVEREIGN_MIND)"
    )
    max_rows: Optional[int] = Field(
        default=None,
        description=f"Maximum rows to return (default: {ResponseLimits.MAX_ROWS})",
        ge=1,
        le=500
    )

@mcp.tool(
    name="sm_query_snowflake",
    annotations={
        "title": "[SM] Execute SQL query on Snowflake (row-limited)",
        "readOnlyHint": False
    }
)
async def snowflake_query(params: SnowflakeQueryInput) -> str:
    """Execute a SQL query against Snowflake with automatic row limiting.
    
    Results are automatically truncated to prevent memory overload.
    Use LIMIT and OFFSET in your query for pagination of large datasets.
    """
    max_rows = params.max_rows or limits.MAX_ROWS
    
    try:
        import snowflake.connector
        
        conn = snowflake.connector.connect(
            user=config.SNOWFLAKE_USER,
            password=config.SNOWFLAKE_PASSWORD,
            account=config.SNOWFLAKE_ACCOUNT,
            warehouse=config.SNOWFLAKE_WAREHOUSE,
            database=params.database or config.SNOWFLAKE_DATABASE,
            role=config.SNOWFLAKE_ROLE
        )
        
        cursor = conn.cursor()
        cursor.execute(params.sql)
        
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        
        # Fetch with limit + 1 to detect if there are more rows
        rows = cursor.fetchmany(max_rows + 1)
        has_more = len(rows) > max_rows
        rows = rows[:max_rows]
        
        results = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                if hasattr(value, 'isoformat'):
                    value = value.isoformat()
                elif isinstance(value, bytes):
                    value = value.decode('utf-8', errors='replace')
                row_dict[col] = value
            results.append(row_dict)
        
        cursor.close()
        conn.close()
        
        response = {
            "success": True,
            "row_count": len(results),
            "data": results
        }
        
        if has_more:
            response["_has_more"] = True
            response["_message"] = f"Results limited to {max_rows} rows. Add LIMIT/OFFSET to query for pagination."
        
        return safe_json_response(response)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

# ============================================================================
# HIVE MIND TOOLS
# ============================================================================

class HiveMindWriteInput(BaseModel):
    """Input for writing to Hive Mind."""
    source: str = Field(..., description="Source identifier (e.g., CLAUDE_CODE)")
    category: str = Field(..., description="Category: CONTEXT, DECISION, ACTION_ITEM, etc")
    summary: str = Field(..., description="Clear summary (max 2000 chars)", max_length=2000)
    workstream: str = Field(default="GENERAL", description="Workstream or project name")
    priority: str = Field(default="MEDIUM", description="HIGH, MEDIUM, or LOW")
    details: Optional[Dict] = Field(default=None, description="JSON details object")
    tags: Optional[List[str]] = Field(default=None, description="List of tags")

@mcp.tool(
    name="hivemind_write",
    annotations={
        "title": "[GATEWAY] Write to Hive Mind shared memory",
        "readOnlyHint": False
    }
)
async def hivemind_write(params: HiveMindWriteInput) -> str:
    """Write an entry to the Sovereign Mind Hive Mind for cross-AI continuity."""
    try:
        import snowflake.connector
        
        conn = snowflake.connector.connect(
            user=config.SNOWFLAKE_USER,
            password=config.SNOWFLAKE_PASSWORD,
            account=config.SNOWFLAKE_ACCOUNT,
            warehouse=config.SNOWFLAKE_WAREHOUSE,
            database=config.SNOWFLAKE_DATABASE,
            role=config.SNOWFLAKE_ROLE
        )
        
        details_json = json.dumps(params.details) if params.details else None
        tags_json = json.dumps(params.tags) if params.tags else None
        
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO SOVEREIGN_MIND.RAW.HIVE_MIND 
            (SOURCE, CATEGORY, WORKSTREAM, SUMMARY, DETAILS, PRIORITY, TAGS, STATUS)
            VALUES (%s, %s, %s, %s, PARSE_JSON(%s), %s, PARSE_JSON(%s), 'ACTIVE')
        """, (
            params.source,
            params.category,
            params.workstream,
            params.summary,
            details_json,
            params.priority,
            tags_json
        ))
        
        cursor.close()
        conn.close()
        
        return json.dumps({
            "success": True,
            "message": "Entry written to Hive Mind",
            "category": params.category,
            "workstream": params.workstream
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

class HiveMindReadInput(BaseModel):
    """Input for reading from Hive Mind."""
    limit: int = Field(default=10, ge=1, le=50, description="Max entries to return")
    category: Optional[str] = Field(default=None, description="Filter by category")
    workstream: Optional[str] = Field(default=None, description="Filter by workstream")
    source: Optional[str] = Field(default=None, description="Filter by source")

@mcp.tool(
    name="hivemind_read",
    annotations={
        "title": "[GATEWAY] Read from Hive Mind shared memory",
        "readOnlyHint": True
    }
)
async def hivemind_read(params: HiveMindReadInput) -> str:
    """Read recent entries from the Sovereign Mind Hive Mind."""
    try:
        import snowflake.connector
        
        conn = snowflake.connector.connect(
            user=config.SNOWFLAKE_USER,
            password=config.SNOWFLAKE_PASSWORD,
            account=config.SNOWFLAKE_ACCOUNT,
            warehouse=config.SNOWFLAKE_WAREHOUSE,
            database=config.SNOWFLAKE_DATABASE,
            role=config.SNOWFLAKE_ROLE
        )
        
        where_clauses = ["STATUS = 'ACTIVE'"]
        params_list = []
        
        if params.category:
            where_clauses.append("CATEGORY = %s")
            params_list.append(params.category)
        if params.workstream:
            where_clauses.append("WORKSTREAM = %s")
            params_list.append(params.workstream)
        if params.source:
            where_clauses.append("SOURCE = %s")
            params_list.append(params.source)
        
        where_sql = " AND ".join(where_clauses)
        params_list.append(params.limit)
        
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT ID, CREATED_AT, SOURCE, CATEGORY, WORKSTREAM, SUMMARY, PRIORITY
            FROM SOVEREIGN_MIND.RAW.HIVE_MIND
            WHERE {where_sql}
            ORDER BY CREATED_AT DESC
            LIMIT %s
        """, params_list)
        
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            row_dict = {}
            for i, col in enumerate(columns):
                value = row[i]
                if hasattr(value, 'isoformat'):
                    value = value.isoformat()
                row_dict[col] = value
            results.append(row_dict)
        
        cursor.close()
        conn.close()
        
        return safe_json_response({
            "success": True,
            "count": len(results),
            "entries": results
        })
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})

# ============================================================================
# ASANA TOOLS (WITH LIMITS)
# ============================================================================

ASANA_BASE_URL = "https://app.asana.com/api/1.0"

async def asana_request(
    method: str,
    endpoint: str,
    json_data: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Dict[str, Any]:
    """Make authenticated request to Asana API."""
    headers = {
        "Authorization": f"Bearer {config.ASANA_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    return await make_api_request(
        method=method,
        url=f"{ASANA_BASE_URL}{endpoint}",
        headers=headers,
        json_data=json_data,
        params=params
    )

class AsanaListTasksInput(BaseModel):
    """Input for listing Asana tasks."""
    project_id: Optional[str] = Field(default=None, description="Project ID to list tasks from")
    assignee: Optional[str] = Field(default=None, description="Assignee (use 'me' for authenticated user)")
    completed: Optional[bool] = Field(default=False, description="Include completed tasks")
    limit: int = Field(default=50, ge=1, le=100, description="Max tasks to return")

@mcp.tool(
    name="asana_list_tasks",
    annotations={
        "title": "[ASANA] List tasks (limited)",
        "readOnlyHint": True
    }
)
async def asana_list_tasks(params: AsanaListTasksInput) -> str:
    """List Asana tasks with automatic limiting."""
    query_params = {
        "limit": min(params.limit, limits.MAX_TASK_COUNT),
        "opt_fields": "name,completed,due_on,assignee.name,projects.name"
    }
    
    if params.project_id:
        query_params["project"] = params.project_id
    if params.assignee:
        query_params["assignee"] = params.assignee
        query_params["workspace"] = config.ASANA_WORKSPACE_ID
    if params.completed is not None:
        query_params["completed_since"] = "now" if not params.completed else None
    
    result = await asana_request("GET", "/tasks", params=query_params)
    
    if "data" in result:
        return safe_json_response({
            "success": True,
            "count": len(result["data"]),
            "tasks": result["data"]
        })
    
    return json.dumps(result, indent=2)

class AsanaCreateTaskInput(BaseModel):
    """Input for creating an Asana task."""
    name: str = Field(..., description="Task name")
    notes: Optional[str] = Field(default=None, description="Task description")
    project_id: Optional[str] = Field(default=None, description="Project to add task to")
    assignee: Optional[str] = Field(default=None, description="Assignee user ID or 'me'")
    due_date: Optional[str] = Field(default=None, description="Due date (YYYY-MM-DD)")

@mcp.tool(
    name="asana_create_task",
    annotations={
        "title": "[ASANA] Create a new task",
        "readOnlyHint": False
    }
)
async def asana_create_task(params: AsanaCreateTaskInput) -> str:
    """Create a new Asana task."""
    task_data = {
        "name": params.name,
        "workspace": config.ASANA_WORKSPACE_ID
    }
    
    if params.notes:
        task_data["notes"] = params.notes
    if params.project_id:
        task_data["projects"] = [params.project_id]
    if params.assignee:
        task_data["assignee"] = params.assignee
    if params.due_date:
        task_data["due_on"] = params.due_date
    
    result = await asana_request("POST", "/tasks", json_data={"data": task_data})
    return json.dumps(result, indent=2)

class AsanaGetTaskInput(BaseModel):
    """Input for getting a task."""
    task_id: str = Field(..., description="Task ID")

@mcp.tool(
    name="asana_get_task",
    annotations={
        "title": "[ASANA] Get task details",
        "readOnlyHint": True
    }
)
async def asana_get_task(params: AsanaGetTaskInput) -> str:
    """Get detailed information about an Asana task."""
    result = await asana_request(
        "GET", 
        f"/tasks/{params.task_id}",
        params={"opt_fields": "name,notes,completed,due_on,assignee.name,projects.name,tags.name,custom_fields"}
    )
    return json.dumps(result, indent=2)

class AsanaCompleteTaskInput(BaseModel):
    """Input for completing a task."""
    task_id: str = Field(..., description="Task ID")

@mcp.tool(
    name="asana_complete_task",
    annotations={
        "title": "[ASANA] Mark task complete",
        "readOnlyHint": False
    }
)
async def asana_complete_task(params: AsanaCompleteTaskInput) -> str:
    """Mark an Asana task as complete."""
    result = await asana_request(
        "PUT",
        f"/tasks/{params.task_id}",
        json_data={"data": {"completed": True}}
    )
    return json.dumps(result, indent=2)

class AsanaAddCommentInput(BaseModel):
    """Input for adding a comment."""
    task_id: str = Field(..., description="Task ID")
    text: str = Field(..., description="Comment text")

@mcp.tool(
    name="asana_add_comment",
    annotations={
        "title": "[ASANA] Add comment to task",
        "readOnlyHint": False
    }
)
async def asana_add_comment(params: AsanaAddCommentInput) -> str:
    """Add a comment to an Asana task."""
    result = await asana_request(
        "POST",
        f"/tasks/{params.task_id}/stories",
        json_data={"data": {"text": params.text}}
    )
    return json.dumps(result, indent=2)

# ============================================================================
# M365 EMAIL TOOLS (WITH LIMITS)
# ============================================================================

class M365ReadEmailsInput(BaseModel):
    """Input for reading M365 emails."""
    user_email: str = Field(default="john@middlegroundcapital.com")
    folder: str = Field(default="inbox")
    top: int = Field(default=25, ge=1, le=50, description="Max emails (capped at 50)")
    filter: Optional[str] = Field(default=None)
    search: Optional[str] = Field(default=None)

@mcp.tool(
    name="m365_read_emails",
    annotations={
        "title": "[M365] Read Emails (limited)",
        "readOnlyHint": True
    }
)
async def m365_read_emails(params: M365ReadEmailsInput) -> str:
    """Read emails with automatic limiting."""
    endpoint = f"/users/{params.user_email}/mailFolders/{params.folder}/messages"
    
    query_params = {
        "$top": min(params.top, limits.MAX_EMAIL_COUNT),
        "$orderby": "receivedDateTime desc",
        "$select": "id,subject,from,receivedDateTime,isRead,bodyPreview,hasAttachments"
    }
    
    if params.filter:
        query_params["$filter"] = params.filter
    if params.search:
        query_params["$search"] = f'"{params.search}"'
    
    result = await m365_graph_request("GET", endpoint, params=query_params)
    
    if "value" in result:
        emails = []
        for email in result["value"]:
            emails.append({
                "id": email.get("id"),
                "subject": email.get("subject"),
                "from": email.get("from", {}).get("emailAddress", {}).get("address"),
                "from_name": email.get("from", {}).get("emailAddress", {}).get("name"),
                "received": email.get("receivedDateTime"),
                "isRead": email.get("isRead"),
                "preview": email.get("bodyPreview", "")[:150],  # Shorter preview
                "hasAttachments": email.get("hasAttachments")
            })
        return safe_json_response({"success": True, "count": len(emails), "emails": emails})
    
    return json.dumps(result, indent=2)

class M365SendEmailInput(BaseModel):
    """Input for sending email."""
    to: List[str] = Field(..., description="Recipient email addresses")
    subject: str = Field(..., description="Email subject")
    body: str = Field(..., description="Email body")
    is_html: bool = Field(default=False)
    user_email: str = Field(default="john@middlegroundcapital.com")

@mcp.tool(
    name="m365_send_email",
    annotations={
        "title": "[M365] Send Email",
        "readOnlyHint": False
    }
)
async def m365_send_email(params: M365SendEmailInput) -> str:
    """Send an email via Microsoft 365."""
    endpoint = f"/users/{params.user_email}/sendMail"
    
    message = {
        "subject": params.subject,
        "body": {
            "contentType": "HTML" if params.is_html else "Text",
            "content": params.body
        },
        "toRecipients": [{"emailAddress": {"address": addr}} for addr in params.to]
    }
    
    result = await m365_graph_request("POST", endpoint, json_data={"message": message})
    
    if result.get("error"):
        return json.dumps(result)
    
    return json.dumps({"success": True, "message": "Email sent successfully"})

# ============================================================================
# GITHUB TOOLS (WITH LIMITS)
# ============================================================================

GITHUB_API = "https://api.github.com"

async def github_request(
    method: str,
    endpoint: str,
    json_data: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Dict[str, Any]:
    """Make authenticated request to GitHub API."""
    headers = {
        "Authorization": f"token {config.GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    return await make_api_request(
        method=method,
        url=f"{GITHUB_API}{endpoint}",
        headers=headers,
        json_data=json_data,
        params=params
    )

@mcp.tool(
    name="github_list_repos",
    annotations={
        "title": "[GITHUB] List repositories",
        "readOnlyHint": True
    }
)
async def github_list_repos() -> str:
    """List GitHub repositories accessible to the authenticated user."""
    result = await github_request("GET", "/user/repos", params={"per_page": 50, "sort": "updated"})
    
    if isinstance(result, list):
        repos = [{
            "name": r.get("name"),
            "full_name": r.get("full_name"),
            "private": r.get("private"),
            "updated_at": r.get("updated_at"),
            "language": r.get("language")
        } for r in result[:limits.MAX_FILE_LIST]]
        return safe_json_response({"success": True, "count": len(repos), "repos": repos})
    
    return json.dumps(result, indent=2)

class GitHubGetFileInput(BaseModel):
    """Input for getting a file."""
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    path: str = Field(..., description="File path")

@mcp.tool(
    name="github_get_file",
    annotations={
        "title": "[GITHUB] Get file contents",
        "readOnlyHint": True
    }
)
async def github_get_file(params: GitHubGetFileInput) -> str:
    """Get contents of a file from GitHub."""
    result = await github_request("GET", f"/repos/{params.owner}/{params.repo}/contents/{params.path}")
    
    if "content" in result:
        import base64
        try:
            content = base64.b64decode(result["content"]).decode('utf-8')
            # Truncate large files
            if len(content) > limits.MAX_RESPONSE_BYTES:
                content = content[:limits.MAX_RESPONSE_BYTES] + "\n\n[FILE TRUNCATED]"
            return json.dumps({
                "success": True,
                "path": result.get("path"),
                "size": result.get("size"),
                "sha": result.get("sha"),
                "content": content
            }, indent=2)
        except Exception as e:
            return json.dumps({"success": False, "error": f"Failed to decode: {e}"})
    
    return json.dumps(result, indent=2)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sse":
        # SSE mode for Claude.ai
        mcp.run(transport="sse")
    else:
        # stdio mode for Claude Code
        mcp.run()
