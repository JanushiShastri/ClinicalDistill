#!/bin/bash

echo "🏥 Starting ClinicalDistill MCP Server..."
echo ""

# Check OpenAI key
if [ -z "$OPENAI_API_KEY" ]; then
    source ~/.zshrc
fi

echo "✅ OpenAI key: ${OPENAI_API_KEY:0:10}..."
echo "✅ Server path: po-community-mcp/python"
echo ""
echo "⚠️  Open a second terminal and run:"
echo "    ngrok http 8000 --host-header=localhost:8000"
echo ""
echo "⚠️  Then update Prompt Opinion MCP URL to:"
echo "    https://YOUR-NGROK-URL/mcp"
echo ""
echo "Starting server..."

cd ~/LLM_PROJECTS/ClinicalDistill/po-community-mcp/python
conda run -n llm-workspace uvicorn main:app --reload --port 8000
