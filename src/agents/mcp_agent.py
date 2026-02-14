"""
Agent that can use MCP tools to answer queries.
Integrates MCP with LLM reasoning.
"""
import logging
import json
import re
from typing import Dict, List, Any, Optional

from src.generation.llm import LLMGenerator
from src.mcp.manager import MCPManager

logger = logging.getLogger(__name__)


class MCPAgent:
    """
    Agent that uses MCP tools to answer queries.

    Capabilities:
    - Analyzes queries to determine which MCP tools needed
    - Executes MCP tools with appropriate parameters
    - Synthesizes tool results into coherent answers
    """

    TOOL_SELECTION_PROMPT = """You are an AI assistant with access to tools via MCP (Model Context Protocol).

Available tools:
{tools_description}

User query: {query}

Analyze the query and determine which tools (if any) you need to use.

Respond in JSON format:
{{
  "needs_tools": true/false,
  "tools_to_use": [
    {{
      "server": "server_name",
      "tool": "tool_name",
      "arguments": {{
        "param1": "value1"
      }},
      "reasoning": "why this tool is needed"
    }}
  ],
  "needs_rag": true/false,
  "reasoning": "overall reasoning for tool selection"
}}

Only respond with valid JSON, no other text."""

    SYNTHESIS_PROMPT = """You are an AI assistant synthesizing information from multiple sources.

User query: {query}

Tool results:
{tool_results}

RAG context (if available):
{rag_context}

Synthesize a comprehensive answer that:
1. Directly addresses the user's question
2. Cites specific sources (tool results and documents)
3. Combines information coherently
4. Prioritizes fresh data from tools when relevant

Answer:"""

    def __init__(
        self,
        mcp_manager: Optional[MCPManager] = None,
        llm: Optional[LLMGenerator] = None
    ):
        """Initialize MCP agent."""
        self.mcp_manager = mcp_manager or MCPManager()
        self.llm = llm or LLMGenerator(temperature=0.1)
        logger.info("Initialized MCPAgent")

    def _get_tools_description(self) -> str:
        """Get formatted description of available tools."""
        all_tools = self.mcp_manager.get_available_tools()

        descriptions = []
        for server_name, tools in all_tools.items():
            for tool in tools:
                desc = f"Server: {server_name}\n"
                desc += f"Tool: {tool['name']}\n"
                desc += f"Description: {tool['description']}\n"
                desc += f"Parameters: {json.dumps(tool['input_schema'], indent=2)}\n"
                descriptions.append(desc)

        return "\n---\n".join(descriptions)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine which tools to use.

        Returns:
            Tool execution plan
        """
        tools_desc = self._get_tools_description()

        prompt = self.TOOL_SELECTION_PROMPT.format(
            tools_description=tools_desc,
            query=query
        )

        response = self.llm.generate_from_prompt(prompt)

        # Parse JSON response
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find raw JSON
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                json_str = json_match.group(0) if json_match else response

            plan = json.loads(json_str)
            logger.info(f"Tool plan: {plan}")
            return plan
        except Exception as e:
            logger.error(f"Failed to parse tool plan: {e}")
            logger.debug(f"Raw response: {response}")
            return {
                "needs_tools": False,
                "tools_to_use": [],
                "needs_rag": True,
                "reasoning": "Failed to parse plan"
            }

    def execute_tools(
        self,
        tool_plan: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Execute multiple tools from the plan.

        Returns:
            Dict mapping tool_name -> result
        """
        results = {}

        for i, tool_spec in enumerate(tool_plan):
            server = tool_spec["server"]
            tool = tool_spec["tool"]
            arguments = tool_spec.get("arguments", {})

            logger.info(f"Executing tool {i+1}/{len(tool_plan)}: {server}/{tool}")

            try:
                result = self.mcp_manager.call_tool(
                    server_name=server,
                    tool_name=tool,
                    arguments=arguments
                )

                results[f"{server}/{tool}"] = {
                    "success": True,
                    "result": result,
                    "reasoning": tool_spec.get("reasoning", "")
                }
            except Exception as e:
                logger.error(f"Tool execution failed: {e}")
                results[f"{server}/{tool}"] = {
                    "success": False,
                    "error": str(e),
                    "reasoning": tool_spec.get("reasoning", "")
                }

        return results

    def synthesize_answer(
        self,
        query: str,
        tool_results: Dict[str, Any],
        rag_context: Optional[str] = None
    ) -> str:
        """
        Synthesize final answer from tool results and RAG context.

        Args:
            query: Original query
            tool_results: Results from tool execution
            rag_context: Optional RAG context

        Returns:
            Synthesized answer
        """
        # Format tool results
        tool_results_str = json.dumps(tool_results, indent=2)

        prompt = self.SYNTHESIS_PROMPT.format(
            query=query,
            tool_results=tool_results_str,
            rag_context=rag_context or "No RAG context provided"
        )

        answer = self.llm.generate_from_prompt(prompt)
        return answer

    def run(
        self,
        query: str,
        rag_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the full MCP agent workflow.

        Args:
            query: User query
            rag_context: Optional RAG context

        Returns:
            Dict with answer and metadata
        """
        logger.info(f"MCPAgent processing query: {query}")

        # Step 1: Analyze query
        plan = self.analyze_query(query)

        # Step 2: Execute tools if needed
        tool_results = {}
        if plan.get("needs_tools", False):
            tool_specs = plan.get("tools_to_use", [])
            if tool_specs:
                tool_results = self.execute_tools(tool_specs)

        # Step 3: Synthesize answer
        answer = self.synthesize_answer(
            query=query,
            tool_results=tool_results,
            rag_context=rag_context
        )

        return {
            "answer": answer,
            "tool_plan": plan,
            "tool_results": tool_results,
            "used_rag": plan.get("needs_rag", False)
        }
