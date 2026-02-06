"""
ReAct Agent for multi-step reasoning and acting.

Implements the ReAct (Reasoning + Acting) pattern for complex queries that require
multiple retrieval steps, query reformulation, and iterative refinement.

Key Concepts:
- Reasoning: Agent thinks about what it needs to do next
- Acting: Agent takes actions (search, rewrite query, answer)
- Iteration: Agent loops until it has enough information or reaches max iterations

References:
- ReAct Paper: https://arxiv.org/abs/2210.03629
"""
import logging
import re
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage

from src.generation.llm import LLMGenerator
from src.agents.query_rewriter import QueryRewriter
from src.config.settings import settings

logger = logging.getLogger(__name__)


# ReAct Prompt Template
REACT_PROMPT = """You are an intelligent agent that can reason and take actions to answer questions.

You have access to these actions:
- SEARCH: Retrieve documents using a query
- REWRITE_QUERY: Reformulate the current query for better results
- ANSWER: Generate final answer when you have enough information
- NEED_MORE_INFO: Continue reasoning if you need more steps
- FINISH: Complete the task

Current State:
- Original Question: {original_query}
- Current Query: {current_query}
- Iteration: {iteration}/{max_iterations}
- Documents Retrieved: {num_documents}
- Previous Actions: {previous_actions}

{documents_summary}

Instructions:
1. Think step-by-step about what to do next
2. Choose the most appropriate action
3. Provide any input needed for that action

Your response MUST follow this exact format:

Thought: [Your reasoning about what to do next]
Action: [One of: SEARCH, REWRITE_QUERY, ANSWER, NEED_MORE_INFO, FINISH]
Action Input: [The input for the action, e.g., a search query or empty if not needed]

Example Response:
Thought: I need more specific information about RAG systems. Let me search for documents.
Action: SEARCH
Action Input: retrieval augmented generation architecture

Now, reason about the current state and decide what to do next:"""


class AgentAction(Enum):
    """
    Actions the ReAct agent can take.

    Each action represents a decision point in the agent's reasoning loop.
    """
    SEARCH = "search"              # Retrieve documents with current query
    REWRITE_QUERY = "rewrite"      # Reformulate query for better results
    ANSWER = "answer"              # Generate final answer from documents
    NEED_MORE_INFO = "need_more"   # Need more information, continue loop
    FINISH = "finish"              # Complete, return results


@dataclass
class AgentState:
    """
    Tracks the state of the ReAct agent during execution.

    This state object is updated throughout the agent's reasoning loop
    and provides a complete trace of the agent's decision-making process.

    Attributes:
        original_query: The user's initial query
        current_query: The query being used in current iteration
        documents: Accumulated retrieved documents across all iterations
        actions: History of actions taken by the agent
        thoughts: Reasoning steps at each iteration
        iterations: Number of iterations completed
        query_evolution: History of query transformations
    """
    original_query: str
    current_query: str
    documents: List[Document] = field(default_factory=list)
    actions: List[AgentAction] = field(default_factory=list)
    thoughts: List[str] = field(default_factory=list)
    iterations: int = 0
    query_evolution: List[str] = field(default_factory=list)

    def add_action(self, action: AgentAction, thought: str):
        """
        Record an action and associated reasoning.

        Args:
            action: The action taken
            thought: The reasoning behind the action
        """
        self.actions.append(action)
        self.thoughts.append(thought)
        logger.info(f"Action: {action.value} | Thought: {thought[:100]}...")

    def add_documents(self, docs: List[Document]):
        """
        Add retrieved documents to state.

        Deduplicates documents by content to avoid redundancy.

        Args:
            docs: Documents to add
        """
        # Simple deduplication by content
        existing_contents = {doc.page_content for doc in self.documents}
        new_docs = [
            doc for doc in docs
            if doc.page_content not in existing_contents
        ]

        self.documents.extend(new_docs)
        logger.info(f"Added {len(new_docs)} new documents (total: {len(self.documents)})")

    def update_query(self, new_query: str):
        """
        Update current query and track evolution.

        Args:
            new_query: The new query formulation
        """
        if new_query != self.current_query:
            self.query_evolution.append(new_query)
            self.current_query = new_query
            logger.info(f"Query updated: '{new_query}'")

    def increment_iteration(self):
        """Increment iteration counter."""
        self.iterations += 1
        logger.info(f"Iteration {self.iterations} complete")


class ReActAgent:
    """
    ReAct Agent for multi-step reasoning and information retrieval.

    The ReAct agent implements a think-act-observe loop:
    1. Think: Reason about what information is needed
    2. Act: Take an action (search, rewrite, answer)
    3. Observe: Review results and decide next step

    This pattern is particularly useful for:
    - Complex queries requiring multiple retrieval steps
    - Queries needing reformulation for better results
    - Multi-hop reasoning across different sources
    - Iterative refinement based on initial results

    Example:
        >>> from src.agents.react_agent import ReActAgent
        >>> agent = ReActAgent(
        ...     retriever=retriever,
        ...     query_rewriter=rewriter,
        ...     llm=llm,
        ...     max_iterations=5
        ... )
        >>> result = agent.run("What is RAG and how does it work?")
        >>> print(result['answer'])
        >>> print(f"Took {len(result['reasoning_trace'])} steps")
    """

    def __init__(
        self,
        retriever,
        query_rewriter: Optional[QueryRewriter] = None,
        llm: Optional[LLMGenerator] = None,
        max_iterations: int = 5,
        temperature: float = 0.3,
    ):
        """
        Initialize ReAct agent.

        Args:
            retriever: Retriever instance for document search
            query_rewriter: Optional QueryRewriter for query optimization
            llm: Optional LLM for reasoning (creates new one if not provided)
            max_iterations: Maximum reasoning iterations (default: 5)
            temperature: LLM temperature for reasoning (default: 0.3)
        """
        self.retriever = retriever
        self.query_rewriter = query_rewriter or QueryRewriter()
        self.llm = llm or LLMGenerator(temperature=temperature, max_tokens=500)
        self.max_iterations = max_iterations
        self.temperature = temperature

        logger.info(
            f"Initialized ReActAgent: max_iterations={max_iterations}, "
            f"temperature={temperature}"
        )

    def run(self, query: str) -> Dict[str, Any]:
        """
        Execute the ReAct reasoning loop for a query.

        The agent will iteratively:
        1. Reason about what action to take
        2. Execute the action (search, rewrite, answer)
        3. Observe results
        4. Repeat until answer is ready or max iterations reached

        Args:
            query: User's question

        Returns:
            Dictionary containing:
            - answer: Final generated answer
            - documents: Retrieved documents used
            - reasoning_trace: List of thoughts at each step
            - query_evolution: History of query reformulations
            - iterations: Number of iterations taken
            - actions_taken: List of actions executed

        Example:
            >>> result = agent.run("What is machine learning?")
            >>> print(result['answer'])
            >>> print(f"Query evolution: {result['query_evolution']}")
        """
        logger.info(f"Starting ReAct loop for query: '{query}'")

        # Initialize state
        state = AgentState(
            original_query=query,
            current_query=query,
            query_evolution=[query],
        )

        try:
            # Main ReAct loop
            answer = None
            while state.iterations < self.max_iterations:
                state.increment_iteration()

                # Step 1: Reason about next action
                action, thought, action_input = self._decide_action(state)
                state.add_action(action, thought)

                # Step 2: Act - execute the action
                observation = self._act(action, action_input, state)

                # Step 3: Observe - check if we should finish
                if action in [AgentAction.ANSWER, AgentAction.FINISH]:
                    answer = observation
                    break

            # If we hit max iterations without answering
            if answer is None:
                logger.warning(
                    f"Reached max iterations ({self.max_iterations}), "
                    "generating answer with available information"
                )
                answer = self._execute_answer(state, "")

            # Format and return final result
            return self._format_result(state, answer)

        except Exception as e:
            logger.error(f"Error in ReAct loop: {str(e)}", exc_info=True)

            # Return fallback result
            return self._format_result(
                state,
                answer=f"I encountered an error: {str(e)}. Please try rephrasing.",
                error=str(e)
            )


    def _decide_action(
        self,
        state: AgentState,
    ) -> Tuple[AgentAction, str, str]:
        """
        Decide what action to take next based on current state.

        This is the "Reasoning" part of ReAct. The agent examines:
        - Current query
        - Documents retrieved so far
        - Iteration count
        - Previous actions

        And uses LLM-based reasoning to decide what to do next.

        Args:
            state: Current agent state

        Returns:
            Tuple of (action, reasoning_thought, action_input)
        """
        try:
            # Use LLM-based reasoning
            thought, action, action_input = self._reason(state)
            return action, thought, action_input

        except Exception as e:
            logger.warning(f"Error in reasoning: {str(e)}, using fallback logic")

            # Fallback logic if reasoning fails
            if state.iterations == 1 and not state.documents:
                return AgentAction.SEARCH, "Starting with initial search", state.current_query
            elif len(state.documents) > 0:
                return AgentAction.ANSWER, "Have documents, generating answer", ""
            elif state.iterations >= self.max_iterations - 1:
                return AgentAction.ANSWER, "Max iterations reached, answering", ""
            else:
                return AgentAction.SEARCH, "Continuing search", state.current_query

    def _reason(
        self,
        state: AgentState,
    ) -> Tuple[str, AgentAction, str]:
        """
        Use LLM to reason about what action to take next.

        Creates a prompt with current state and asks the LLM to:
        1. Think about what to do
        2. Choose an action
        3. Provide action input

        Args:
            state: Current agent state

        Returns:
            Tuple of (thought, action, action_input)

        Raises:
            ValueError: If LLM response cannot be parsed
        """
        # Build documents summary
        if state.documents:
            docs_summary = "Documents Retrieved:\n"
            for i, doc in enumerate(state.documents[:3], 1):  # Show first 3
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                docs_summary += f"{i}. {preview}\n"
            if len(state.documents) > 3:
                docs_summary += f"... and {len(state.documents) - 3} more documents\n"
        else:
            docs_summary = "No documents retrieved yet.\n"

        # Build previous actions summary
        if state.actions:
            prev_actions = ", ".join([a.value for a in state.actions[-3:]])  # Last 3
        else:
            prev_actions = "None"

        # Format prompt
        prompt = REACT_PROMPT.format(
            original_query=state.original_query,
            current_query=state.current_query,
            iteration=state.iterations + 1,
            max_iterations=self.max_iterations,
            num_documents=len(state.documents),
            previous_actions=prev_actions,
            documents_summary=docs_summary,
        )

        # Get LLM response
        messages = [HumanMessage(content=prompt)]
        response = self.llm.generate(messages)

        # Parse response
        thought, action, action_input = self._parse_react_response(response)

        logger.info(
            f"Reasoning: thought='{thought[:50]}...', "
            f"action={action.value}, input='{action_input[:30]}...'"
        )

        return thought, action, action_input

    def _parse_react_response(
        self,
        response: str,
    ) -> Tuple[str, AgentAction, str]:
        """
        Parse structured response from LLM.

        Extracts "Thought:", "Action:", and "Action Input:" from the response.

        Args:
            response: LLM's response text

        Returns:
            Tuple of (thought, action, action_input)

        Raises:
            ValueError: If response cannot be parsed
        """
        # Extract thought
        thought_match = re.search(
            r'Thought:\s*(.+?)(?=\nAction:|\n\n|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        thought = thought_match.group(1).strip() if thought_match else ""

        # Extract action
        action_match = re.search(
            r'Action:\s*(\w+)',
            response,
            re.IGNORECASE
        )
        if not action_match:
            raise ValueError(f"Could not parse action from response: {response}")

        action_str = action_match.group(1).upper()

        # Map to AgentAction enum
        action_mapping = {
            'SEARCH': AgentAction.SEARCH,
            'REWRITE_QUERY': AgentAction.REWRITE_QUERY,
            'REWRITE': AgentAction.REWRITE_QUERY,
            'ANSWER': AgentAction.ANSWER,
            'NEED_MORE_INFO': AgentAction.NEED_MORE_INFO,
            'NEED_MORE': AgentAction.NEED_MORE_INFO,
            'FINISH': AgentAction.FINISH,
        }

        action = action_mapping.get(action_str)
        if not action:
            logger.warning(f"Unknown action '{action_str}', defaulting to SEARCH")
            action = AgentAction.SEARCH

        # Extract action input
        input_match = re.search(
            r'Action Input:\s*(.+?)(?=\n\n|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        action_input = input_match.group(1).strip() if input_match else ""

        # Validate and clean
        if not thought:
            thought = "Continuing with next step"

        return thought, action, action_input

    def _act(
        self,
        action: AgentAction,
        action_input: str,
        state: AgentState,
    ) -> str:
        """
        Execute the chosen action and return observation.

        This is the "Acting" part of ReAct. Takes the action decided
        by reasoning and executes it, updating state and returning
        what was observed.

        Args:
            action: The action to execute
            action_input: Input/parameter for the action
            state: Current agent state

        Returns:
            Observation string describing what happened

        Example:
            >>> observation = agent._act(AgentAction.SEARCH, "RAG systems", state)
            >>> # Returns: "Retrieved 3 documents about RAG systems"
        """
        logger.info(f"Acting: {action.value} with input: '{action_input[:50]}...'")

        try:
            if action == AgentAction.SEARCH:
                return self._execute_search(state, action_input)

            elif action == AgentAction.REWRITE_QUERY:
                return self._execute_rewrite(state, action_input)

            elif action == AgentAction.ANSWER or action == AgentAction.FINISH:
                return self._execute_answer(state, action_input)

            elif action == AgentAction.NEED_MORE_INFO:
                return (
                    "Continuing to next iteration to gather more information."
                )

            else:
                logger.warning(f"Unknown action: {action}")
                return f"Unknown action: {action.value}"

        except Exception as e:
            logger.error(f"Error executing action {action.value}: {str(e)}")
            return f"Error: {str(e)}"

    def _execute_search(
        self,
        state: AgentState,
        query_override: str = "",
    ) -> str:
        """
        Execute document retrieval with current or overridden query.

        Args:
            state: Current agent state
            query_override: Optional query to use instead of state.current_query

        Returns:
            Observation string
        """
        search_query = query_override or state.current_query
        logger.info(f"Executing SEARCH with query: '{search_query}'")

        try:
            # Retrieve documents
            docs = self.retriever.retrieve(search_query)

            # Add to state
            state.add_documents(docs)

            observation = (
                f"Retrieved {len(docs)} documents for query: '{search_query}'. "
                f"Total documents: {len(state.documents)}"
            )
            logger.info(observation)
            return observation

        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return f"Search failed: {str(e)}"

    def _execute_rewrite(
        self,
        state: AgentState,
        query_override: str = "",
    ) -> str:
        """
        Rewrite current query for better retrieval.

        Args:
            state: Current agent state
            query_override: Optional specific query to rewrite

        Returns:
            Observation string
        """
        original_query = query_override or state.current_query
        logger.info(f"Executing REWRITE_QUERY for: '{original_query}'")

        try:
            # Rewrite query
            new_query = self.query_rewriter.rewrite(original_query)

            # Update state
            state.update_query(new_query)

            observation = (
                f"Query rewritten from '{original_query}' to '{new_query}'"
            )
            logger.info(observation)
            return observation

        except Exception as e:
            logger.error(f"Error during rewrite: {str(e)}")
            return f"Rewrite failed: {str(e)}, keeping original query"

    def _execute_answer(
        self,
        state: AgentState,
        additional_context: str = "",
    ) -> str:
        """
        Generate final answer from accumulated documents.

        Args:
            state: Current agent state
            additional_context: Optional additional context to include

        Returns:
            Generated answer string
        """
        logger.info(
            f"Executing ANSWER with {len(state.documents)} documents"
        )

        try:
            # If no documents, return fallback
            if not state.documents:
                return (
                    f"I couldn't find relevant information to answer your question: "
                    f"'{state.original_query}'. Please try rephrasing or providing "
                    "more context."
                )

            # Build context from documents
            context = "\n\n".join([
                f"Document {i+1}:\n{doc.page_content}"
                for i, doc in enumerate(state.documents[:5])  # Use top 5
            ])

            # Add additional context if provided
            if additional_context:
                context += f"\n\nAdditional Context:\n{additional_context}"

            # Build prompt
            prompt = f"""Based on the following documents, please answer the user's question.

Question: {state.original_query}

Documents:
{context}

Instructions:
- Provide a clear, comprehensive answer based on the documents
- Cite specific information from the documents
- If the documents don't fully answer the question, say so
- Be concise but thorough

Answer:"""

            # Generate answer
            messages = [HumanMessage(content=prompt)]
            answer = self.llm.generate(messages)

            logger.info("Answer generated successfully")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return (
                f"I encountered an error generating an answer: {str(e)}. "
                "The retrieved documents may contain relevant information."
            )

    def _format_result(
        self,
        state: AgentState,
        answer: str,
        error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Format final result with all tracking information.

        Args:
            state: Current agent state
            answer: Final answer generated
            error: Optional error message

        Returns:
            Dictionary containing complete result with:
            - answer: Final generated answer
            - documents: Retrieved documents used
            - reasoning_trace: List of thoughts at each step
            - query_evolution: History of query reformulations
            - iterations: Number of iterations taken
            - actions_taken: List of actions executed
            - error: Optional error message
        """
        result = {
            'answer': answer,
            'documents': state.documents,
            'reasoning_trace': state.thoughts,
            'query_evolution': state.query_evolution,
            'iterations': state.iterations,
            'actions_taken': [action.value for action in state.actions],
        }

        if error:
            result['error'] = error

        logger.info(
            f"ReAct loop complete: {state.iterations} iterations, "
            f"{len(state.documents)} documents, "
            f"{len(state.query_evolution)} query versions"
        )

        return result

    def get_agent_config(self) -> Dict[str, Any]:
        """
        Get current agent configuration.

        Returns:
            Dictionary with agent parameters
        """
        return {
            'max_iterations': self.max_iterations,
            'temperature': self.temperature,
            'has_query_rewriter': self.query_rewriter is not None,
            'llm_config': self.llm.get_model_info(),
        }


# Convenience function
def create_react_agent(
    retriever,
    max_iterations: int = 5,
    **kwargs,
) -> ReActAgent:
    """
    Create a ReAct agent instance.

    Args:
        retriever: Retriever instance for document search
        max_iterations: Maximum reasoning iterations
        **kwargs: Additional ReActAgent configuration

    Returns:
        ReActAgent instance

    Example:
        >>> agent = create_react_agent(retriever, max_iterations=3)
        >>> result = agent.run("What is RAG?")
    """
    return ReActAgent(
        retriever=retriever,
        max_iterations=max_iterations,
        **kwargs,
    )
