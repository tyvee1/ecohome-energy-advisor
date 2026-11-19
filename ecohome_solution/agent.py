"""
EcoHome Energy Advisor agent, built as an explicit LangGraph graph.

The starter used LangGraph's prebuilt ``create_react_agent``. That works, but it hides
the three things this project asks you to define: the state schema, the nodes and the
edges. This file builds the same tool-calling loop by hand so each part is visible:

    START -> agent --(tool calls?)--> tools -> agent -> ... -> END
                   \\--(no tool calls)--> END

* **Schema** (``AdvisorState``): the conversation (``messages``, merged by the
  ``add_messages`` reducer so each node only returns what it adds), plus the
  context the advisor works in (``context``) and a counter of model turns
  (``steps``) used to stop a run that keeps calling tools.
* **Nodes**: ``agent`` calls the model with the system prompt, the context and the
  conversation; ``tools`` runs the tool calls the model asked for.
* **Edges**: a conditional edge after ``agent`` goes to ``tools`` when the last
  message contains tool calls and to ``END`` otherwise; ``tools`` always returns
  to ``agent``.

Context awareness: every run is told today's date, the day of the week and the
current hour, because "tomorrow" or "this week" cannot be answered without them,
and a model does not know the date on its own.

Error handling: a tool that raises is turned into an error message the model can
read (``ToolNode(handle_tool_errors=True)``); the number of model turns is capped
(``max_steps``); a failure calling the model is returned as a readable final
message with ``error`` set in the result, rather than crashing the caller.
"""
import os
from datetime import datetime, timedelta
from typing import Annotated, Any, Dict, Optional

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tools import TOOL_KIT

load_dotenv()

VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"


class AdvisorState(TypedDict, total=False):
    """Graph state: the conversation, the context the advisor works in, and a step counter."""
    messages: Annotated[list[AnyMessage], add_messages]
    context: str
    steps: int


class Agent:
    def __init__(self, instructions: str, model: str = "gpt-4o-mini", max_steps: int = 12,
                 llm: Any = None, tools: Optional[list] = None):
        """
        Build the Energy Advisor.

        Args:
            instructions: the system prompt (who the advisor is, what to do, how to answer).
            model: model name on the course endpoint (default "gpt-4o-mini").
            max_steps: most model turns per question; stops a run that keeps calling tools.
            llm: optional chat model to use instead of the course endpoint (used for offline tests).
            tools: optional tool list (defaults to every tool in tools.TOOL_KIT).
        """
        if not instructions or not instructions.strip():
            raise ValueError("instructions must not be empty: the system prompt defines the advisor")
        self.instructions = instructions
        self.model = model
        self.max_steps = max_steps
        self.tools = list(tools or TOOL_KIT)

        # LLM: temperature 0 for repeatable recommendations; the course endpoint and key.
        self.llm = llm or ChatOpenAI(
            model=model,
            temperature=0.0,
            base_url=VOCAREUM_BASE_URL,
            api_key=os.getenv("VOCAREUM_API_KEY"),
            timeout=60,
            max_retries=2,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    # ---------------------------------------------------------------- graph

    def _agent_node(self, state: AdvisorState) -> Dict[str, Any]:
        """Call the model with the instructions, the run's context and the conversation so far."""
        steps = state.get("steps", 0)
        if steps >= self.max_steps:
            return {"messages": [AIMessage(content=(
                "I stopped before finishing because this question needed more steps than allowed "
                f"({self.max_steps}). Please ask a narrower question."))], "steps": steps}
        system = [SystemMessage(content=self.instructions)]
        if state.get("context"):
            system.append(SystemMessage(content=f"Context for this request:\n{state['context']}"))
        reply = self.llm_with_tools.invoke(system + list(state["messages"]))
        return {"messages": [reply], "steps": steps + 1}

    @staticmethod
    def _route(state: AdvisorState) -> str:
        """Go to the tools when the model asked for any; otherwise the answer is final."""
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    def _build_graph(self):
        g = StateGraph(AdvisorState)
        g.add_node("agent", self._agent_node)
        g.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", self._route, ["tools", END])
        g.add_edge("tools", "agent")
        return g.compile()

    # ---------------------------------------------------------------- public API

    @staticmethod
    def build_context(context: Optional[str] = None, now: Optional[datetime] = None) -> str:
        """Today's date and time, a seven-day calendar, plus whatever the caller supplied (location, preferences).

        The calendar names the weekday of each of the next seven dates. Models are unreliable at calendar
        arithmetic: given only today's date, the advisor once took "Wednesday" to be a Monday. With the dates
        listed, "Wednesday" is a lookup, not a calculation.
        """
        now = now or datetime.now()
        tomorrow = now + timedelta(days=1)
        week = ", ".join(f"{d:%A} {d:%Y-%m-%d}" for d in (now + timedelta(days=k) for k in range(1, 8)))
        lines = [f"Current date and time: {now:%Y-%m-%d %H:%M} ({now:%A}).",
                 f"Tomorrow is {tomorrow:%Y-%m-%d} ({tomorrow:%A}).",
                 f"The next seven days: {week}."]
        if context:
            lines.append(context.strip())
        return "\n".join(lines)

    def invoke(self, question: str, context: str = None) -> Dict[str, Any]:
        """
        Ask the Energy Advisor a question about energy optimization.

        Args:
            question (str): The user's question about energy optimization.
            context (str): Extra context for this request, e.g. "Location: San Francisco, CA".
                Today's date and time are always added.

        Returns:
            dict: the final graph state. ``messages`` holds the whole exchange (the question,
            every tool call and tool result, and the final answer as the last message);
            ``steps`` counts model turns; ``error`` is set only if the model could not be reached.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        state = {"messages": [("user", question)], "context": self.build_context(context), "steps": 0}
        try:
            return self.graph.invoke(state, config={"recursion_limit": 2 * self.max_steps + 5})
        except Exception as e:  # network, authentication, rate limit
            return {"messages": [HumanMessage(content=question), AIMessage(content=f"Sorry, I could not complete this request: {type(e).__name__}.")],
                    "context": state["context"], "steps": 0, "error": f"{type(e).__name__}: {e}"}

    def get_agent_tools(self):
        """Names of the tools available to the Energy Advisor."""
        return [t.name for t in self.tools]
