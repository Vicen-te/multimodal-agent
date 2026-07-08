# LangGraph

LangGraph is a library for building stateful, multi-step applications with large
language models. Where a plain LangChain chain runs a fixed linear sequence,
LangGraph models the computation as a graph of nodes connected by edges, which
lets the flow contain cycles. This matters for agents, because an agent often
needs to loop: think, call a tool, observe the result, think again, and only
then answer.

A LangGraph program is built around three ideas. The state is a typed dictionary
that every node reads from and writes to; reducers such as `add_messages` control
how updates merge into the running state instead of overwriting it. Nodes are
plain functions that take the state and return a partial update. Edges connect
nodes; conditional edges inspect the current state and choose the next node at
runtime, which is how routing decisions like "should I call a tool or answer now"
are expressed.

The canonical agent pattern is the ReAct loop: an `agent` node asks the model
what to do, a conditional edge routes to a `tools` node when the model emits a
tool call, and the `tools` node feeds results back to the `agent` node. The loop
ends when the model produces a final answer with no tool call. Because the graph
is explicit, you can add extra nodes such as a reflection step that critiques the
draft answer and routes back for another attempt when it is incomplete.
