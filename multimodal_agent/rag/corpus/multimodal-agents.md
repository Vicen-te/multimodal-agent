# Multimodal agents

A multimodal agent reasons over more than one kind of input, most often images
together with text. A robust design keeps the agent's main reasoning loop in
text and treats vision as a tool. When the agent decides the image is relevant,
it calls a vision tool that turns the picture into a textual description or
answer; that text then flows back into the same loop as any other observation.

This text-centred design has practical benefits. The agent's reasoning stays easy
to log and debug, because every step is text. Vision composes naturally with
other text-based tools such as a document search, since they all return text the
agent can weigh together. And the agent can choose, per question, whether to look
at the image, search the documentation, do both, or neither.

Routing is the heart of such an agent. A clear system prompt tells it to use the
vision tool for questions about the image's content, the search tool for
questions about documentation, and both when a question spans the two. Evaluating
the agent therefore has two layers: routing accuracy, meaning did it pick the
right tools, and answer quality, meaning was the final response correct.
