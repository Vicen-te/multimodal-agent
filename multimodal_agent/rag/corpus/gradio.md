# Gradio

Gradio is a Python library for building web interfaces around machine-learning
models with very little code. You describe inputs and outputs, point them at a
function, and Gradio serves a browser UI and a matching HTTP API. It is the
native way to publish a demo on Hugging Face Spaces, which builds and hosts a
Gradio app straight from a repository.

For chat applications, `gr.ChatInterface` wraps a function into a full
conversation UI with history. Setting `multimodal=True` lets the user attach
files such as images next to their text message, so a single widget handles both
modalities. When the function is written as a Python generator that yields
partial strings, Gradio streams those updates into the chat bubble token by
token, which gives the responsive feel users expect from an assistant.

Keeping the interface thin is good practice: the UI layer should only collect the
image and text, hand them to the agent, and render whatever streams back. All the
reasoning, tool routing, and retrieval belong in the agent code behind it, not in
the frontend.
