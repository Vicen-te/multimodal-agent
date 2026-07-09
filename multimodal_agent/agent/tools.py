"""Tool schemas the agent binds to the model for routing.

These Pydantic models describe the tools to the language model so it can emit
tool calls. Execution happens in the graph's tool node, which has access to the
session image and the retriever; that is why the schemas carry only the arguments
the model itself can supply.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnalyzeImage(BaseModel):
    """Analyze the image the user attached to answer a question about its visual content.

    Use this when the question refers to what is shown in the image, such as code
    on a screenshot, a chart, an object, a diagram, or an error message.
    """

    query: str = Field(description="What to look for or answer about the image")


class SearchDocs(BaseModel):
    """Search the technical documentation knowledge base.

    Use this when the question is about concepts, tools, or how-to information that
    would be found in documentation rather than in the image.
    """

    query: str = Field(description="The search query to look up in the documentation")


TOOLS: list[type[BaseModel]] = [AnalyzeImage, SearchDocs]
