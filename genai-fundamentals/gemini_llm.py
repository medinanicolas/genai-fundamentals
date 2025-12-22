from __future__ import annotations
import os
from typing import Any, List, Optional, Union, cast, Sequence
from pydantic import ValidationError

from neo4j_graphrag.exceptions import LLMGenerationError
from neo4j_graphrag.llm.base import LLMInterface
from neo4j_graphrag.llm.types import (
    BaseMessage,
    LLMResponse,
    MessageList,
    ToolCall,
    ToolCallResponse,
)
from neo4j_graphrag.message_history import MessageHistory
from neo4j_graphrag.tool import Tool
from neo4j_graphrag.types import LLMMessage

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore

class GeminiLLM(LLMInterface):
    """Interface for Google Gemini models via the new google-genai SDK."""

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        model_params: Optional[dict[str, Any]] = None,
        api_key: Optional[str] = None,
        system_instruction: Optional[str] = None,
        **kwargs: Any,
    ):
        if genai is None:
            raise ImportError("pip install google-genai")
        
        super().__init__(model_name, model_params, None)
        
        resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not resolved_api_key:
            raise ValueError("Google API Key required (env: GOOGLE_API_KEY)")
        
        # New SDK: Client-based initialization
        self.client = genai.Client(api_key=resolved_api_key)
        
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.model_params = model_params or {}
        self.kwargs = kwargs

    def get_messages(self, input: str, message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None) -> list[types.Content]:
        """Converts history to the new SDK types.Content format."""
        messages = []
        
        if message_history:
            if isinstance(message_history, MessageHistory):
                message_history = message_history.messages
            
            try:
                MessageList(messages=cast(list[BaseMessage], message_history))
            except ValidationError:
                pass 

            for message in message_history:
                role = message.get("role")
                content = message.get("content", "")
                
                # 'assistant' maps to 'model'
                gemini_role = "model" if role == "assistant" else "user"
                
                messages.append(types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)]
                ))

        # Append current user input
        messages.append(types.Content(
            role="user",
            parts=[types.Part.from_text(text=input)]
        ))
        return messages

    def _get_config(self, tools: Optional[Sequence[Tool]] = None) -> types.GenerateContentConfig:
        """Helper to build the configuration object."""
        config_args = self.model_params.copy()
        
        # Add system instruction if present
        if self.system_instruction:
            config_args["system_instruction"] = self.system_instruction

        # Add tools if present
        if tools:
            # Convert Neo4j tools to Gemini FunctionDeclarations
            declarations = [
                types.FunctionDeclaration(
                    name=tool.get_name(),
                    description=tool.get_description(),
                    parameters=tool.get_parameters(exclude=["additional_properties"])
                )
                for tool in tools
            ]
            config_args["tools"] = [types.Tool(function_declarations=declarations)]
            
        return types.GenerateContentConfig(**config_args)

    def invoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            # Handle system instruction override
            current_instruction = system_instruction or self.system_instruction
            
            # Prepare config
            config = self._get_config()
            if current_instruction:
                 config.system_instruction = current_instruction

            messages = self.get_messages(input, message_history)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=config
            )
            return LLMResponse(content=response.text)
        except Exception as e:
            raise LLMGenerationError(f"Error calling GeminiLLM: {e}") from e

    async def ainvoke(
        self,
        input: str,
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> LLMResponse:
        try:
            current_instruction = system_instruction or self.system_instruction
            config = self._get_config()
            if current_instruction:
                 config.system_instruction = current_instruction

            messages = self.get_messages(input, message_history)
            
            # Async call via .aio
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=config
            )
            return LLMResponse(content=response.text)
        except Exception as e:
            raise LLMGenerationError(f"Error calling GeminiLLM: {e}") from e

    def invoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        try:
            current_instruction = system_instruction or self.system_instruction
            config = self._get_config(tools=tools)
            if current_instruction:
                 config.system_instruction = current_instruction

            messages = self.get_messages(input, message_history)
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=config
            )
            return self._parse_tool_response(response)
        except Exception as e:
             raise LLMGenerationError(f"Error calling GeminiLLM with tools: {e}") from e

    async def ainvoke_with_tools(
        self,
        input: str,
        tools: Sequence[Tool],
        message_history: Optional[Union[List[LLMMessage], MessageHistory]] = None,
        system_instruction: Optional[str] = None,
    ) -> ToolCallResponse:
        try:
            current_instruction = system_instruction or self.system_instruction
            config = self._get_config(tools=tools)
            if current_instruction:
                 config.system_instruction = current_instruction

            messages = self.get_messages(input, message_history)
            
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=messages,
                config=config
            )
            return self._parse_tool_response(response)
        except Exception as e:
             raise LLMGenerationError(f"Error calling GeminiLLM with tools: {e}") from e

    def _parse_tool_response(self, response: Any) -> ToolCallResponse:
        # Extract function calls from the new response object structure
        tool_calls = []
        if response.function_calls:
            for fc in response.function_calls:
                tool_calls.append(ToolCall(
                    name=fc.name,
                    arguments=fc.args
                ))
                
        return ToolCallResponse(
            tool_calls=tool_calls,
            content=None,
        )