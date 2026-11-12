"""Policy backed by an OpenAI-compatible chat-completions endpoint (e.g. vLLM serving a MatAct checkpoint).

Example server for a trained Qwen3 checkpoint:

    vllm serve outputs/qwen3-8b-matact --served-model-name matact \
        --enable-auto-tool-choice --tool-call-parser hermes

The canonical trajectory text (``<tool_call>`` blocks) is converted to OpenAI ``tool_calls`` /
``tool`` messages for the request, and tool calls in the response are converted back, so that
parsing, logging and training data use one format.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from ..env.tools import openai_schema
from ..protocol import format_call, parse
from .policies import Policy


class OpenAIChatPolicy(Policy):
    name = "openai"

    def __init__(self, model, base_url="http://localhost:8000/v1", api_key=None, temperature=0.0, top_p=1.0,
                 max_tokens=1024, timeout=300, use_tools_param=True, sanitize_names=False, seed=None,
                 chat_template_kwargs=None):
        self.model, self.base_url = model, base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.temperature, self.top_p, self.max_tokens, self.timeout = temperature, top_p, max_tokens, timeout
        self.use_tools_param, self.sanitize_names, self.seed = use_tools_param, sanitize_names, seed
        self.chat_template_kwargs = chat_template_kwargs
        self.name = f"openai:{model}"
        self.tools = self._sanitize(openai_schema())

    def _tool_name(self, name: str) -> str:
        # some APIs only accept [a-zA-Z0-9_-] tool names
        return name.replace(".", "__") if self.sanitize_names else name

    def _sanitize(self, tools):
        for tool in tools:
            tool["function"]["name"] = self._tool_name(tool["function"]["name"])
        return tools

    def _messages(self, messages):
        out, call_id = [], None
        for m in messages:
            if m["role"] == "assistant":
                parsed = parse(m["content"])
                if parsed.kind == "call" and self.use_tools_param:
                    call_id = f"call_{len(out)}"
                    out.append({"role": "assistant", "content": parsed.rationale or None, "tool_calls": [{
                        "id": call_id, "type": "function",
                        "function": {"name": self._tool_name(parsed.action["name"]),
                                     "arguments": json.dumps(parsed.action["arguments"], ensure_ascii=False)}}]})
                else:
                    call_id = None
                    out.append({"role": "assistant", "content": m["content"]})
            elif m["role"] == "tool":
                if call_id:
                    out.append({"role": "tool", "tool_call_id": call_id, "content": m["content"]})
                else:  # observation of an unparsable turn: there is no tool call to answer
                    out.append({"role": "user", "content": f"<observation>\n{m['content']}\n</observation>"})
                call_id = None
            else:
                out.append({"role": m["role"], "content": m["content"]})
        return out

    def act(self, messages, observation):
        payload = {"model": self.model, "messages": self._messages(messages), "temperature": self.temperature,
                   "top_p": self.top_p, "max_tokens": self.max_tokens}
        if self.use_tools_param:
            payload["tools"] = self.tools
        if self.seed is not None:
            payload["seed"] = self.seed
        if self.chat_template_kwargs:
            payload["chat_template_kwargs"] = self.chat_template_kwargs
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.base_url + "/chat/completions", data=json.dumps(payload).encode(),
                                         headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"chat completion failed ({exc.code}): {exc.read().decode(errors='replace')[:500]}") from exc
        message = data["choices"][0]["message"]
        text = message.get("content") or ""
        for call in message.get("tool_calls") or []:
            fn = call["function"]
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"__unparsed__": args}
            text += "\n" + format_call(fn["name"].replace("__", "."), args)
        return text.strip()
