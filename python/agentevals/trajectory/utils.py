__all__ = [
    "_is_trajectory_superset",
    "_extract_tool_calls",
    "_get_matcher_for_tool_name",
    "_normalize_to_openai_messages_list",
    "_convert_to_openai_message",
]

import json

from agentevals.types import (
    ChatCompletionMessage,
    ToolArgsMatchMode,
    ToolArgsMatchOverrides,
)
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import convert_to_openai_messages
from typing import Callable, Optional, Union


# More flexible version of converting to OpenAI messages for trajectories
def _convert_to_openai_message(
    message: Union[ChatCompletionMessage, BaseMessage, dict],
) -> ChatCompletionMessage:
    if not isinstance(message, BaseMessage):
        if not isinstance(message, dict):
            message = dict(message)
        if message.get("role") in ["ai", "assistant"] and message.get("tool_calls"):
            message["tool_calls"] = [
                {**tool_call, "id": tool_call.get("id", "")}
                for tool_call in message["tool_calls"]
            ]
        if message.get("role") == "tool" and message.get("tool_call_id") is None:
            message["tool_call_id"] = ""
        if message.get("content") is None:
            message["content"] = ""
    converted = convert_to_openai_messages([message])[0]  # type: ignore
    if isinstance(message, BaseMessage):
        if message.id is not None and converted.get("id") is None:
            converted["id"] = message.id
    else:
        if message.get("id") is not None and converted.get("id") is None:
            converted["id"] = message.get("id")
    return converted  # type: ignore


def _normalize_to_openai_messages_list(
    messages: Optional[
        Union[
            list[ChatCompletionMessage], list[BaseMessage], ChatCompletionMessage, dict
        ]
    ],
) -> list[ChatCompletionMessage]:
    if messages is None:
        return []
    if isinstance(messages, dict):
        if "role" in messages:
            messages = [messages]  # type: ignore
        elif "messages" in messages:
            messages = messages["messages"]  # type: ignore
        else:
            raise ValueError("if messages is a dict, it must contain a 'messages' key")
    if not isinstance(messages, list):
        messages = [messages]  # type: ignore
    return [_convert_to_openai_message(message) for message in messages]  # type: ignore


def _normalize_tool_call(tool_call: dict) -> dict:
    if "function" in tool_call:
        return {
            "name": tool_call["function"]["name"],
            "args": json.loads(tool_call["function"]["arguments"]),
        }
    else:
        return tool_call


def _extract_tool_calls(messages: list[ChatCompletionMessage]) -> list[dict]:
    tool_calls: list[dict] = []
    for message in messages:
        if "tool_calls" in message:
            normalized_tool_calls = [
                _normalize_tool_call(tool_call)
                for tool_call in message["tool_calls"] or []
            ]
            tool_calls.extend(normalized_tool_calls)
    return tool_calls


def _is_trajectory_superset(
    outputs: list[ChatCompletionMessage],
    reference_outputs: list[ChatCompletionMessage],
    tool_args_match_mode: ToolArgsMatchMode,
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides] = None,
):
    output_tool_calls = _extract_tool_calls(outputs)
    reference_tool_calls = _extract_tool_calls(reference_outputs)

    # Match reference calls to output calls as a bipartite matching: an
    # output call can cover a reference call when the names are equal and
    # the args match under the matcher. Greedy first-fit assignment is
    # order-dependent (an early reference call can claim the only output
    # call a later reference call could use), so find a maximum matching
    # with augmenting paths (Kuhn's algorithm) instead.
    # Maps output call index -> reference call index it is matched to.
    matched_output_calls: dict[int, int] = {}

    def try_assign(ref_idx: int, seen: set[int]) -> bool:
        ref_call = reference_tool_calls[ref_idx]
        matcher = _get_matcher_for_tool_name(
            ref_call["name"], tool_args_match_mode, tool_args_match_overrides
        )
        for out_idx, out_call in enumerate(output_tool_calls):
            if out_call["name"] != ref_call["name"] or out_idx in seen:
                continue
            if not matcher(out_call["args"], ref_call["args"]):
                continue
            seen.add(out_idx)
            if out_idx not in matched_output_calls or try_assign(
                matched_output_calls[out_idx], seen
            ):
                matched_output_calls[out_idx] = ref_idx
                return True
        return False

    for ref_idx in range(len(reference_tool_calls)):
        if not try_assign(ref_idx, set()):
            return False

    return True


def _exact_match(tool_call: dict, reference_tool_call: dict) -> bool:
    return tool_call == reference_tool_call


def _subset_match(tool_call: dict, reference_tool_call: dict) -> bool:
    # Every key-value pair in tool_call must exist in reference_tool_call
    return all(
        key in reference_tool_call and reference_tool_call[key] == value
        for key, value in tool_call.items()
    )


def _superset_match(tool_call: dict, reference_tool_call: dict) -> bool:
    # Every key-value pair in reference_tool_call must exist in tool_call
    return all(
        key in tool_call and tool_call[key] == value
        for key, value in reference_tool_call.items()
    )


def _ignore_match(tool_call: dict, reference_tool_call: dict) -> bool:
    return True


def _get_matcher_for_comparison_mode(
    mode: ToolArgsMatchMode,
) -> Callable[[dict, dict], bool]:
    if mode == "exact":
        return _exact_match
    elif mode == "subset":
        return _subset_match
    elif mode == "superset":
        return _superset_match
    else:
        return _ignore_match


def _get_partial_matcher_on_keys(keys: list[str]) -> Callable[[dict, dict], bool]:
    def get_nested_value(d: dict, key_path: str):
        current = d
        for part in key_path.split("."):
            if not isinstance(current, dict):
                return None
            current = current.get(part)  # type: ignore
            if current is None:
                return None
        return current

    def matcher(output_call: dict, reference_call: dict) -> bool:
        return all(
            get_nested_value(output_call, key) == get_nested_value(reference_call, key)
            for key in keys
        )

    return matcher


def _get_matcher_for_tool_name(
    tool_call_name: str,
    tool_args_match_mode: ToolArgsMatchMode,
    tool_args_match_overrides: Optional[ToolArgsMatchOverrides],
) -> Callable[[dict, dict], bool]:
    matcher = _get_matcher_for_comparison_mode(tool_args_match_mode)
    if tool_args_match_overrides is not None and tool_args_match_overrides.get(
        tool_call_name, False
    ):
        override = tool_args_match_overrides.get(tool_call_name)
        if isinstance(override, str):
            matcher = _get_matcher_for_comparison_mode(override)
        elif callable(override):
            matcher = override
        elif isinstance(override, list):
            matcher = _get_partial_matcher_on_keys(override)
        else:
            raise ValueError(f"Invalid tool args match override: {override}")
    return matcher
