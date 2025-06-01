"""
Utility functions for the freelancer analyzer.

This module contains shared utility functions used across different components
of the freelancer analyzer system.
"""

import json
from typing import Any, Dict, List, Union
import numpy as np


def serialize_for_json(obj: Any) -> Any:
    """
    Custom serializer to handle numpy types and other non-JSON serializable types.

    Args:
        obj: Object to serialize

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def convert_for_json_display(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """
    Recursively convert data structure to be JSON serializable.

    This function handles nested dictionaries and lists, converting
    all numpy types and other non-JSON types to serializable formats.

    Args:
        data: Data structure to convert

    Returns:
        JSON-serializable version of the data structure
    """
    if isinstance(data, dict):
        return {k: convert_for_json_display(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_for_json_display(item) for item in data]
    else:
        return serialize_for_json(data)


def format_data_as_json(data: Union[Dict, List, Any], indent: int = 2) -> str:
    """
    Convert data to a formatted JSON string.

    Args:
        data: Data to convert to JSON
        indent: Number of spaces for indentation

    Returns:
        Formatted JSON string
    """
    converted_data = convert_for_json_display(data)
    return json.dumps(converted_data, indent=indent, ensure_ascii=False)
