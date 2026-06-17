# This file is part of the LLM4AD project (https://github.com/Optima-CityU/llm4ad).
# Last Revision: 2025/2/16
#
# ------------------------------- Copyright --------------------------------
# Copyright (c) 2025 Optima Group.
# 
# Permission is granted to use the LLM4AD platform for research purposes. 
# All publications, software, or other works that utilize this platform 
# or any part of its codebase must acknowledge the use of "LLM4AD" and 
# cite the following reference:
# 
# Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, 
# Zhichao Lu, and Qingfu Zhang, "LLM4AD: A Platform for Algorithm Design 
# with Large Language Model," arXiv preprint arXiv:2412.17287 (2024).
# 
# For inquiries regarding commercial use or licensing, please contact 
# http://www.llm4ad.com/contact.html
# --------------------------------------------------------------------------

from __future__ import annotations

import http.client
import json
import socket
import time
from typing import Any
import traceback
from urllib.parse import urlparse
from ...base import LLM


class LLMApiError(RuntimeError):
    """Raised when the LLM API cannot return a valid completion after retries."""


_NON_RETRYABLE_HTTP_STATUS = {400, 401, 403, 404}


def _normalize_base_url(host: str) -> tuple[str, str]:
    """Validate and normalize the host/base URL accepted by HTTPSConnection."""
    if host is None:
        raise ValueError("HttpsApi host must not be None. Use a host like 'api.moonshot.cn'.")

    base_url = str(host).strip()
    if not base_url:
        raise ValueError("HttpsApi host must not be empty. Use a host like 'api.moonshot.cn'.")
    if any(char.isspace() for char in base_url):
        raise ValueError(f"HttpsApi host must not contain whitespace. Got {host!r}.")

    if '://' in base_url:
        parsed = urlparse(base_url)
        if parsed.scheme != 'https':
            raise ValueError(
                f"HttpsApi host must use HTTPS. Got {host!r}; use a host like 'api.moonshot.cn'.")
        normalized_host = parsed.netloc
        base_path = parsed.path
    else:
        normalized_host, separator, base_path = base_url.partition('/')
        if separator:
            base_path = '/' + base_path

    if not normalized_host:
        raise ValueError(f"HttpsApi host must include a domain. Got {host!r}.")

    base_path = base_path.rstrip('/') or '/v1'
    if base_path.endswith('/chat/completions'):
        request_path = base_path
    else:
        request_path = f'{base_path}/chat/completions'
    return normalized_host, request_path


class HttpsApi(LLM):
    def __init__(self, host, key, model, timeout=60, max_retries=5, retry_delay=2, **kwargs):
        """Https API
        Args:
            host   : host name or HTTPS base URL, such as 'api.moonshot.ai' or
                     'https://api.kimi.com/coding/v1'.
            key    : API key.
            model  : LLM model name.
            timeout: API timeout.
            max_retries: Maximum retry attempts before raising the last API error.
            retry_delay: Seconds to wait between retry attempts.
        """
        super().__init__(**kwargs)
        self._host, self._request_path = _normalize_base_url(host)
        self._key = key
        self._model = model
        self._timeout = timeout
        self._max_retries = max(1, int(max_retries))
        self._retry_delay = retry_delay
        self._kwargs = kwargs
        self._cumulative_error = 0

    def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
        """
        Sends a request to the LLM and retrieves the generated response.

        This method supports multiple input formats for backward compatibility:
        1. Explicit 'messages' list via kwargs.
        2. A message list passed directly as the 'prompt'.
        3. Multimodal inputs (text + base64 images).
        4. Simple string prompts.

        Args:
            prompt: The text prompt or a list of message dictionaries.
            **kwargs: Can include 'image64s' (list of base64 strings) or 'messages'.

        Returns:
            The string content of the LLM's response.
        """
        image64s = kwargs.get('image64s', None)  # List[str]
        messages_input = kwargs.get('messages', None)

        # --- 1. Priority: Explicit messages list ---
        if messages_input is not None:
            if isinstance(messages_input, dict):
                messages = [messages_input]
            else:
                messages = messages_input

        # --- 2. Legacy Support: prompt passed as a pre-constructed list ---
        elif not isinstance(prompt, str):
            messages = prompt

        # --- 3. Construction from String + Optional Images ---
        else:
            text_content = prompt.strip()

            if image64s:
                # Construct multimodal content structure
                content = [{
                    "type": "text",
                    "text": text_content
                }]
                for image in image64s:
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image}",
                        }
                    })
                messages = [{'role': 'user', 'content': content}]

            else:
                # Construct standard text-only message
                messages = [{'role': 'user', 'content': text_content}]

        # Retry loop for handling network or API transient errors
        for attempt in range(1, self._max_retries + 1):
            try:
                conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)

                # Prepare standard OpenAI-compatible payload
                payload = json.dumps({
                    'max_tokens': self._kwargs.get('max_tokens', 8192),
                    'top_p': self._kwargs.get('top_p', None),
                    'temperature': self._kwargs.get('temperature', 1.0),
                    'model': self._model,
                    'messages': messages
                })
                headers = {
                    'Authorization': f'Bearer {self._key}',
                    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                    'Content-Type': 'application/json'
                }
                conn.request('POST', self._request_path, payload, headers)
                res = conn.getresponse()
                raw_data = res.read().decode('utf-8', errors='replace')
                if res.status in _NON_RETRYABLE_HTTP_STATUS:
                    body_preview = raw_data[:300].replace('\n', ' ').replace('\r', ' ')
                    if not body_preview:
                        body_preview = '<empty body>'
                    raise LLMApiError(
                        f'Non-retryable HTTP response from {self._host} '
                        f'(HTTP {res.status} {res.reason}): {body_preview}'
                    )
                try:
                    data = json.loads(raw_data)
                except json.JSONDecodeError as parse_error:
                    body_preview = raw_data[:300].replace('\n', ' ').replace('\r', ' ')
                    if not body_preview:
                        body_preview = '<empty body>'
                    raise RuntimeError(
                        f'Non-JSON response from {self._host} '
                        f'(HTTP {res.status} {res.reason}): {body_preview}'
                    ) from parse_error

                if 'error' in data:
                    api_error = data['error']
                    if isinstance(api_error, dict):
                        message = api_error.get('message', str(api_error))
                        error_type = api_error.get('type')
                        detail = f'{message} ({error_type})' if error_type else message
                    else:
                        detail = str(api_error)
                    raise RuntimeError(f'LLM API error from {self._host}: {detail}')

                # Extract content from the standard response format
                try:
                    response = data['choices'][0]['message']['content']
                except (KeyError, IndexError, TypeError) as parse_error:
                    raise RuntimeError(f'Unexpected LLM API response from {self._host}: {data}') from parse_error
                # Reset error counter on success
                self._cumulative_error = 0
                return response

            except socket.gaierror as e:
                raise LLMApiError(
                    f'Cannot resolve API host {self._host!r}. '
                    f"Use a valid API host or HTTPS base URL, for example 'api.moonshot.ai' "
                    f"or 'https://api.kimi.com/coding/v1'. "
                    f'Original error: {e}'
                ) from e

            except LLMApiError:
                raise

            except Exception as e:
                self._cumulative_error += 1

                if attempt >= self._max_retries:
                    raise LLMApiError(
                        f'{self.__class__.__name__} failed after {self._max_retries} attempts. '
                        f'Please check your API host, API key, model, quota, and provider status. '
                        f'Last error: {e}'
                    ) from e

                if self.debug_mode:
                    print(f'{self.__class__.__name__} attempt {attempt}/{self._max_retries} failed: '
                          f'{traceback.format_exc()}')
                else:
                    print(f'{self.__class__.__name__} attempt {attempt}/{self._max_retries} failed: {e}')
                time.sleep(self._retry_delay)

    # def draw_sample(self, prompt: str | Any, *args, **kwargs) -> str:
    #     """
    #     Handle message construction:
    #     - If 'messages' is explicitly provided, use it as the payload.
    #     - If 'messages' is None, build it from 'prompt' and 'images':
    #         a) Text only: Wrap prompt in a standard user message format.
    #         b) Multimodal: Combine prompt text and image URLs into a single user message content list.
    #     """
    #     image64s = kwargs.get('image64s', None)  # List[str]
    #     messages_input = kwargs.get('messages', None)   # messages
    #
    #     if messages_input is not None:
    #         if isinstance(messages_input, dict):
    #             messages = [messages_input]  # 单消息包装为列表
    #         else:
    #             messages = messages_input
    #     else:
    #         content = []
    #         content.append({
    #                 "type": "text",
    #                 "text": prompt.strip()
    #             })
    #
    #         if image64s is not None:
    #             for image in image64s:
    #                 content.append({
    #                     "type": "image_url",
    #                     "image_url": {
    #                         "url": f"data:image/png;base64,{image}",
    #                     }
    #                 })
    #
    #         messages = [{
    #             'role': 'user',
    #             'content': content
    #         }]
    #
    #     while True:
    #         try:
    #             conn = http.client.HTTPSConnection(self._host, timeout=self._timeout)
    #             payload = json.dumps({
    #                 'max_tokens': self._kwargs.get('max_tokens', 8192),
    #                 'top_p': self._kwargs.get('top_p', None),
    #                 'temperature': self._kwargs.get('temperature', 1.0),
    #                 'model': self._model,
    #                 'messages': messages
    #             })
    #             headers = {
    #                 'Authorization': f'Bearer {self._key}',
    #                 'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    #                 'Content-Type': 'application/json'
    #             }
    #             conn.request('POST', '/v1/chat/completions', payload, headers)
    #             res = conn.getresponse()
    #             data = res.read().decode('utf-8')
    #             data = json.loads(data)
    #             # print(data)
    #             response = data['choices'][0]['message']['content']
    #             if self.debug_mode:
    #                 self._cumulative_error = 0
    #             return response
    #         except Exception as e:
    #             self._cumulative_error += 1
    #             if self.debug_mode:
    #                 if self._cumulative_error == 10:
    #                     raise RuntimeError(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
    #                                        f'You may check your API host and API key.')
    #             else:
    #                 print(f'{self.__class__.__name__} error: {traceback.format_exc()}.'
    #                       f'You may check your API host and API key.')
    #                 time.sleep(2)
    #             continue
