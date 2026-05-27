"""Tests for collectors/cloudflare_kv.py — all HTTP calls are mocked."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from collectors.cloudflare_kv import delete_thread_ts_from_kv, sync_thread_ts_to_kv

_ACCOUNT_ID = "test-account-id"
_NAMESPACE_ID = "test-namespace-id"
_API_TOKEN = "test-api-token"
_BASE = "https://api.cloudflare.com/client/v4"


# ---------------------------------------------------------------------------
# sync_thread_ts_to_kv
# ---------------------------------------------------------------------------

class TestSyncThreadTsToKv:
    def test_correct_url(self):
        mock_resp = MagicMock()
        mock_resp.ok = True

        with patch("collectors.cloudflare_kv.requests.put", return_value=mock_resp) as mock_put:
            sync_thread_ts_to_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, ["1234567890.123456"])

        expected_url = f"{_BASE}/accounts/{_ACCOUNT_ID}/storage/kv/namespaces/{_NAMESPACE_ID}/bulk"
        mock_put.assert_called_once()
        actual_url = mock_put.call_args.args[0]
        assert actual_url == expected_url

    def test_correct_payload_shape(self):
        mock_resp = MagicMock()
        mock_resp.ok = True

        keys = ["1111111111.000001", "2222222222.000002"]
        with patch("collectors.cloudflare_kv.requests.put", return_value=mock_resp) as mock_put:
            sync_thread_ts_to_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, keys)

        call_kwargs = mock_put.call_args.kwargs
        payload = json.loads(call_kwargs["data"])
        assert payload == [{"key": "1111111111.000001", "value": "1"}, {"key": "2222222222.000002", "value": "1"}]

    def test_correct_auth_header(self):
        mock_resp = MagicMock()
        mock_resp.ok = True

        with patch("collectors.cloudflare_kv.requests.put", return_value=mock_resp) as mock_put:
            sync_thread_ts_to_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, ["ts-1"])

        headers = mock_put.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {_API_TOKEN}"
        assert headers["Content-Type"] == "application/json"

    def test_handles_request_failure_gracefully(self):
        """RequestException must not propagate — KV sync failure is non-fatal."""
        with patch(
            "collectors.cloudflare_kv.requests.put",
            side_effect=requests.RequestException("connection refused"),
        ):
            # Should not raise
            sync_thread_ts_to_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, ["ts-1"])

    def test_handles_non_ok_response_gracefully(self):
        """Non-2xx response must not raise."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 403
        mock_resp.text = "Forbidden"

        with patch("collectors.cloudflare_kv.requests.put", return_value=mock_resp):
            # Should not raise
            sync_thread_ts_to_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, ["ts-1"])

    def test_empty_list_makes_no_http_call(self):
        """Empty thread_ts list → no HTTP request at all."""
        with patch("collectors.cloudflare_kv.requests.put") as mock_put:
            sync_thread_ts_to_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, [])

        mock_put.assert_not_called()

    def test_single_key_payload(self):
        mock_resp = MagicMock()
        mock_resp.ok = True

        with patch("collectors.cloudflare_kv.requests.put", return_value=mock_resp) as mock_put:
            sync_thread_ts_to_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, ["only-key"])

        payload = json.loads(mock_put.call_args.kwargs["data"])
        assert len(payload) == 1
        assert payload[0] == {"key": "only-key", "value": "1"}


# ---------------------------------------------------------------------------
# delete_thread_ts_from_kv
# ---------------------------------------------------------------------------

class TestDeleteThreadTsFromKv:
    def test_correct_url(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200

        ts = "1234567890.123456"
        with patch("collectors.cloudflare_kv.requests.delete", return_value=mock_resp) as mock_del:
            delete_thread_ts_from_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, ts)

        expected_url = (
            f"{_BASE}/accounts/{_ACCOUNT_ID}/storage/kv/namespaces/{_NAMESPACE_ID}/values/{ts}"
        )
        mock_del.assert_called_once()
        actual_url = mock_del.call_args.args[0]
        assert actual_url == expected_url

    def test_correct_auth_header(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200

        with patch("collectors.cloudflare_kv.requests.delete", return_value=mock_resp) as mock_del:
            delete_thread_ts_from_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, "ts-1")

        headers = mock_del.call_args.kwargs["headers"]
        assert headers["Authorization"] == f"Bearer {_API_TOKEN}"

    def test_404_is_not_an_error(self):
        """404 means key was already gone — should be treated as success, not error."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 404

        with patch("collectors.cloudflare_kv.requests.delete", return_value=mock_resp):
            # Should not raise or log an error
            delete_thread_ts_from_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, "ts-gone")

    def test_non_404_error_response_handled_gracefully(self):
        """Non-404 error response must not raise."""
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("collectors.cloudflare_kv.requests.delete", return_value=mock_resp):
            # Should not raise
            delete_thread_ts_from_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, "ts-1")

    def test_handles_request_failure_gracefully(self):
        """RequestException must not propagate."""
        with patch(
            "collectors.cloudflare_kv.requests.delete",
            side_effect=requests.RequestException("timeout"),
        ):
            # Should not raise
            delete_thread_ts_from_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, "ts-1")

    def test_200_response_is_success(self):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200

        with patch("collectors.cloudflare_kv.requests.delete", return_value=mock_resp) as mock_del:
            delete_thread_ts_from_kv(_ACCOUNT_ID, _NAMESPACE_ID, _API_TOKEN, "ts-1")

        mock_del.assert_called_once()
