"""Cloudflare KV namespace sync helpers."""
import json
import logging
import requests

_log = logging.getLogger(__name__)
_BASE = "https://api.cloudflare.com/client/v4"


def sync_thread_ts_to_kv(
    account_id: str,
    namespace_id: str,
    api_token: str,
    thread_ts_keys: list[str],
) -> None:
    """Bulk-write all active thread_ts keys to KV so the worker can validate them.

    Each key is written with value "1". Uses the Cloudflare KV bulk write endpoint.
    Logs errors but does not raise — KV sync failure should not abort the main job.
    """
    if not thread_ts_keys:
        _log.debug("No thread_ts keys to sync to KV")
        return

    url = f"{_BASE}/accounts/{account_id}/storage/kv/namespaces/{namespace_id}/bulk"
    payload = [{"key": ts, "value": "1"} for ts in thread_ts_keys]
    headers = {"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"}

    try:
        resp = requests.put(url, headers=headers, data=json.dumps(payload), timeout=15)
        if not resp.ok:
            _log.error("KV bulk write failed: %s %s", resp.status_code, resp.text)
        else:
            _log.info("Synced %d thread_ts key(s) to Cloudflare KV", len(thread_ts_keys))
    except requests.RequestException as exc:
        _log.error("KV bulk write request failed: %s", exc)


def delete_thread_ts_from_kv(
    account_id: str,
    namespace_id: str,
    api_token: str,
    thread_ts: str,
) -> None:
    """Delete a single thread_ts key from KV after a reply is handled."""
    url = f"{_BASE}/accounts/{account_id}/storage/kv/namespaces/{namespace_id}/values/{thread_ts}"
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        resp = requests.delete(url, headers=headers, timeout=15)
        if not resp.ok and resp.status_code != 404:
            _log.error("KV delete failed for %s: %s %s", thread_ts, resp.status_code, resp.text)
        else:
            _log.info("Deleted thread_ts %s from Cloudflare KV", thread_ts)
    except requests.RequestException as exc:
        _log.error("KV delete request failed: %s", exc)
