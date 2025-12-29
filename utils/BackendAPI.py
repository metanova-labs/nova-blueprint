from typing import Any, Dict, Optional
import json
import os

import aiohttp
import bittensor as bt


class BackendAPI:
    "Singleton helper class for communicating with the Backend API"

    _instance: Optional["BackendAPI"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BackendAPI, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._base_url = os.getenv("BACKEND_API_URL")
        self._api_token = os.getenv("BACKEND_API_TOKEN")
        self._session: Optional[aiohttp.ClientSession] = None
        if not self._base_url:
            bt.logging.warning("BACKEND_API_URL not set; API calls will fail.")
        if not self._api_token:
            bt.logging.warning("BACKEND_API_TOKEN not set; API calls will fail.")

        self._initialized = True

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        return headers

    async def _get_session(self):
        if self._session is None or self._session.closed:
            try:
                self._session = aiohttp.ClientSession(
                    base_url=self._base_url,
                    headers=self._get_headers(),
                    timeout=aiohttp.ClientTimeout(
                        total=30,  # total timeout
                        connect=10,  # connection timeout
                        sock_read=20,  # socket read timeout
                    ),
                )
            except Exception as e:
                bt.logging.error(f"Failed to create aiohttp session: {e}")
                raise
        return self._session

    async def submit_epoch_results(self, payload: Dict[str, Any]):
        if not self._base_url:
            bt.logging.error("BACKEND_API_URL not set; cannot submit results.")
            return False

        try:
            session = await self._get_session()
            async with session.post(
                "competitions/submit-epoch",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                # Read response body so we can log useful details on failure.
                raw_text = await response.text()
                try:
                    body: Any = json.loads(raw_text) if raw_text else None
                except Exception:
                    body = raw_text or None

                if response.status >= 400:
                    bt.logging.error(
                        f"Error submitting epoch results: "
                        f"{response.status} {response.reason}, "
                        f"body={str(body)[:500]}"
                    )
                    return False

                bt.logging.info(
                    f"Successfully submitted epoch results: status={response.status}"
                )
                return True
        except aiohttp.ClientError as e:
            bt.logging.error(f"Error submitting epoch results: {e}")
            return False

        except Exception as e:
            bt.logging.error(
                f"Unexpected error submitting epoch results: {e}", exc_info=True
            )
            return False

    async def close(self):
        if self._session and not self._session.closed:
            try:
                await self._session.close()
            except Exception as e:
                bt.logging.warning(f"Error closing session: {e}")
            finally:
                self._session = None
