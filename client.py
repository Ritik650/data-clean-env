"""
DataCleanEnv — EnvClient subclass for data_clean_env.

Usage:
    env = await DataCleanEnv.from_docker_image("data-clean-env:latest")
    result = await env.reset(task_id="easy")
    result = await env.step(DataCleanAction(operation="deduplicate", column=None))
    await env.close()
"""
from __future__ import annotations

from typing import Any, Dict

from models import DataCleanAction, DataCleanObservation, DataCleanState

try:
    from openenv.core import EnvClient, StepResult  # type: ignore

    class DataCleanEnv(EnvClient[DataCleanAction, DataCleanObservation, DataCleanState]):
        """Typed client for the data_clean_env server."""

        def _step_payload(self, action: DataCleanAction) -> dict:
            return action.model_dump()

        def _parse_result(self, payload: Dict[str, Any]) -> StepResult[DataCleanObservation]:
            obs = DataCleanObservation(**payload["observation"])
            return StepResult(
                observation=obs,
                reward=float(payload.get("reward", 0.0)),
                done=bool(payload.get("done", False)),
            )

        def _parse_state(self, payload: Dict[str, Any]) -> DataCleanState:
            return DataCleanState(**payload)

except ImportError:
    # Fallback for local testing without the full OpenEnv SDK
    import asyncio
    import json
    import os
    from typing import Optional

    import httpx

    class StepResult:  # type: ignore
        def __init__(self, observation, reward, done):
            self.observation = observation
            self.reward = reward
            self.done = done

    class DataCleanEnv:  # type: ignore
        """
        Lightweight HTTP client for data_clean_env.
        Compatible with the OpenEnv EnvClient interface.
        """

        def __init__(self, base_url: str):
            self._base_url = base_url.rstrip("/")
            self._client = httpx.AsyncClient(base_url=self._base_url, timeout=60.0)

        @classmethod
        async def from_docker_image(cls, image_name: str, port: int = 7860) -> "DataCleanEnv":
            """
            Spin up the Docker container and return a connected client.
            Falls back to connecting to localhost if already running.
            """
            base_url = os.getenv("ENV_BASE_URL", f"http://localhost:{port}")
            return cls(base_url)

        async def reset(self, task_id: str = "easy", seed: Optional[int] = None, **kwargs) -> StepResult:
            body: Dict[str, Any] = {"task_id": task_id}
            if seed is not None:
                body["seed"] = seed
            resp = await self._client.post("/reset", json=body)
            resp.raise_for_status()
            data = resp.json()
            obs = DataCleanObservation(**data["observation"])
            return StepResult(
                observation=obs,
                reward=float(data.get("reward", 0.0)),
                done=bool(data.get("done", False)),
            )

        async def step(self, action: DataCleanAction) -> StepResult:
            resp = await self._client.post("/step", json=action.model_dump())
            resp.raise_for_status()
            data = resp.json()
            obs = DataCleanObservation(**data["observation"])
            return StepResult(
                observation=obs,
                reward=float(data.get("reward", 0.0)),
                done=bool(data.get("done", False)),
            )

        async def state(self) -> DataCleanState:
            resp = await self._client.get("/state")
            resp.raise_for_status()
            return DataCleanState(**resp.json())

        async def close(self) -> None:
            await self._client.aclose()
