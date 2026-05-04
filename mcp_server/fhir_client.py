import httpx


class FhirClient:
    def __init__(self, base_url: str, token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token

    def _build_url(self, path: str) -> str:
        path = path.lstrip("/")
        return f"{self.base_url}/{path}"

    # ── MODIFIED: centralized headers with Content-Type for FHIR ──────────
    def _headers(self) -> dict:
        headers = {"Content-Type": "application/fhir+json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers
    # ─────────────────────────────────────────────────────────────────────

    async def _get(self, path: str, params: dict[str, str] | None = None) -> dict | None:
        url = self._build_url(path)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(
                    url,
                    headers=self._headers(),  # MODIFIED: use _headers()
                    params=params
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError:
                raise

    async def read(self, path: str) -> dict | None:
        return await self._get(path)

    async def search(
        self,
        resource_type: str,
        search_parameters: dict[str, str] | None = None,
    ) -> dict | None:
        return await self._get(resource_type, params=search_parameters)

    # ── ADDED: create method for FHIR write-back ──────────────────────────
    # Original FhirClient only had read() and search().
    # We added create() to support writing Condition resources back to FHIR.
    # Note: Currently returns 403 on Prompt Opinion platform — write access
    # is restricted for external MCP servers. Implementation is correct
    # and works on FHIR servers with write permissions.
    async def create(self, resource_type: str, resource: dict) -> dict | None:
        """Create a new FHIR resource via POST"""
        url = self._build_url(resource_type)
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    headers=self._headers(),
                    json=resource
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as e:
                print(f"FHIR create error: {e.response.status_code} — {e.response.text}")
                raise
    # ─────────────────────────────────────────────────────────────────────