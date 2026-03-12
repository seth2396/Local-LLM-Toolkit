from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Optional

from .BaseIngester import BaseIngester
from .BaseItem import BaseItem

# TODO: SharePointLoader
# SharePointItem does not have a local file path, so existing file-based loaders
# (PdfLoader, DocxLoader, etc.) cannot load it directly.
# A dedicated SharePointLoader needs to be implemented in loaders/ that:
#   1. Authenticates with the Microsoft Graph API (reusing token logic from SharePointIngester)
#   2. Downloads the file content as bytes using the item's drive_id and item_id:
#      GET /drives/{drive_id}/items/{item_id}/content
#   3. Writes the bytes to a temporary file (tempfile.NamedTemporaryFile)
#   4. Delegates to the appropriate file-based loader based on item.ext
#   5. Cleans up the temp file after loading


@dataclass(frozen=True)
class SharePointItem(BaseItem):
    """
    Structured representation of a discovered SharePoint document.

    Attributes:
        name:        File name including extension.
        source:      Full web URL to the item in SharePoint.
        doctype:     Logical type derived from the file extension (e.g., 'docx', 'pdf').
        item_id:     Unique SharePoint item ID.
        site_url:    URL of the SharePoint site the item belongs to.
        drive_id:    ID of the document library drive.
        ext:         File extension including the dot (e.g., '.docx').
        size_bytes:  File size in bytes.
        modified_ts: Last modified timestamp (Unix epoch seconds).
        created_by:  Display name of the creator, if available.
    """
    name: str
    source: str
    doctype: str
    item_id: str
    site_url: str
    drive_id: str
    ext: str
    size_bytes: Optional[int] = None
    modified_ts: Optional[float] = None
    created_by: Optional[str] = None

    def to_metadata(self) -> dict:
        return {k: v for k, v in {
            "name": self.name,
            "source": self.source,
            "doctype": self.doctype,
            "item_id": self.item_id,
            "site_url": self.site_url,
            "drive_id": self.drive_id,
            "ext": self.ext,
            "size_bytes": self.size_bytes,
            "modified_ts": self.modified_ts,
            "created_by": self.created_by,
        }.items() if v is not None}


class SharePointIngester(BaseIngester):
    """
    Ingests documents from a SharePoint site via the Microsoft Graph API.

    Authenticates using client credentials (app registration with Sites.Read.All
    permission) and retrieves all files in the target document library using
    the /delta endpoint for efficient full enumeration.

    Args:
        site_url:      URL of the SharePoint site (e.g., 'https://tenant.sharepoint.com/sites/mysite').
        tenant_id:     Azure AD tenant ID.
        client_id:     App registration client ID.
        client_secret: App registration client secret.
        drive_name:    Name of the document library to ingest. Defaults to 'Documents'.
        ext_filter:    If provided, only return files with these extensions (e.g., {'.pdf', '.docx'}).
    """

    GRAPH_BASE = "https://graph.microsoft.com/v1.0"

    def __init__(
        self,
        site_url: str,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        drive_name: str = "Documents",
        ext_filter: Optional[set[str]] = None,
    ):
        self.site_url = site_url
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.drive_name = drive_name
        self.ext_filter = {e.lower() for e in ext_filter} if ext_filter else None

    def collect(self) -> list[SharePointItem]:
        """
        Authenticate with Microsoft Graph and return all matching documents.

        Returns:
            List of SharePointItem, one per discovered document.
        """
        token = self._get_token()
        site_id = self._get_site_id(token)
        drive_id = self._get_drive_id(token, site_id)
        return self._list_all_items(token, drive_id)

    def _get_token(self) -> str:
        """Acquire an OAuth2 access token using client credentials."""
        try:
            import msal
        except ImportError:
            raise ImportError("msal is required for SharePointIngester: pip install msal")

        app = msal.ConfidentialClientApplication(
            client_id=self.client_id,
            client_credential=self.client_secret,
            authority=f"https://login.microsoftonline.com/{self.tenant_id}",
        )
        result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
        if "access_token" not in result:
            raise RuntimeError(f"Failed to acquire token: {result.get('error_description')}")
        return result["access_token"]

    def _get_site_id(self, token: str) -> str:
        """Resolve the SharePoint site URL to a Graph site ID."""
        import requests
        parsed = urlparse(self.site_url)
        hostname = parsed.hostname
        site_path = parsed.path.strip("/")
        url = f"{self.GRAPH_BASE}/sites/{hostname}:/{site_path}"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        return resp.json()["id"]

    def _get_drive_id(self, token: str, site_id: str) -> str:
        """Find the drive ID for the named document library."""
        import requests
        url = f"{self.GRAPH_BASE}/sites/{site_id}/drives"
        resp = requests.get(url, headers={"Authorization": f"Bearer {token}"})
        resp.raise_for_status()
        for drive in resp.json().get("value", []):
            if drive.get("name") == self.drive_name:
                return drive["id"]
        raise ValueError(f"Drive '{self.drive_name}' not found in site '{self.site_url}'.")

    def _list_all_items(self, token: str, drive_id: str) -> list[SharePointItem]:
        """
        Fetch all files from the drive using the /delta endpoint.

        /delta returns all items in a single paginated response, avoiding
        the need for recursive folder traversal.
        """
        import requests
        headers = {"Authorization": f"Bearer {token}"}
        url = f"{self.GRAPH_BASE}/drives/{drive_id}/root/delta"
        items = []

        while url:
            resp = requests.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("value", []):
                if "folder" in item:
                    continue

                name = item.get("name", "")
                ext = Path(name).suffix.lower()

                if self.ext_filter and ext not in self.ext_filter:
                    continue

                modified_ts = None
                raw_modified = item.get("lastModifiedDateTime")
                if raw_modified:
                    dt = datetime.fromisoformat(raw_modified.replace("Z", "+00:00"))
                    modified_ts = dt.timestamp()

                items.append(SharePointItem(
                    name=name,
                    source=item.get("webUrl", ""),
                    doctype=ext.lstrip("."),
                    item_id=item["id"],
                    site_url=self.site_url,
                    drive_id=drive_id,
                    ext=ext,
                    size_bytes=item.get("size"),
                    modified_ts=modified_ts,
                    created_by=item.get("createdBy", {}).get("user", {}).get("displayName"),
                ))

            url = data.get("@odata.nextLink")

        return items
