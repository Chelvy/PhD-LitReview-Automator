"""
Google Docs API integration for maintaining living PhD documents.
Supports: append text, insert at section, export to DOCX/PDF.
Auth: Service Account (for server-side automation) or OAuth2 (for personal use).
Docs: https://developers.google.com/docs/api/
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import structlog

from config.settings import settings

logger = structlog.get_logger(__name__)


def _get_docs_service():
    """Build Google Docs API service with service account or OAuth2."""
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        SCOPES = [
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/drive",
        ]

        # Try service account first (preferred for automation)
        sa_file = settings.google_service_account_file
        if os.path.exists(sa_file):
            creds = service_account.Credentials.from_service_account_file(
                sa_file, scopes=SCOPES
            )
            return build("docs", "v1", credentials=creds)

        # Fall back to OAuth2
        token_file = settings.google_token_file
        creds_file = settings.google_oauth_credentials_file
        creds = None

        if os.path.exists(token_file):
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            elif os.path.exists(creds_file):
                flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
                creds = flow.run_local_server(port=0)
            else:
                raise FileNotFoundError(
                    "No Google credentials found. Set GOOGLE_SERVICE_ACCOUNT_FILE or "
                    "GOOGLE_OAUTH_CREDENTIALS_FILE in .env"
                )

            with open(token_file, "w") as f:
                f.write(creds.to_json())

        return build("docs", "v1", credentials=creds)

    except ImportError:
        raise ImportError(
            "Google API libraries not installed. Run: "
            "pip install google-api-python-client google-auth-oauthlib"
        )


def _get_drive_service():
    """Build Google Drive API service (same auth as Docs)."""
    try:
        from google.oauth2 import service_account
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        SCOPES = ["https://www.googleapis.com/auth/drive"]
        sa_file = settings.google_service_account_file
        token_file = settings.google_token_file

        if os.path.exists(sa_file):
            creds = service_account.Credentials.from_service_account_file(sa_file, scopes=SCOPES)
        elif os.path.exists(token_file):
            from google.oauth2.credentials import Credentials
            creds = Credentials.from_authorized_user_file(token_file, SCOPES)
            if creds.expired:
                creds.refresh(Request())
        else:
            raise FileNotFoundError("No Google credentials found")

        return build("drive", "v3", credentials=creds)
    except ImportError:
        raise ImportError("Google API libraries not installed.")


class GoogleDocsTool:
    """
    Tool for reading from and writing to Google Docs documents.
    Maintains the Literature Review and Research Proposal as living documents.
    """

    def __init__(self) -> None:
        self.enabled = settings.has_google_docs
        self._service = None
        self._drive = None
        if not self.enabled:
            logger.warning("google_docs_disabled", reason="No Google credentials configured")

    @property
    def service(self):
        if self._service is None:
            self._service = _get_docs_service()
        return self._service

    @property
    def drive(self):
        if self._drive is None:
            self._drive = _get_drive_service()
        return self._drive

    def get_document(self, doc_id: str) -> Optional[dict[str, Any]]:
        """Fetch full document content."""
        if not self.enabled:
            return None
        try:
            return self.service.documents().get(documentId=doc_id).execute()
        except Exception as e:
            logger.error("google_docs_get_failed", doc_id=doc_id, error=str(e))
            return None

    def get_document_text(self, doc_id: str) -> str:
        """Extract plain text from a Google Doc."""
        doc = self.get_document(doc_id)
        if not doc:
            return ""

        texts = []
        for elem in doc.get("body", {}).get("content", []):
            paragraph = elem.get("paragraph")
            if paragraph:
                for pe in paragraph.get("elements", []):
                    tr = pe.get("textRun")
                    if tr:
                        texts.append(tr.get("content", ""))
        return "".join(texts)

    def append_text(
        self,
        doc_id: str,
        text: str,
        heading_level: Optional[int] = None,
    ) -> bool:
        """
        Append text to the end of a Google Doc.
        heading_level: None=normal, 1=H1, 2=H2, 3=H3
        """
        if not self.enabled:
            return False

        try:
            # Get current end index
            doc = self.get_document(doc_id)
            if not doc:
                return False

            content = doc.get("body", {}).get("content", [])
            end_index = content[-1].get("endIndex", 1) - 1 if content else 1

            requests = []

            # Insert text
            requests.append({
                "insertText": {
                    "location": {"index": end_index},
                    "text": "\n" + text,
                }
            })

            # Apply heading style if requested
            if heading_level:
                style_map = {1: "HEADING_1", 2: "HEADING_2", 3: "HEADING_3"}
                style = style_map.get(heading_level, "NORMAL_TEXT")
                requests.append({
                    "updateParagraphStyle": {
                        "range": {
                            "startIndex": end_index,
                            "endIndex": end_index + len(text) + 1,
                        },
                        "paragraphStyle": {"namedStyleType": style},
                        "fields": "namedStyleType",
                    }
                })

            self.service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": requests}
            ).execute()

            logger.info("google_docs_appended", doc_id=doc_id, text_len=len(text))
            return True

        except Exception as e:
            logger.error("google_docs_append_failed", doc_id=doc_id, error=str(e))
            return False

    def insert_at_section(
        self,
        doc_id: str,
        section_heading: str,
        text_to_insert: str,
        after: bool = True,
    ) -> bool:
        """
        Find a section heading and insert text after/before it.
        Falls back to appending if section not found.
        """
        if not self.enabled:
            return False

        try:
            doc = self.get_document(doc_id)
            if not doc:
                return False

            content = doc.get("body", {}).get("content", [])
            insert_index = None

            for elem in content:
                paragraph = elem.get("paragraph", {})
                for pe in paragraph.get("elements", []):
                    tr = pe.get("textRun", {})
                    if section_heading.lower() in tr.get("content", "").lower():
                        if after:
                            insert_index = elem.get("endIndex", 1)
                        else:
                            insert_index = elem.get("startIndex", 1)
                        break
                if insert_index:
                    break

            if insert_index is None:
                logger.warning(
                    "google_docs_section_not_found",
                    section=section_heading,
                    doc_id=doc_id,
                    action="appending_to_end"
                )
                return self.append_text(doc_id, text_to_insert)

            self.service.documents().batchUpdate(
                documentId=doc_id,
                body={"requests": [{"insertText": {
                    "location": {"index": insert_index},
                    "text": "\n" + text_to_insert,
                }}]}
            ).execute()

            return True

        except Exception as e:
            logger.error("google_docs_insert_failed", doc_id=doc_id, error=str(e))
            return False

    def export_as_docx(self, doc_id: str, output_path: str) -> bool:
        """Export a Google Doc as .docx file."""
        if not self.enabled:
            return False
        try:
            response = self.drive.files().export(
                fileId=doc_id,
                mimeType="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ).execute()
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(response)
            logger.info("google_docs_exported", doc_id=doc_id, path=output_path)
            return True
        except Exception as e:
            logger.error("google_docs_export_failed", doc_id=doc_id, error=str(e))
            return False

    def save_markdown_locally(
        self,
        content: str,
        filename: str,
        directory: Optional[str] = None,
    ) -> str:
        """
        Save document content as a local Markdown file.
        Used as fallback when Google Docs is not available,
        and as the primary version-controlled copy.
        """
        out_dir = Path(directory or settings.outputs_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        out_path.write_text(content, encoding="utf-8")
        logger.info("markdown_saved", path=str(out_path))
        return str(out_path)
