"""
Google Drive MCP Tool

Tool for searching and retrieving files from Google Drive using OAuth2 authentication.
"""
import logging
import os
import io
from typing import Dict, Any, List, Optional
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError

from ..base import MCPTool, MCPToolSchema, MCPToolParameter
from ..exceptions import MCPExecutionError, MCPConnectionError

logger = logging.getLogger(__name__)

# Google Drive API scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


class GoogleDriveTool(MCPTool):
    """
    MCP tool for accessing Google Drive files.

    Supports:
    - Searching files by query
    - Downloading file content
    - Extracting text from Google Docs, plain text, and PDFs
    """

    def __init__(
        self,
        credentials_path: str = "./credentials/google_credentials.json",
        token_path: str = "./credentials/google_token.json"
    ):
        """
        Initialize Google Drive tool.

        Args:
            credentials_path: Path to OAuth2 credentials JSON file
            token_path: Path to store/load OAuth2 tokens

        Raises:
            MCPConnectionError: If credentials file doesn't exist
        """
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.service = None

        # Check credentials file exists
        if not os.path.exists(credentials_path):
            raise MCPConnectionError(
                server_name="google_drive",
                reason=f"Credentials file not found at {credentials_path}. "
                       f"See credentials/README.md for setup instructions."
            )

        # Initialize base class
        super().__init__()

        logger.info(f"Initialized GoogleDriveTool with credentials: {credentials_path}")

    def _create_schema(self) -> MCPToolSchema:
        """Create tool schema."""
        return MCPToolSchema(
            name="google_drive_search",
            description=(
                "Search and retrieve files from Google Drive. "
                "Supports text files, Google Docs, and PDFs. "
                "Returns file metadata and content for text-based files."
            ),
            parameters=[
                MCPToolParameter(
                    name="query",
                    type="string",
                    description=(
                        "Search query string. Examples: "
                        "'name contains \"report\"', "
                        "'modifiedTime > \"2024-01-01\"', "
                        "'fullText contains \"RAG\"'. "
                        "See Google Drive API query syntax for advanced options."
                    ),
                    required=True
                ),
                MCPToolParameter(
                    name="max_results",
                    type="integer",
                    description="Maximum number of files to return",
                    required=False,
                    default=10
                ),
                MCPToolParameter(
                    name="file_types",
                    type="array",
                    description=(
                        "Optional list of MIME types to filter. "
                        "Examples: ['text/plain', 'application/vnd.google-apps.document', 'application/pdf']"
                    ),
                    required=False,
                    default=None
                ),
            ],
            category="google_drive",
            version="1.0.0"
        )

    def _authenticate(self) -> Any:
        """
        Authenticate with Google Drive API using OAuth2.

        Returns:
            Google Drive service object

        Raises:
            MCPConnectionError: If authentication fails
        """
        if self.service:
            return self.service

        creds = None

        # Load existing token if available
        if os.path.exists(self.token_path):
            try:
                creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
                logger.debug(f"Loaded existing credentials from {self.token_path}")
            except Exception as e:
                logger.warning(f"Failed to load token: {e}. Will re-authenticate.")

        # If no valid credentials, authenticate
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    logger.info("Refreshing expired Google Drive token")
                    creds.refresh(Request())
                except Exception as e:
                    logger.error(f"Token refresh failed: {e}")
                    creds = None

            if not creds:
                try:
                    logger.info("Starting OAuth2 flow for Google Drive")
                    flow = InstalledAppFlow.from_client_secrets_file(
                        self.credentials_path, SCOPES
                    )
                    creds = flow.run_local_server(port=0)
                    logger.info("OAuth2 flow completed successfully")
                except Exception as e:
                    raise MCPConnectionError(
                        server_name="google_drive",
                        reason=f"OAuth2 authentication failed: {str(e)}"
                    )

            # Save the credentials for future use
            try:
                with open(self.token_path, 'w') as token:
                    token.write(creds.to_json())
                logger.info(f"Saved credentials to {self.token_path}")
            except Exception as e:
                logger.warning(f"Failed to save token: {e}")

        # Build the service
        try:
            self.service = build('drive', 'v3', credentials=creds)
            logger.info("Google Drive service initialized")
            return self.service
        except Exception as e:
            raise MCPConnectionError(
                server_name="google_drive",
                reason=f"Failed to build Drive service: {str(e)}"
            )

    def _search_files(
        self,
        query: str,
        max_results: int = 10,
        file_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for files in Google Drive.

        Args:
            query: Search query string
            max_results: Maximum number of results
            file_types: Optional list of MIME types to filter

        Returns:
            List of file metadata dictionaries

        Raises:
            MCPExecutionError: If search fails
        """
        service = self._authenticate()

        # Build query with MIME type filter
        full_query = query
        if file_types:
            mime_conditions = " or ".join([f"mimeType='{mt}'" for mt in file_types])
            full_query = f"({query}) and ({mime_conditions})"

        try:
            logger.info(f"Searching Google Drive: '{full_query}'")

            results = service.files().list(
                q=full_query,
                pageSize=max_results,
                fields="files(id, name, mimeType, modifiedTime, webViewLink, size)",
                orderBy="modifiedTime desc"
            ).execute()

            files = results.get('files', [])
            logger.info(f"Found {len(files)} files matching query")

            return files

        except HttpError as e:
            logger.error(f"Google Drive API error: {e}")
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"Google Drive search failed: {str(e)}",
                original_exception=e,
                parameters={"query": query, "max_results": max_results}
            )

    def _get_file_content(self, file_id: str, mime_type: str) -> Optional[str]:
        """
        Get text content from a file.

        Args:
            file_id: Google Drive file ID
            mime_type: File MIME type

        Returns:
            File content as string, or None if not text-based

        Raises:
            MCPExecutionError: If download fails
        """
        service = self._authenticate()

        try:
            # Handle different file types
            if mime_type == 'application/vnd.google-apps.document':
                # Export Google Doc as plain text
                logger.debug(f"Exporting Google Doc {file_id} as text")
                request = service.files().export_media(
                    fileId=file_id,
                    mimeType='text/plain'
                )
            elif mime_type in ['text/plain', 'text/markdown', 'text/csv']:
                # Download text file directly
                logger.debug(f"Downloading text file {file_id}")
                request = service.files().get_media(fileId=file_id)
            elif mime_type == 'application/pdf':
                # Note: PDF text extraction not implemented yet
                # Would require pypdf or similar library
                logger.warning(f"PDF text extraction not yet implemented for {file_id}")
                return None
            else:
                logger.debug(f"Unsupported MIME type for content extraction: {mime_type}")
                return None

            # Download content
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)

            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    logger.debug(f"Download progress: {int(status.progress() * 100)}%")

            # Decode content
            content = file_buffer.getvalue().decode('utf-8')
            logger.debug(f"Successfully retrieved content ({len(content)} chars)")

            return content

        except HttpError as e:
            logger.error(f"Failed to get file content for {file_id}: {e}")
            # Don't raise exception for content retrieval failures
            # Just log and return None
            return None

        except Exception as e:
            logger.error(f"Unexpected error getting content for {file_id}: {e}")
            return None

    def _execute(
        self,
        query: str,
        max_results: int = 10,
        file_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute Google Drive search and retrieval.

        Args:
            query: Search query string
            max_results: Maximum number of results
            file_types: Optional MIME type filters

        Returns:
            Dictionary with search results and file contents
        """
        try:
            # Search for files
            files = self._search_files(query, max_results, file_types)

            # Process each file
            results = []
            for file_metadata in files:
                file_id = file_metadata['id']
                mime_type = file_metadata['mimeType']
                name = file_metadata['name']

                logger.info(f"Processing file: {name} ({mime_type})")

                # Get file content if it's text-based
                content = self._get_file_content(file_id, mime_type)

                # Build result entry
                result = {
                    "id": file_id,
                    "name": name,
                    "mime_type": mime_type,
                    "modified_time": file_metadata.get('modifiedTime'),
                    "url": file_metadata.get('webViewLink'),
                    "size": file_metadata.get('size'),
                    "has_content": content is not None,
                }

                if content:
                    # Truncate very large content
                    if len(content) > 50000:
                        logger.warning(f"Truncating large file content: {name}")
                        result["content"] = content[:50000]
                        result["content_truncated"] = True
                    else:
                        result["content"] = content
                        result["content_truncated"] = False

                results.append(result)

            return {
                "success": True,
                "result": {
                    "files": results,
                    "total_results": len(results),
                    "query": query,
                }
            }

        except MCPExecutionError:
            # Re-raise MCP errors
            raise

        except Exception as e:
            logger.error(f"Unexpected error in Google Drive tool: {e}")
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"Unexpected error: {str(e)}",
                original_exception=e,
                parameters={"query": query, "max_results": max_results}
            )

    def clear_token(self) -> bool:
        """
        Delete saved OAuth token to force re-authentication.

        Returns:
            True if token was deleted, False if it didn't exist
        """
        if os.path.exists(self.token_path):
            try:
                os.remove(self.token_path)
                self.service = None
                logger.info(f"Cleared OAuth token: {self.token_path}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete token: {e}")
                return False
        return False
