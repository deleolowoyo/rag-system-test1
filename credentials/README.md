# Google Drive Credentials Setup

This directory contains OAuth2 credentials for Google Drive MCP integration.

## Prerequisites

1. A Google Cloud Platform account
2. A project with Google Drive API enabled

## Setup Instructions

### 1. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google Drive API:
   - Navigate to "APIs & Services" > "Library"
   - Search for "Google Drive API"
   - Click "Enable"

### 2. Create OAuth2 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. Select "Desktop app" as the application type
4. Name it (e.g., "RAG System Drive Access")
5. Click "Create"

### 3. Download Credentials

1. Click the download icon next to your newly created OAuth client
2. Save the JSON file as `google_credentials.json` in this directory
3. The file should look like:
   ```json
   {
     "installed": {
       "client_id": "your-client-id.apps.googleusercontent.com",
       "project_id": "your-project-id",
       "auth_uri": "https://accounts.google.com/o/oauth2/auth",
       "token_uri": "https://oauth2.googleapis.com/token",
       "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
       "client_secret": "your-client-secret",
       "redirect_uris": ["http://localhost"]
     }
   }
   ```

### 4. First-Time Authentication

The first time you run the RAG system with Google Drive MCP enabled:

1. A browser window will open asking you to authenticate
2. Sign in with your Google account
3. Grant the requested permissions
4. The system will save a token to `google_token.json`
5. Future runs will use this token (no browser required)

### Token Refresh

- OAuth tokens expire after a period of time
- The system will automatically refresh tokens when needed
- If refresh fails, delete `google_token.json` and re-authenticate

## Security Notes

⚠️ **IMPORTANT**:

- **NEVER commit these files to git**
- The `.gitignore` already excludes this directory
- These credentials grant access to your Google Drive
- Treat them like passwords

## Files in this Directory

- `google_credentials.json` - OAuth2 client credentials (you create this)
- `google_token.json` - Access/refresh tokens (auto-generated on first use)
- `README.md` - This file (safe to commit)

## Troubleshooting

### "File not found" error

- Ensure `google_credentials.json` exists in this directory
- Check the path in your `.env` file matches: `./credentials/google_credentials.json`

### "Invalid credentials" error

- Re-download credentials from Google Cloud Console
- Ensure the Google Drive API is enabled for your project

### Authentication browser not opening

- Check firewall settings
- Try running: `python -m webbrowser http://localhost`
- Manual flow: copy the URL from console and open in browser

### Token expired

- Delete `google_token.json`
- Re-run the application to re-authenticate

## Alternative: Service Account

For production deployments, consider using a service account instead:

1. Create a service account in Google Cloud Console
2. Download the service account JSON key
3. Share Drive folders with the service account email
4. Update the code to use service account authentication

See Phase 3 documentation for implementation details.
