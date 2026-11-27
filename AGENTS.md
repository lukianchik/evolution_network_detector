# Working on the detector

Run commands from the repository root using Python 3.12. Keep the script entry points used by the panel, MCP and Shinka subprocesses working. Use temporary experiment directories in tests. Run Ruff and unittest discovery before submitting changes.

Keep the official UNSW test split outside every search path. Mock providers in automated checks. Never commit credentials, quota databases, downloaded datasets or run directories.
