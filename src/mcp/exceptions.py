"""
MCP Exception Classes

Custom exceptions for MCP (Model Context Protocol) tool execution and management.
"""


class MCPError(Exception):
    """Base exception for all MCP-related errors."""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize MCP error.

        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """Return formatted error message."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class MCPToolNotFoundError(MCPError):
    """Raised when a requested MCP tool does not exist."""

    def __init__(self, tool_name: str, available_tools: list = None):
        """
        Initialize tool not found error.

        Args:
            tool_name: Name of the tool that was not found
            available_tools: Optional list of available tool names
        """
        details = {"tool_name": tool_name}
        if available_tools:
            details["available_tools"] = ", ".join(available_tools)

        message = f"MCP tool '{tool_name}' not found"
        if available_tools:
            message += f". Available tools: {', '.join(available_tools)}"

        super().__init__(message, details)
        self.tool_name = tool_name
        self.available_tools = available_tools or []


class MCPParameterValidationError(MCPError):
    """Raised when MCP tool parameters fail validation."""

    def __init__(
        self,
        tool_name: str,
        parameter_name: str = None,
        expected_type: str = None,
        received_value: any = None,
        validation_message: str = None
    ):
        """
        Initialize parameter validation error.

        Args:
            tool_name: Name of the tool
            parameter_name: Name of the invalid parameter
            expected_type: Expected parameter type
            received_value: Actual value received
            validation_message: Custom validation error message
        """
        details = {"tool_name": tool_name}

        if parameter_name:
            details["parameter"] = parameter_name
        if expected_type:
            details["expected_type"] = expected_type
        if received_value is not None:
            details["received_value"] = str(received_value)
            details["received_type"] = type(received_value).__name__

        # Build error message
        if validation_message:
            message = f"Parameter validation failed for tool '{tool_name}': {validation_message}"
        elif parameter_name and expected_type:
            message = f"Invalid parameter '{parameter_name}' for tool '{tool_name}': expected {expected_type}"
            if received_value is not None:
                message += f", got {type(received_value).__name__}"
        else:
            message = f"Parameter validation failed for tool '{tool_name}'"

        super().__init__(message, details)
        self.tool_name = tool_name
        self.parameter_name = parameter_name
        self.expected_type = expected_type
        self.received_value = received_value


class MCPExecutionError(MCPError):
    """Raised when MCP tool execution fails."""

    def __init__(
        self,
        tool_name: str,
        error_message: str,
        original_exception: Exception = None,
        parameters: dict = None
    ):
        """
        Initialize execution error.

        Args:
            tool_name: Name of the tool that failed
            error_message: Description of the execution error
            original_exception: Original exception that caused the failure
            parameters: Parameters that were passed to the tool
        """
        details = {
            "tool_name": tool_name,
            "error": error_message,
        }

        if original_exception:
            details["original_error"] = str(original_exception)
            details["error_type"] = type(original_exception).__name__

        if parameters:
            # Don't log sensitive parameters
            safe_params = {k: v for k, v in parameters.items() if k not in ['password', 'api_key', 'token', 'secret']}
            if safe_params:
                details["parameters"] = str(safe_params)

        message = f"Tool '{tool_name}' execution failed: {error_message}"

        super().__init__(message, details)
        self.tool_name = tool_name
        self.error_message = error_message
        self.original_exception = original_exception
        self.parameters = parameters


class MCPToolRegistrationError(MCPError):
    """Raised when registering an MCP tool fails."""

    def __init__(self, tool_name: str, reason: str):
        """
        Initialize tool registration error.

        Args:
            tool_name: Name of the tool that failed to register
            reason: Reason for registration failure
        """
        details = {"tool_name": tool_name, "reason": reason}
        message = f"Failed to register tool '{tool_name}': {reason}"

        super().__init__(message, details)
        self.tool_name = tool_name
        self.reason = reason


class MCPConnectionError(MCPError):
    """Raised when MCP client cannot connect to a server."""

    def __init__(self, server_name: str, reason: str, server_config: dict = None):
        """
        Initialize connection error.

        Args:
            server_name: Name of the server
            reason: Reason for connection failure
            server_config: Optional server configuration (sensitive data excluded)
        """
        details = {"server_name": server_name, "reason": reason}
        if server_config:
            # Exclude sensitive fields
            safe_config = {k: v for k, v in server_config.items()
                          if k not in ['api_key', 'token', 'password', 'secret']}
            if safe_config:
                details["config"] = str(safe_config)

        message = f"Failed to connect to MCP server '{server_name}': {reason}"

        super().__init__(message, details)
        self.server_name = server_name
        self.reason = reason
