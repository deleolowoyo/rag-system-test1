"""
MCP Base Classes

Base abstract classes for MCP (Model Context Protocol) tools and components.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, validator

from .exceptions import MCPParameterValidationError, MCPExecutionError

logger = logging.getLogger(__name__)


class MCPToolParameter(BaseModel):
    """Schema for a single MCP tool parameter."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (string, integer, boolean, array, object)")
    description: str = Field(..., description="Human-readable parameter description")
    required: bool = Field(default=False, description="Whether parameter is required")
    default: Any = Field(default=None, description="Default value if not provided")
    enum: Optional[List[Any]] = Field(default=None, description="Allowed values (if restricted)")

    @validator('type')
    def validate_type(cls, v):
        """Validate parameter type."""
        valid_types = ['string', 'integer', 'number', 'boolean', 'array', 'object', 'null']
        if v not in valid_types:
            raise ValueError(f"Invalid parameter type: {v}. Must be one of {valid_types}")
        return v


class MCPToolSchema(BaseModel):
    """Complete schema for an MCP tool."""

    name: str = Field(..., description="Tool name (unique identifier)")
    description: str = Field(..., description="Human-readable tool description")
    parameters: List[MCPToolParameter] = Field(
        default_factory=list,
        description="List of tool parameters"
    )
    category: Optional[str] = Field(
        default="general",
        description="Tool category (filesystem, database, web, etc.)"
    )
    version: Optional[str] = Field(default="1.0.0", description="Tool version")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert schema to dictionary format.

        Returns:
            Dictionary representation suitable for LLM tool use
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: {
                        "type": param.type,
                        "description": param.description,
                        **({"enum": param.enum} if param.enum else {}),
                        **({"default": param.default} if param.default is not None else {}),
                    }
                    for param in self.parameters
                },
                "required": [param.name for param in self.parameters if param.required],
            },
            "category": self.category,
            "version": self.version,
        }

    def get_required_parameters(self) -> List[str]:
        """Get list of required parameter names."""
        return [param.name for param in self.parameters if param.required]

    def get_parameter(self, name: str) -> Optional[MCPToolParameter]:
        """Get parameter by name."""
        for param in self.parameters:
            if param.name == name:
                return param
        return None


class MCPTool(ABC):
    """
    Abstract base class for MCP tools.

    All MCP tools must inherit from this class and implement:
    - schema property: Returns MCPToolSchema
    - _execute() method: Performs the actual tool operation
    """

    def __init__(self):
        """Initialize MCP tool."""
        self._schema = self._create_schema()
        logger.debug(f"Initialized MCP tool: {self.name}")

    @property
    def name(self) -> str:
        """Get tool name."""
        return self._schema.name

    @property
    def description(self) -> str:
        """Get tool description."""
        return self._schema.description

    @property
    def schema(self) -> MCPToolSchema:
        """Get complete tool schema."""
        return self._schema

    @property
    def category(self) -> str:
        """Get tool category."""
        return self._schema.category

    @abstractmethod
    def _create_schema(self) -> MCPToolSchema:
        """
        Create and return the tool's schema.

        Must be implemented by subclasses.

        Returns:
            MCPToolSchema defining the tool's interface

        Example:
            return MCPToolSchema(
                name="read_file",
                description="Read contents of a file",
                parameters=[
                    MCPToolParameter(
                        name="file_path",
                        type="string",
                        description="Path to the file to read",
                        required=True
                    )
                ],
                category="filesystem"
            )
        """
        pass

    @abstractmethod
    def _execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with given parameters.

        Must be implemented by subclasses.

        Args:
            **kwargs: Tool parameters as defined in schema

        Returns:
            Dictionary with execution results. Should include:
            - success: bool indicating if execution succeeded
            - result: The actual result data (if successful)
            - error: Error message (if failed)

        Raises:
            MCPExecutionError: If execution fails

        Example:
            def _execute(self, file_path: str) -> Dict[str, Any]:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read()
                    return {
                        "success": True,
                        "result": {"content": content, "path": file_path}
                    }
                except Exception as e:
                    raise MCPExecutionError(
                        tool_name=self.name,
                        error_message=str(e),
                        original_exception=e,
                        parameters={"file_path": file_path}
                    )
        """
        pass

    def validate_parameters(self, **kwargs) -> bool:
        """
        Validate parameters against schema.

        Args:
            **kwargs: Parameters to validate

        Returns:
            True if validation succeeds

        Raises:
            MCPParameterValidationError: If validation fails
        """
        logger.debug(f"Validating parameters for tool '{self.name}': {kwargs}")

        # Check for required parameters
        required_params = self._schema.get_required_parameters()
        for param_name in required_params:
            if param_name not in kwargs or kwargs[param_name] is None:
                raise MCPParameterValidationError(
                    tool_name=self.name,
                    parameter_name=param_name,
                    validation_message=f"Required parameter '{param_name}' is missing"
                )

        # Validate each provided parameter
        for param_name, param_value in kwargs.items():
            param_schema = self._schema.get_parameter(param_name)

            if not param_schema:
                logger.warning(
                    f"Unknown parameter '{param_name}' provided to tool '{self.name}'. "
                    f"This parameter will be ignored."
                )
                continue

            # Type validation (basic)
            if not self._validate_type(param_value, param_schema.type):
                raise MCPParameterValidationError(
                    tool_name=self.name,
                    parameter_name=param_name,
                    expected_type=param_schema.type,
                    received_value=param_value
                )

            # Enum validation
            if param_schema.enum and param_value not in param_schema.enum:
                raise MCPParameterValidationError(
                    tool_name=self.name,
                    parameter_name=param_name,
                    validation_message=f"Value must be one of {param_schema.enum}, got '{param_value}'"
                )

        logger.debug(f"Parameter validation successful for tool '{self.name}'")
        return True

    def _validate_type(self, value: Any, expected_type: str) -> bool:
        """
        Validate value against expected type.

        Args:
            value: Value to validate
            expected_type: Expected type string

        Returns:
            True if type matches
        """
        if value is None:
            return expected_type == 'null'

        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': (list, tuple),
            'object': dict,
        }

        expected_python_type = type_map.get(expected_type)
        if not expected_python_type:
            logger.warning(f"Unknown type '{expected_type}', skipping type validation")
            return True

        return isinstance(value, expected_python_type)

    def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Execute tool with validation and error handling.

        This is the public method that should be called to execute the tool.
        It handles validation and wraps _execute() with error handling.

        Args:
            **kwargs: Tool parameters

        Returns:
            Dictionary with execution results

        Raises:
            MCPParameterValidationError: If parameter validation fails
            MCPExecutionError: If execution fails
        """
        logger.info(f"Executing MCP tool '{self.name}'")

        try:
            # Validate parameters
            self.validate_parameters(**kwargs)

            # Add default values for missing optional parameters
            params = self._add_defaults(kwargs)

            # Execute tool
            result = self._execute(**params)

            logger.info(f"Tool '{self.name}' executed successfully")
            return result

        except MCPParameterValidationError:
            logger.error(f"Parameter validation failed for tool '{self.name}'")
            raise

        except MCPExecutionError:
            logger.error(f"Execution failed for tool '{self.name}'")
            raise

        except Exception as e:
            logger.error(f"Unexpected error executing tool '{self.name}': {str(e)}")
            raise MCPExecutionError(
                tool_name=self.name,
                error_message=f"Unexpected error: {str(e)}",
                original_exception=e,
                parameters=kwargs
            )

    def _add_defaults(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add default values for missing optional parameters.

        Args:
            params: Provided parameters

        Returns:
            Parameters with defaults added
        """
        result = params.copy()

        for param in self._schema.parameters:
            if param.name not in result and param.default is not None:
                result[param.name] = param.default
                logger.debug(f"Using default value for '{param.name}': {param.default}")

        return result

    def get_schema_dict(self) -> Dict[str, Any]:
        """
        Get schema as dictionary.

        Returns:
            Dictionary representation of schema
        """
        return self._schema.to_dict()

    def __str__(self) -> str:
        """String representation of tool."""
        return f"MCPTool({self.name}, category={self.category})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"MCPTool(name='{self.name}', "
            f"description='{self.description}', "
            f"category='{self.category}', "
            f"parameters={len(self._schema.parameters)})"
        )
