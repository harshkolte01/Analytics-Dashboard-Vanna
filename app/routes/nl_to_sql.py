"""
Natural Language to SQL API routes
Handles conversion of natural language questions to SQL queries
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import logging
import time

from services.sql_generator import SQLGenerator
from services.schema_provider import SchemaProvider
from services.sql_executor import SQLExecutor
from services.nl_parser import NLParser
from services.result_formatter import ResultFormatter
from services.sql_validator import SQLValidator

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1", tags=["Natural Language to SQL"])

# Request/Response models
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    format: str = Field(default="json", description="Output format: json, csv, table, chart_data, summary")
    include_explanation: bool = Field(default=True, description="Include query explanation")
    execute_query: bool = Field(default=True, description="Execute the generated query")

class QueryResponse(BaseModel):
    question: str
    sql: str
    results: Optional[List[Dict[str, Any]]] = None
    format: str
    execution_time: float
    row_count: int
    explanation: Optional[str] = None
    confidence: float
    tables_used: List[str]
    error: Optional[str] = None

class SchemaResponse(BaseModel):
    tables: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    sample_data: Optional[Dict[str, Any]] = None

class ValidationRequest(BaseModel):
    sql_query: str = Field(..., description="SQL query to validate")

class ValidationResponse(BaseModel):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]

# Initialize services
sql_generator = SQLGenerator()
schema_provider = SchemaProvider()
sql_executor = SQLExecutor()
nl_parser = NLParser()
result_formatter = ResultFormatter()
sql_validator = SQLValidator()

@router.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process a natural language query and return SQL with results
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: {request.question}")
        
        # Parse natural language to understand intent
        parsed_query = await nl_parser.parse(request.question, request.context)
        
        # Generate SQL from natural language
        sql_result = await sql_generator.generate_sql(
            question=request.question,
            context=request.context
        )
        
        results = None
        row_count = 0
        error = None
        
        if request.execute_query and sql_result.get("sql"):
            try:
                # Execute the SQL query
                execution_result = await sql_executor.execute_query(sql_result["sql"])
                results = execution_result.get("results", [])
                row_count = len(results) if results else 0
                
                # Format results according to requested format
                if request.format != "json":
                    results = await result_formatter.format_results(
                        results, request.format
                    )
                    
            except Exception as e:
                logger.error(f"Query execution failed: {str(e)}")
                error = f"Query execution failed: {str(e)}"
        
        execution_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Generate explanation if requested
        explanation = None
        if request.include_explanation and sql_result.get("sql"):
            explanation = await sql_generator.explain_query(sql_result["sql"])
        
        return QueryResponse(
            question=request.question,
            sql=sql_result.get("sql", ""),
            results=results,
            format=request.format,
            execution_time=execution_time,
            row_count=row_count,
            explanation=explanation,
            confidence=sql_result.get("confidence", 0.0),
            tables_used=sql_result.get("tables_used", []),
            error=error
        )
        
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        execution_time = (time.time() - start_time) * 1000
        
        return QueryResponse(
            question=request.question,
            sql="",
            results=None,
            format=request.format,
            execution_time=execution_time,
            row_count=0,
            explanation=None,
            confidence=0.0,
            tables_used=[],
            error=str(e)
        )

@router.get("/schema", response_model=SchemaResponse)
async def get_schema(include_sample_data: bool = False):
    """
    Get database schema information for AI context
    """
    try:
        logger.info("Fetching database schema")
        
        schema_info = await schema_provider.get_schema_info()
        
        sample_data = None
        if include_sample_data:
            sample_data = await schema_provider.get_sample_data()
        
        return SchemaResponse(
            tables=schema_info.get("tables", []),
            relationships=schema_info.get("relationships", []),
            sample_data=sample_data
        )
        
    except Exception as e:
        logger.error(f"Schema retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")

@router.post("/validate", response_model=ValidationResponse)
async def validate_query(request: ValidationRequest):
    """
    Validate SQL query syntax and safety
    """
    try:
        logger.info(f"Validating SQL query: {request.sql_query[:100]}...")
        
        validation_result = await sql_validator.validate_query(request.sql_query)
        
        return ValidationResponse(
            is_valid=validation_result.get("is_valid", False),
            errors=validation_result.get("errors", []),
            warnings=validation_result.get("warnings", []),
            suggestions=validation_result.get("suggestions", [])
        )
        
    except Exception as e:
        logger.error(f"Query validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query validation failed: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Check AI service health status
    """
    try:
        # Test database connection
        schema_status = await schema_provider.test_connection()
        
        # Test SQL generator
        generator_status = await sql_generator.test_connection()
        
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "services": {
                "database": "connected" if schema_status else "disconnected",
                "sql_generator": "ready" if generator_status else "not_ready",
                "groq_api": "connected" if generator_status else "disconnected"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "timestamp": time.time(),
            "error": str(e)
        }

@router.get("/suggestions")
async def get_suggestions():
    """
    Get suggested questions based on database schema
    """
    try:
        logger.info("Generating query suggestions")
        
        suggestions = await nl_parser.generate_suggestions()
        
        return {
            "suggestions": suggestions,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Suggestion generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Suggestion generation failed: {str(e)}")

@router.get("/config")
async def get_config():
    """
    Get AI service configuration and status
    """
    from app.config import settings
    
    return {
        "service": "Vanna AI Natural Language to SQL",
        "version": "1.0.0",
        "model": settings.groq_model,
        "database": "PostgreSQL",
        "security": {
            "allowed_operations": settings.allowed_operations_list,
            "query_timeout": settings.query_timeout,
            "max_complexity": settings.max_query_complexity
        },
        "features": [
            "Natural language to SQL conversion",
            "Query explanation and validation",
            "Multi-format result output",
            "Schema-aware query generation",
            "Safety constraints and validation"
        ]
    }