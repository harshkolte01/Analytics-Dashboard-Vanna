"""
FastAPI main application for Vanna AI service
Provides natural language to SQL conversion endpoints
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import structlog
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from app.config import settings
from services.groq_client import GroqClient
from services.schema_provider import SchemaProvider
from services.sql_generator import SQLGenerator
from services.sql_executor import SQLExecutor
from services.nl_parser import NaturalLanguageParser
from services.result_formatter import ResultFormatter
from services.errors import VannaServiceError, ValidationError, DatabaseError

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Global service instances
services = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Vanna AI service", version="1.0.0")
    
    try:
        # Initialize services
        services['groq_client'] = GroqClient()
        services['schema_provider'] = SchemaProvider()
        services['sql_generator'] = SQLGenerator()
        services['sql_executor'] = SQLExecutor()
        services['nl_parser'] = NaturalLanguageParser()
        services['result_formatter'] = ResultFormatter()
        
        # Initialize async services
        await services['schema_provider'].initialize()
        await services['sql_generator'].initialize()
        await services['sql_executor'].initialize()
        
        logger.info("All services initialized successfully")
        
    except Exception as e:
        logger.error("Failed to initialize services", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Vanna AI service")
    
    try:
        # Close services
        if 'schema_provider' in services:
            await services['schema_provider'].close()
        if 'sql_generator' in services:
            await services['sql_generator'].close()
        if 'sql_executor' in services:
            await services['sql_executor'].close()
            
        logger.info("All services closed successfully")
        
    except Exception as e:
        logger.error("Error during shutdown", error=str(e))


# Create FastAPI app
app = FastAPI(
    title="Vanna AI Service",
    description="Natural Language to SQL conversion service using Vanna AI and Groq",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    format: Optional[str] = Field("json", description="Output format")
    include_explanation: Optional[bool] = Field(True, description="Include query explanation")
    execute_query: Optional[bool] = Field(True, description="Execute the generated query")


class QueryResponse(BaseModel):
    success: bool
    question: str
    sql_query: Optional[str] = None
    explanation: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    timestamp: str


class SchemaResponse(BaseModel):
    success: bool
    schema_info: Optional[str] = None
    tables: Optional[List[str]] = None
    error: Optional[str] = None


class ValidationRequest(BaseModel):
    sql_query: str = Field(..., description="SQL query to validate")


class ValidationResponse(BaseModel):
    valid: bool
    message: str
    query: str
    suggestions: Optional[List[str]] = None


class HealthResponse(BaseModel):
    status: str
    services: Dict[str, bool]
    timestamp: str


# Dependency injection
def get_services():
    """Get service instances"""
    return services


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Vanna AI Service",
        "version": "1.0.0",
        "description": "Natural Language to SQL conversion service",
        "endpoints": {
            "POST /query": "Convert natural language to SQL and execute",
            "GET /schema": "Get database schema information",
            "POST /validate": "Validate SQL query",
            "GET /health": "Service health check"
        }
    }


@app.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    services: Dict = Depends(get_services)
):
    """
    Process natural language query and return SQL with results
    """
    try:
        logger.info("Processing query request", question=request.question)
        
        # Parse natural language query
        nl_parser = services['nl_parser']
        parsed_query = await nl_parser.parse_query(request.question)
        
        # Generate SQL
        sql_generator = services['sql_generator']
        generation_result = await sql_generator.generate_sql_from_question(
            question=request.question,
            context=request.context,
            include_explanation=request.include_explanation
        )
        
        if not generation_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"SQL generation failed: {generation_result.get('error', 'Unknown error')}"
            )
        
        sql_query = generation_result['sql_query']
        explanation = generation_result.get('explanation', '')
        
        # Execute query if requested
        results = None
        if request.execute_query:
            sql_executor = services['sql_executor']
            execution_result = await sql_executor.execute_query(sql_query)
            
            if execution_result['success']:
                # Format results
                result_formatter = services['result_formatter']
                formatted_result = await result_formatter.format_query_result(
                    execution_result,
                    format_type=request.format
                )
                results = formatted_result
            else:
                # Query execution failed, but still return the SQL
                results = {
                    'success': False,
                    'error': execution_result.get('error', 'Query execution failed'),
                    'execution_time': execution_result.get('execution_time', 0)
                }
        
        # Prepare response
        response = QueryResponse(
            success=True,
            question=request.question,
            sql_query=sql_query,
            explanation=explanation,
            results=results,
            metadata={
                'parsed_query': parsed_query,
                'generation_metadata': generation_result.get('metadata', {}),
                'format': request.format,
                'executed': request.execute_query
            },
            timestamp=str(asyncio.get_event_loop().time())
        )
        
        logger.info("Query processed successfully", 
                   question=request.question,
                   executed=request.execute_query)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query processing failed", error=str(e), question=request.question)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/schema", response_model=SchemaResponse)
async def get_schema(
    include_sample_data: bool = False,
    services: Dict = Depends(get_services)
):
    """
    Get database schema information
    """
    try:
        logger.info("Getting schema information", include_sample_data=include_sample_data)
        
        schema_provider = services['schema_provider']
        
        # Get schema info
        schema_info = await schema_provider.get_schema_info(include_sample_data)
        
        # Get table names
        table_names = await schema_provider.get_table_names()
        
        return SchemaResponse(
            success=True,
            schema_info=schema_info,
            tables=table_names
        )
        
    except Exception as e:
        logger.error("Failed to get schema", error=str(e))
        return SchemaResponse(
            success=False,
            error=str(e)
        )


@app.post("/validate", response_model=ValidationResponse)
async def validate_query(
    request: ValidationRequest,
    services: Dict = Depends(get_services)
):
    """
    Validate SQL query syntax and safety
    """
    try:
        logger.info("Validating SQL query", query_preview=request.sql_query[:100])
        
        sql_executor = services['sql_executor']
        validation_result = await sql_executor.validate_query_syntax(request.sql_query)
        
        return ValidationResponse(
            valid=validation_result['valid'],
            message=validation_result['message'],
            query=request.sql_query,
            suggestions=[]  # Could add suggestions in the future
        )
        
    except Exception as e:
        logger.error("Query validation failed", error=str(e))
        return ValidationResponse(
            valid=False,
            message=f"Validation error: {str(e)}",
            query=request.sql_query
        )


@app.get("/health", response_model=HealthResponse)
async def health_check(services: Dict = Depends(get_services)):
    """
    Service health check endpoint
    """
    try:
        service_health = {}
        
        # Check each service
        if 'groq_client' in services:
            service_health['groq'] = await services['groq_client'].health_check()
        
        if 'sql_executor' in services:
            service_health['database'] = await services['sql_executor'].health_check()
        
        if 'sql_generator' in services:
            service_health['sql_generator'] = await services['sql_generator'].health_check()
        
        # Overall status
        overall_status = "healthy" if all(service_health.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            services=service_health,
            timestamp=str(asyncio.get_event_loop().time())
        )
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return HealthResponse(
            status="unhealthy",
            services={},
            timestamp=str(asyncio.get_event_loop().time())
        )


@app.get("/metrics")
async def get_metrics():
    """
    Get service metrics (basic implementation)
    """
    # This could be enhanced with proper metrics collection
    return {
        "service": "vanna-ai",
        "uptime": "unknown",
        "requests_total": "unknown",
        "errors_total": "unknown"
    }


class ExplainRequest(BaseModel):
    sql: str = Field(..., description="SQL query to explain")


class ExplainResponse(BaseModel):
    explanation: str
    sql: str


@app.post("/explain", response_model=ExplainResponse)
async def explain_query(
    request: ExplainRequest,
    services: Dict = Depends(get_services)
):
    """
    Explain a SQL query in natural language
    """
    try:
        logger.info("Explaining SQL query", query_preview=request.sql[:100])
        
        # Use the SQL generator to explain the query
        sql_generator = services['sql_generator']
        explanation_result = await sql_generator.explain_sql_query(request.sql)
        
        if not explanation_result['success']:
            raise HTTPException(
                status_code=400,
                detail=f"Query explanation failed: {explanation_result.get('error', 'Unknown error')}"
            )
        
        return ExplainResponse(
            explanation=explanation_result['explanation'],
            sql=request.sql
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Query explanation failed", error=str(e), sql=request.sql)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# Error handlers
@app.exception_handler(VannaServiceError)
async def vanna_service_error_handler(request, exc: VannaServiceError):
    """Handle Vanna service specific errors"""
    logger.error("Vanna service error", error=str(exc), error_code=exc.error_code)
    return JSONResponse(
        status_code=400,
        content={
            "error": exc.message,
            "error_code": exc.error_code,
            "details": exc.details
        }
    )


@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    """Handle validation errors"""
    logger.error("Validation error", error=str(exc))
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "message": str(exc)
        }
    )


@app.exception_handler(DatabaseError)
async def database_error_handler(request, exc: DatabaseError):
    """Handle database errors"""
    logger.error("Database error", error=str(exc))
    return JSONResponse(
        status_code=503,
        content={
            "error": "Database service unavailable",
            "message": str(exc)
        }
    )


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )