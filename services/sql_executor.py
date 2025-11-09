"""
SQL executor service with safety constraints
Executes SQL queries with read-only permissions and safety checks
"""

import asyncio
import re
import time
from typing import Dict, List, Optional, Any, Tuple
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import get_database_config, get_security_config
from .errors import QueryExecutionError, SQLValidationError, TimeoutError, DatabaseError

logger = structlog.get_logger(__name__)


class SQLExecutor:
    """Executes SQL queries with safety constraints and monitoring"""
    
    def __init__(self):
        self.db_config = get_database_config()
        self.security_config = get_security_config()
        self.engine = None
        self.async_engine = None
        
        # SQL safety patterns
        self.dangerous_patterns = [
            r'\b(DROP|DELETE|UPDATE|INSERT|ALTER|CREATE|TRUNCATE|REPLACE)\b',
            r'\b(EXEC|EXECUTE|CALL)\b',
            r'--.*$',  # SQL comments
            r'/\*.*?\*/',  # Multi-line comments
            r'\b(UNION\s+ALL|UNION)\b.*\b(SELECT)\b',  # Potential injection
            r';\s*\w+',  # Multiple statements
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.MULTILINE) 
                                for pattern in self.dangerous_patterns]
    
    async def initialize(self):
        """Initialize database connection for query execution"""
        try:
            db_url = self.db_config["url"]
            if db_url.startswith("postgresql://"):
                async_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            else:
                async_url = db_url
                
            self.async_engine = create_async_engine(
                async_url,
                pool_size=self.db_config["pool_size"],
                max_overflow=self.db_config["max_overflow"],
                echo=self.db_config["echo"]
            )
            
            logger.info("SQL executor initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize SQL executor", error=str(e))
            raise DatabaseError(f"Failed to initialize database connection: {str(e)}")
    
    async def execute_query(
        self, 
        sql_query: str, 
        parameters: Optional[Dict] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute SQL query with safety checks and monitoring
        
        Args:
            sql_query: SQL query to execute
            parameters: Query parameters for parameterized queries
            timeout: Query timeout in seconds
            
        Returns:
            Dictionary containing query results and metadata
            
        Raises:
            SQLValidationError: If query fails safety validation
            QueryExecutionError: If query execution fails
            TimeoutError: If query exceeds timeout
        """
        start_time = time.time()
        
        try:
            # Validate query safety
            self._validate_query_safety(sql_query)
            
            # Set timeout
            query_timeout = timeout or self.security_config["query_timeout"]
            
            if not self.async_engine:
                await self.initialize()
            
            logger.info("Executing SQL query", 
                       query_preview=sql_query[:100] + "..." if len(sql_query) > 100 else sql_query)
            
            # Execute query with timeout
            async with self.async_engine.begin() as conn:
                # Set statement timeout
                await conn.execute(text(f"SET statement_timeout = {query_timeout * 1000}"))
                
                result = await conn.execute(text(sql_query), parameters or {})
                
                # Fetch results
                rows = result.fetchall()
                columns = list(result.keys()) if rows else []
                
                execution_time = time.time() - start_time
                
                # Convert rows to dictionaries
                data = []
                for row in rows:
                    data.append(dict(zip(columns, row)))
                
                query_result = {
                    "success": True,
                    "data": data,
                    "columns": columns,
                    "row_count": len(data),
                    "execution_time": round(execution_time, 3),
                    "query": sql_query,
                    "timestamp": time.time()
                }
                
                logger.info("Query executed successfully", 
                           row_count=len(data),
                           execution_time=execution_time)
                
                return query_result
                
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            logger.error("Query timeout exceeded", 
                        timeout=query_timeout,
                        execution_time=execution_time)
            raise TimeoutError(f"Query exceeded timeout of {query_timeout} seconds")
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Query execution failed", 
                        error=str(e),
                        execution_time=execution_time)
            
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "execution_time": round(execution_time, 3),
                "query": sql_query,
                "timestamp": time.time()
            }
    
    def _validate_query_safety(self, sql_query: str) -> None:
        """
        Validate that the SQL query is safe to execute
        
        Args:
            sql_query: SQL query to validate
            
        Raises:
            SQLValidationError: If query fails safety checks
        """
        # Remove extra whitespace and normalize
        normalized_query = ' '.join(sql_query.strip().split())
        
        # Check if query is empty
        if not normalized_query:
            raise SQLValidationError("Empty query provided")
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(normalized_query):
                raise SQLValidationError(
                    f"Query contains potentially dangerous operation: {pattern.pattern}"
                )
        
        # Ensure query starts with SELECT
        if not normalized_query.upper().startswith('SELECT'):
            raise SQLValidationError("Only SELECT queries are allowed")
        
        # Check for multiple statements (basic check)
        if normalized_query.count(';') > 1:
            raise SQLValidationError("Multiple statements are not allowed")
        
        # Check query complexity (basic heuristic)
        complexity_score = self._calculate_query_complexity(normalized_query)
        max_complexity = self.security_config["max_complexity"]
        
        if complexity_score > max_complexity:
            raise SQLValidationError(
                f"Query complexity ({complexity_score}) exceeds maximum allowed ({max_complexity})"
            )
        
        logger.debug("Query passed safety validation", complexity=complexity_score)
    
    def _calculate_query_complexity(self, sql_query: str) -> int:
        """
        Calculate a complexity score for the SQL query
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Complexity score (higher = more complex)
        """
        complexity = 0
        query_upper = sql_query.upper()
        
        # Count JOINs
        complexity += query_upper.count('JOIN') * 5
        
        # Count subqueries
        complexity += query_upper.count('SELECT') * 3
        
        # Count aggregations
        aggregations = ['SUM', 'COUNT', 'AVG', 'MAX', 'MIN', 'GROUP BY']
        for agg in aggregations:
            complexity += query_upper.count(agg) * 2
        
        # Count CASE statements
        complexity += query_upper.count('CASE') * 3
        
        # Count UNION operations
        complexity += query_upper.count('UNION') * 4
        
        # Count window functions
        complexity += query_upper.count('OVER') * 4
        
        # Count CTEs (WITH clauses)
        complexity += query_upper.count('WITH') * 3
        
        # Base complexity for any query
        complexity += 1
        
        return complexity
    
    async def explain_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Get query execution plan using EXPLAIN
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Query execution plan information
        """
        try:
            self._validate_query_safety(sql_query)
            
            if not self.async_engine:
                await self.initialize()
            
            explain_query = f"EXPLAIN (FORMAT JSON, ANALYZE FALSE) {sql_query}"
            
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text(explain_query))
                plan = result.fetchone()[0]
                
                return {
                    "success": True,
                    "execution_plan": plan,
                    "query": sql_query
                }
                
        except Exception as e:
            logger.error("Failed to explain query", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "query": sql_query
            }
    
    async def validate_query_syntax(self, sql_query: str) -> Dict[str, Any]:
        """
        Validate SQL query syntax without executing it
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Validation result
        """
        try:
            self._validate_query_safety(sql_query)
            
            if not self.async_engine:
                await self.initialize()
            
            # Use EXPLAIN to validate syntax without execution
            explain_query = f"EXPLAIN {sql_query}"
            
            async with self.async_engine.begin() as conn:
                await conn.execute(text(explain_query))
                
                return {
                    "valid": True,
                    "message": "Query syntax is valid",
                    "query": sql_query
                }
                
        except SQLValidationError as e:
            return {
                "valid": False,
                "message": f"Safety validation failed: {str(e)}",
                "query": sql_query
            }
        except Exception as e:
            return {
                "valid": False,
                "message": f"Syntax error: {str(e)}",
                "query": sql_query
            }
    
    async def get_query_stats(self, sql_query: str) -> Dict[str, Any]:
        """
        Get estimated statistics for a query without executing it
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Query statistics and estimates
        """
        try:
            explain_result = await self.explain_query(sql_query)
            
            if not explain_result["success"]:
                return explain_result
            
            plan = explain_result["execution_plan"][0]["Plan"]
            
            return {
                "success": True,
                "estimated_rows": plan.get("Plan Rows", 0),
                "estimated_cost": plan.get("Total Cost", 0),
                "estimated_width": plan.get("Plan Width", 0),
                "node_type": plan.get("Node Type", "Unknown"),
                "query": sql_query
            }
            
        except Exception as e:
            logger.error("Failed to get query stats", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "query": sql_query
            }
    
    async def health_check(self) -> bool:
        """Check if database connection is healthy"""
        try:
            if not self.async_engine:
                await self.initialize()
            
            async with self.async_engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
                return True
                
        except Exception as e:
            logger.error("SQL executor health check failed", error=str(e))
            return False
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("SQL executor connections closed")