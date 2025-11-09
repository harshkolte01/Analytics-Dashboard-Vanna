"""
Groq API client for LLM integration
Handles communication with Groq's Llama 3.1 70B model
"""

import asyncio
import httpx
from typing import Dict, List, Optional, Any
import structlog
from app.config import get_groq_config
from .errors import GroqAPIError, RateLimitError

logger = structlog.get_logger(__name__)


class GroqClient:
    """Async client for Groq API integration using direct HTTP calls"""
    
    def __init__(self):
        self.config = get_groq_config()
        self.api_key = self.config["api_key"]
        self.model = self.config["model"]
        self.max_tokens = self.config["max_tokens"]
        self.temperature = self.config["temperature"]
        self.base_url = self.config["base_url"]
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            timeout=30.0,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    async def generate_sql(
        self,
        question: str,
        schema_info: str,
        context: Optional[str] = None,
        examples: Optional[List[Dict]] = None
    ) -> str:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            schema_info: Database schema information
            context: Additional context for the query
            examples: Example queries for few-shot learning
            
        Returns:
            Generated SQL query string
            
        Raises:
            GroqAPIError: If API call fails
            RateLimitError: If rate limit is exceeded
        """
        try:
            prompt = self._build_sql_prompt(question, schema_info, context, examples)
            
            logger.info("Generating SQL with Groq",
                       question=question,
                       model=self.model)
            
            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "stop": ["--", "/*", "*/"]
            }
            
            # Make HTTP request to Groq API
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            sql_query = result["choices"][0]["message"]["content"].strip()
            
            # Clean up the SQL query
            sql_query = self._clean_sql_query(sql_query)
            
            logger.info("SQL generated successfully", 
                       query_length=len(sql_query))
            
            return sql_query
            
        except Exception as e:
            if "rate_limit" in str(e).lower():
                logger.error("Groq rate limit exceeded", error=str(e))
                raise RateLimitError(f"Rate limit exceeded: {str(e)}")
            else:
                logger.error("Groq API error", error=str(e))
                raise GroqAPIError(f"Failed to generate SQL: {str(e)}")
    
    async def explain_query(self, sql_query: str, question: str) -> str:
        """
        Generate explanation for a SQL query
        
        Args:
            sql_query: SQL query to explain
            question: Original natural language question
            
        Returns:
            Human-readable explanation of the query
        """
        try:
            prompt = f"""
            Explain this SQL query in simple business terms:
            
            Original Question: {question}
            SQL Query: {sql_query}
            
            Provide a clear, non-technical explanation of what this query does and what results it will return.
            """
            
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that explains SQL queries in simple business terms."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.3
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
            
        except Exception as e:
            logger.error("Failed to explain query", error=str(e))
            return "Unable to generate explanation for this query."
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for SQL generation"""
        return """
        You are an expert SQL analyst for a business analytics system. Your task is to convert natural language questions into accurate PostgreSQL queries.

        IMPORTANT RULES:
        1. Only generate SELECT statements - no INSERT, UPDATE, DELETE, or DDL
        2. Use proper PostgreSQL syntax and functions
        3. Always use table aliases for readability
        4. Include appropriate WHERE clauses for filtering
        5. Use JOINs when querying multiple tables
        6. Format dates properly using PostgreSQL date functions
        7. Handle NULL values appropriately
        8. Use LIMIT for large result sets when appropriate
        9. Return only the SQL query, no explanations or comments
        10. Use proper aggregation functions (SUM, COUNT, AVG, etc.)
        11. CRITICAL: Always use double-quoted table names exactly as shown in schema (e.g., "Vendor", "Invoice", "Payment") because PostgreSQL table names are case-sensitive
        12. CRITICAL: Column names should also be double-quoted if they contain capital letters (e.g., "invoiceDate", "invoiceTotal")

        Focus on business analytics queries related to:
        - Invoice analysis and trends
        - Vendor performance and spending
        - Financial reporting and cash flow
        - Document processing statistics
        - Payment terms and due dates
        """
    
    def _build_sql_prompt(
        self,
        question: str,
        schema_info: str,
        context: Optional[str] = None,
        examples: Optional[List[Dict]] = None
    ) -> str:
        """Build the complete prompt for SQL generation"""
        prompt_parts = [
            f"Database Schema:\n{schema_info}",
            f"\nQuestion: {question}"
        ]
        
        if context:
            prompt_parts.insert(-1, f"\nContext: {context}")
        
        if examples:
            examples_text = "\n\nExample queries:\n"
            for i, example in enumerate(examples[:3], 1):  # Limit to 3 examples
                examples_text += f"{i}. Question: {example.get('question', '')}\n"
                examples_text += f"   SQL: {example.get('sql', '')}\n\n"
            prompt_parts.insert(-1, examples_text)
        
        prompt_parts.append("\nGenerate a PostgreSQL query to answer this question:")
        
        return "\n".join(prompt_parts)
    
    def _clean_sql_query(self, sql_query: str) -> str:
        """Clean and format the generated SQL query"""
        # Remove markdown code blocks if present
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0]
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0]
        
        # Remove comments and extra whitespace
        lines = []
        for line in sql_query.split('\n'):
            line = line.strip()
            if line and not line.startswith('--') and not line.startswith('/*'):
                lines.append(line)
        
        sql_query = ' '.join(lines)
        
        # Ensure query ends with semicolon
        if not sql_query.rstrip().endswith(';'):
            sql_query = sql_query.rstrip() + ';'
        
        return sql_query
    
    async def health_check(self) -> bool:
        """Check if Groq API is accessible"""
        try:
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant."
                    },
                    {
                        "role": "user",
                        "content": "Say 'OK'"
                    }
                ],
                "max_tokens": 10,
                "temperature": 0
            }
            
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error("Groq health check failed", error=str(e))
            return False
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()