"""
SQL generation service using Vanna AI with Groq LLM integration
Orchestrates natural language to SQL conversion with context awareness
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Tuple
import structlog
from vanna.base import VannaBase
from app.config import settings
from .groq_client import GroqClient
from .schema_provider import SchemaProvider
from .errors import SQLGenerationError, ValidationError

logger = structlog.get_logger(__name__)


class GroqVannaAdapter(VannaBase):
    """Custom Vanna adapter that uses Groq for LLM operations"""
    
    def __init__(self, groq_client: GroqClient, config: Optional[Dict] = None):
        VannaBase.__init__(self, config=config)
        self.groq_client = groq_client
        
    def system_message(self, message: str) -> Any:
        """System message for the LLM"""
        return {"role": "system", "content": message}
    
    def user_message(self, message: str) -> Any:
        """User message for the LLM"""
        return {"role": "user", "content": message}
    
    def assistant_message(self, message: str) -> Any:
        """Assistant message for the LLM"""
        return {"role": "assistant", "content": message}
    
    async def submit_prompt(self, prompt: str, **kwargs) -> str:
        """Submit prompt to Groq LLM"""
        try:
            # Extract schema info and context from kwargs
            schema_info = kwargs.get('schema_info', '')
            context = kwargs.get('context', '')
            examples = kwargs.get('examples', [])
            
            return await self.groq_client.generate_sql(
                question=prompt,
                schema_info=schema_info,
                context=context,
                examples=examples
            )
        except Exception as e:
            logger.error("Failed to submit prompt to Groq", error=str(e))
            raise SQLGenerationError(f"LLM prompt submission failed: {str(e)}")
    
    # Required abstract methods from VannaBase
    def add_ddl(self, ddl: str) -> str:
        """Add DDL to training data (simplified implementation)"""
        return f"DDL added: {ddl[:100]}..."
    
    def add_documentation(self, documentation: str) -> str:
        """Add documentation to training data (simplified implementation)"""
        return f"Documentation added: {documentation[:100]}..."
    
    def add_question_sql(self, question: str, sql: str) -> str:
        """Add question-SQL pair to training data (simplified implementation)"""
        return f"Question-SQL pair added: {question[:50]}... -> {sql[:50]}..."
    
    def generate_embedding(self, data: str) -> List[float]:
        """Generate embedding for data (simplified implementation)"""
        # Return a dummy embedding for now
        return [0.0] * 384
    
    def get_related_ddl(self, question: str, **kwargs) -> List[str]:
        """Get related DDL for question (simplified implementation)"""
        return []
    
    def get_related_documentation(self, question: str, **kwargs) -> List[str]:
        """Get related documentation for question (simplified implementation)"""
        return []
    
    def get_similar_question_sql(self, question: str, **kwargs) -> List[str]:
        """Get similar question-SQL pairs (simplified implementation)"""
        return []
    
    def get_training_data(self, **kwargs) -> List[Dict]:
        """Get training data (simplified implementation)"""
        return []
    
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """Remove training data (simplified implementation)"""
        return True


class SQLGenerator:
    """Main SQL generation service orchestrating Vanna AI with Groq"""
    
    def __init__(self):
        self.groq_client = GroqClient()
        self.schema_provider = SchemaProvider()
        self.vanna_adapter = None
        self.training_data = []
        self.context_cache = {}
        
    async def initialize(self):
        """Initialize the SQL generator with all dependencies"""
        try:
            # Initialize dependencies
            await self.schema_provider.initialize()
            
            # Create Vanna adapter with Groq
            config = {
                "model": settings.vanna_model_name,
                "api_key": settings.vanna_api_key,
                "path": "./vanna_data"  # Local storage for vector embeddings
            }
            
            self.vanna_adapter = GroqVannaAdapter(
                groq_client=self.groq_client,
                config=config
            )
            
            # Load training data and examples
            await self._load_training_data()
            
            logger.info("SQL generator initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize SQL generator", error=str(e))
            raise SQLGenerationError(f"Failed to initialize SQL generator: {str(e)}")
    
    async def generate_sql_from_question(
        self,
        question: str,
        context: Optional[Dict] = None,
        include_explanation: bool = True
    ) -> Dict[str, Any]:
        """
        Generate SQL query from natural language question
        
        Args:
            question: Natural language question
            context: Additional context (user preferences, filters, etc.)
            include_explanation: Whether to include query explanation
            
        Returns:
            Dictionary containing SQL query, explanation, and metadata
        """
        try:
            if not self.vanna_adapter:
                await self.initialize()
            
            logger.info("Generating SQL from question", question=question)
            
            # Get schema information
            schema_info = await self.schema_provider.get_schema_info(
                include_sample_data=False
            )
            
            # Get relevant examples based on question
            examples = await self._get_relevant_examples(question)
            
            # Build context string
            context_str = self._build_context_string(context)
            
            # Generate SQL using Vanna + Groq
            sql_query = await self.vanna_adapter.submit_prompt(
                prompt=question,
                schema_info=schema_info,
                context=context_str,
                examples=examples
            )
            
            # Generate explanation if requested
            explanation = ""
            if include_explanation:
                explanation = await self.groq_client.explain_query(sql_query, question)
            
            # Extract query metadata
            metadata = self._extract_query_metadata(sql_query, question)
            
            result = {
                "success": True,
                "sql_query": sql_query,
                "explanation": explanation,
                "question": question,
                "metadata": metadata,
                "context": context,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info("SQL generated successfully", 
                       query_length=len(sql_query),
                       has_explanation=bool(explanation))
            
            return result
            
        except Exception as e:
            logger.error("Failed to generate SQL", error=str(e), question=question)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "question": question,
                "context": context,
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def improve_sql_with_feedback(
        self,
        original_question: str,
        original_sql: str,
        feedback: str,
        execution_result: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Improve SQL query based on user feedback or execution results
        
        Args:
            original_question: Original natural language question
            original_sql: Original SQL query
            feedback: User feedback or error description
            execution_result: Results from query execution
            
        Returns:
            Improved SQL query and explanation
        """
        try:
            # Build improvement prompt
            improvement_prompt = self._build_improvement_prompt(
                original_question, original_sql, feedback, execution_result
            )
            
            # Get schema info
            schema_info = await self.schema_provider.get_schema_info()
            
            # Generate improved SQL
            improved_sql = await self.vanna_adapter.submit_prompt(
                prompt=improvement_prompt,
                schema_info=schema_info
            )
            
            # Generate explanation for improved query
            explanation = await self.groq_client.explain_query(
                improved_sql, original_question
            )
            
            return {
                "success": True,
                "improved_sql": improved_sql,
                "explanation": explanation,
                "original_sql": original_sql,
                "feedback": feedback,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error("Failed to improve SQL", error=str(e))
            return {
                "success": False,
                "error": str(e),
                "original_sql": original_sql,
                "feedback": feedback
            }
    
    async def explain_sql_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Explain a SQL query in natural language
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Dictionary containing explanation and metadata
        """
        try:
            logger.info("Explaining SQL query", query_preview=sql_query[:100])
            
            # Use Groq client to explain the query
            explanation = await self.groq_client.explain_query(sql_query, "")
            
            # Extract query metadata
            metadata = self._extract_query_metadata(sql_query, "")
            
            result = {
                "success": True,
                "explanation": explanation,
                "sql_query": sql_query,
                "metadata": metadata,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            logger.info("SQL query explained successfully")
            
            return result
            
        except Exception as e:
            logger.error("Failed to explain SQL query", error=str(e), sql=sql_query)
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "sql_query": sql_query,
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def _load_training_data(self):
        """Load training data and examples for few-shot learning"""
        try:
            # Define common business query examples
            self.training_data = [
                {
                    "question": "What are the top 10 vendors by total spending?",
                    "sql": """
                    SELECT v.vendorName, SUM(i.invoiceTotal) as total_spending
                    FROM Vendor v
                    JOIN Invoice i ON v.id = i.vendorId
                    WHERE i.invoiceTotal IS NOT NULL
                    GROUP BY v.id, v.vendorName
                    ORDER BY total_spending DESC
                    LIMIT 10;
                    """,
                    "category": "vendor_analysis"
                },
                {
                    "question": "Show monthly invoice trends for the last 12 months",
                    "sql": """
                    SELECT
                        DATE_TRUNC('month', i.invoiceDate) as month,
                        COUNT(*) as invoice_count,
                        SUM(i.invoiceTotal) as total_amount
                    FROM Invoice i
                    WHERE i.invoiceDate >= CURRENT_DATE - INTERVAL '12 months'
                    AND i.invoiceTotal IS NOT NULL
                    GROUP BY DATE_TRUNC('month', i.invoiceDate)
                    ORDER BY month;
                    """,
                    "category": "trend_analysis"
                },
                {
                    "question": "Which departments have the highest spending?",
                    "sql": """
                    SELECT
                        d.name as department_name,
                        o.name as organization_name,
                        SUM(i.invoiceTotal) as total_spending
                    FROM Department d
                    JOIN Organization o ON d.organizationId = o.id
                    JOIN Document doc ON doc.departmentId = d.id
                    JOIN Invoice i ON i.documentId = doc.id
                    WHERE i.invoiceTotal IS NOT NULL
                    GROUP BY d.id, d.name, o.name
                    ORDER BY total_spending DESC;
                    """,
                    "category": "department_analysis"
                },
                {
                    "question": "Show overdue invoices with payment details",
                    "sql": """
                    SELECT
                        i.invoiceNumber,
                        v.vendorName,
                        i.invoiceTotal,
                        p.dueDate,
                        CURRENT_DATE - p.dueDate as days_overdue
                    FROM Invoice i
                    JOIN Vendor v ON i.vendorId = v.id
                    JOIN Payment p ON p.invoiceId = i.id
                    WHERE p.dueDate < CURRENT_DATE
                    AND i.invoiceTotal IS NOT NULL
                    ORDER BY days_overdue DESC;
                    """,
                    "category": "payment_analysis"
                }
            ]
            
            logger.info("Training data loaded", examples_count=len(self.training_data))
            
        except Exception as e:
            logger.error("Failed to load training data", error=str(e))
            # Continue without training data
            self.training_data = []
    
    async def _get_relevant_examples(self, question: str, limit: int = 3) -> List[Dict]:
        """Get relevant examples based on question similarity"""
        try:
            # Simple keyword-based matching for now
            # In production, you might use vector similarity
            question_lower = question.lower()
            
            relevant_examples = []
            
            # Score examples based on keyword overlap
            for example in self.training_data:
                score = 0
                example_words = example["question"].lower().split()
                question_words = question_lower.split()
                
                # Calculate simple overlap score
                for word in question_words:
                    if word in example_words:
                        score += 1
                
                if score > 0:
                    relevant_examples.append((score, example))
            
            # Sort by score and return top examples
            relevant_examples.sort(key=lambda x: x[0], reverse=True)
            
            return [example for _, example in relevant_examples[:limit]]
            
        except Exception as e:
            logger.error("Failed to get relevant examples", error=str(e))
            return []
    
    def _build_context_string(self, context: Optional[Dict]) -> str:
        """Build context string from context dictionary"""
        if not context:
            return ""
        
        context_parts = []
        
        if context.get("date_range"):
            context_parts.append(f"Date range: {context['date_range']}")
        
        if context.get("organization"):
            context_parts.append(f"Organization: {context['organization']}")
        
        if context.get("department"):
            context_parts.append(f"Department: {context['department']}")
        
        if context.get("vendor"):
            context_parts.append(f"Vendor: {context['vendor']}")
        
        if context.get("filters"):
            context_parts.append(f"Additional filters: {context['filters']}")
        
        return " | ".join(context_parts)
    
    def _extract_query_metadata(self, sql_query: str, question: str) -> Dict[str, Any]:
        """Extract metadata from the generated SQL query"""
        metadata = {
            "query_type": "SELECT",
            "tables_involved": [],
            "has_joins": False,
            "has_aggregation": False,
            "has_grouping": False,
            "has_ordering": False,
            "estimated_complexity": "low"
        }
        
        sql_upper = sql_query.upper()
        
        # Extract table names (basic regex)
        import re
        table_pattern = r'FROM\s+(\w+)|JOIN\s+(\w+)'
        matches = re.findall(table_pattern, sql_upper)
        for match in matches:
            table = match[0] or match[1]
            if table and table not in metadata["tables_involved"]:
                metadata["tables_involved"].append(table.lower())
        
        # Check for various SQL features
        metadata["has_joins"] = "JOIN" in sql_upper
        metadata["has_aggregation"] = any(agg in sql_upper for agg in ["SUM", "COUNT", "AVG", "MAX", "MIN"])
        metadata["has_grouping"] = "GROUP BY" in sql_upper
        metadata["has_ordering"] = "ORDER BY" in sql_upper
        
        # Estimate complexity
        complexity_score = 0
        complexity_score += len(metadata["tables_involved"]) * 2
        complexity_score += sql_upper.count("JOIN") * 3
        complexity_score += sql_upper.count("SELECT") * 2
        
        if complexity_score < 5:
            metadata["estimated_complexity"] = "low"
        elif complexity_score < 15:
            metadata["estimated_complexity"] = "medium"
        else:
            metadata["estimated_complexity"] = "high"
        
        return metadata
    
    def _build_improvement_prompt(
        self,
        original_question: str,
        original_sql: str,
        feedback: str,
        execution_result: Optional[Dict] = None
    ) -> str:
        """Build prompt for SQL improvement"""
        prompt_parts = [
            f"Original question: {original_question}",
            f"Original SQL: {original_sql}",
            f"Feedback/Issue: {feedback}"
        ]
        
        if execution_result and not execution_result.get("success"):
            prompt_parts.append(f"Execution error: {execution_result.get('error', 'Unknown error')}")
        
        prompt_parts.append("Please generate an improved SQL query that addresses the feedback and resolves any issues.")
        
        return "\n\n".join(prompt_parts)
    
    async def health_check(self) -> bool:
        """Check if all components are healthy"""
        try:
            groq_healthy = await self.groq_client.health_check()
            schema_healthy = await self.schema_provider.validate_table_access("Invoice")
            
            return groq_healthy and schema_healthy
            
        except Exception as e:
            logger.error("SQL generator health check failed", error=str(e))
            return False
    
    async def close(self):
        """Close all connections and cleanup"""
        if self.schema_provider:
            await self.schema_provider.close()
        
        logger.info("SQL generator closed")