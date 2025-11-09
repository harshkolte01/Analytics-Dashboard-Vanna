"""
Database schema provider service
Extracts and provides database schema metadata to the LLM
"""

import asyncio
from typing import Dict, List, Optional, Any
import structlog
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from app.config import get_database_config
from .errors import DatabaseError, SchemaError

logger = structlog.get_logger(__name__)


class SchemaProvider:
    """Provides database schema information for SQL generation"""
    
    def __init__(self):
        self.config = get_database_config()
        self.engine = None
        self.async_engine = None
        self._schema_cache = {}
        self._cache_timestamp = None
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Create async engine for schema operations
            db_url = self.config["url"]
            if db_url.startswith("postgresql://"):
                async_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            else:
                async_url = db_url
                
            self.async_engine = create_async_engine(
                async_url,
                pool_size=self.config["pool_size"],
                max_overflow=self.config["max_overflow"],
                echo=self.config["echo"]
            )
            
            logger.info("Schema provider initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize schema provider", error=str(e))
            raise DatabaseError(f"Failed to initialize database connection: {str(e)}")
    
    async def get_schema_info(self, include_sample_data: bool = False) -> str:
        """
        Get comprehensive schema information as formatted string
        
        Args:
            include_sample_data: Whether to include sample data from tables
            
        Returns:
            Formatted schema information string
        """
        try:
            if not self.async_engine:
                await self.initialize()
            
            schema_info = []
            
            # Get table information
            tables = await self._get_table_info()
            
            schema_info.append("=== DATABASE SCHEMA ===\n")
            
            for table_name, table_info in tables.items():
                schema_info.append(f'Table: "{table_name}"')
                schema_info.append(f"Description: {table_info.get('description', 'Business data table')}")
                
                # Add columns
                schema_info.append("Columns:")
                for col in table_info['columns']:
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    default = f" DEFAULT {col['default']}" if col['default'] else ""
                    # Quote column names that contain capital letters
                    col_name = f'"{col["name"]}"' if any(c.isupper() for c in col["name"]) else col["name"]
                    schema_info.append(f"  - {col_name}: {col['type']} {nullable}{default}")
                    if col.get('comment'):
                        schema_info.append(f"    Comment: {col['comment']}")
                
                # Add relationships
                if table_info.get('foreign_keys'):
                    schema_info.append("Foreign Keys:")
                    for fk in table_info['foreign_keys']:
                        fk_col = f'"{fk["column"]}"' if any(c.isupper() for c in fk["column"]) else fk["column"]
                        ref_table = f'"{fk["referenced_table"]}"' if any(c.isupper() for c in fk["referenced_table"]) else fk["referenced_table"]
                        ref_col = f'"{fk["referenced_column"]}"' if any(c.isupper() for c in fk["referenced_column"]) else fk["referenced_column"]
                        schema_info.append(f"  - {fk_col} -> {ref_table}.{ref_col}")
                
                # Add indexes
                if table_info.get('indexes'):
                    schema_info.append("Indexes:")
                    for idx in table_info['indexes']:
                        schema_info.append(f"  - {idx['name']}: {', '.join(idx['columns'])}")
                
                # Add sample data if requested
                if include_sample_data:
                    sample_data = await self._get_sample_data(table_name)
                    if sample_data:
                        schema_info.append("Sample Data (first 3 rows):")
                        for row in sample_data[:3]:
                            schema_info.append(f"  {dict(row)}")
                
                schema_info.append("")  # Empty line between tables
            
            # Add business context
            schema_info.append(self._get_business_context())
            
            return "\n".join(schema_info)
            
        except Exception as e:
            logger.error("Failed to get schema info", error=str(e))
            raise SchemaError(f"Failed to retrieve schema information: {str(e)}")
    
    async def _get_table_info(self) -> Dict[str, Dict]:
        """Get detailed information about all tables"""
        tables_info = {}
        
        async with self.async_engine.begin() as conn:
            # Get table names (PostgreSQL compatible)
            result = await conn.execute(text("""
                SELECT
                    t.table_name,
                    COALESCE(obj_description(c.oid), '') as table_comment
                FROM information_schema.tables t
                LEFT JOIN pg_class c ON c.relname = t.table_name
                LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE t.table_schema = 'public'
                AND t.table_type = 'BASE TABLE'
                AND (n.nspname = 'public' OR n.nspname IS NULL)
                ORDER BY t.table_name
            """))
            
            table_names = [(row[0], row[1] or '') for row in result]
            
            for table_name, table_comment in table_names:
                tables_info[table_name] = {
                    'description': table_comment or self._get_table_description(table_name),
                    'columns': await self._get_column_info(conn, table_name),
                    'foreign_keys': await self._get_foreign_keys(conn, table_name),
                    'indexes': await self._get_indexes(conn, table_name)
                }
        
        return tables_info
    
    async def _get_column_info(self, conn, table_name: str) -> List[Dict]:
        """Get column information for a table"""
        result = await conn.execute(text("""
            SELECT 
                column_name,
                data_type,
                is_nullable,
                column_default,
                character_maximum_length,
                numeric_precision,
                numeric_scale,
                col_description(pgc.oid, ordinal_position) as column_comment
            FROM information_schema.columns isc
            LEFT JOIN pg_class pgc ON pgc.relname = isc.table_name
            WHERE table_schema = 'public' 
            AND table_name = :table_name
            ORDER BY ordinal_position
        """), {"table_name": table_name})
        
        columns = []
        for row in result:
            col_type = row[1]
            if row[4]:  # character_maximum_length
                col_type += f"({row[4]})"
            elif row[5] and row[6]:  # numeric_precision and scale
                col_type += f"({row[5]},{row[6]})"
            
            columns.append({
                'name': row[0],
                'type': col_type,
                'nullable': row[2] == 'YES',
                'default': row[3],
                'comment': row[7]
            })
        
        return columns
    
    async def _get_foreign_keys(self, conn, table_name: str) -> List[Dict]:
        """Get foreign key information for a table"""
        result = await conn.execute(text("""
            SELECT
                kcu.column_name,
                ccu.table_name AS referenced_table,
                ccu.column_name AS referenced_column
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu 
                ON tc.constraint_name = kcu.constraint_name
            JOIN information_schema.constraint_column_usage ccu 
                ON ccu.constraint_name = tc.constraint_name
            WHERE tc.constraint_type = 'FOREIGN KEY'
            AND tc.table_name = :table_name
        """), {"table_name": table_name})
        
        return [
            {
                'column': row[0],
                'referenced_table': row[1],
                'referenced_column': row[2]
            }
            for row in result
        ]
    
    async def _get_indexes(self, conn, table_name: str) -> List[Dict]:
        """Get index information for a table"""
        result = await conn.execute(text("""
            SELECT
                i.relname as index_name,
                array_agg(a.attname ORDER BY c.ordinality) as column_names
            FROM pg_class t
            JOIN pg_index ix ON t.oid = ix.indrelid
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN unnest(ix.indkey) WITH ORDINALITY c(attnum, ordinality) ON true
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = c.attnum
            WHERE t.relname = :table_name
            AND i.relname NOT LIKE '%_pkey'
            GROUP BY i.relname
        """), {"table_name": table_name})
        
        return [
            {
                'name': row[0],
                'columns': row[1]
            }
            for row in result
        ]
    
    async def _get_sample_data(self, table_name: str, limit: int = 3) -> List[Dict]:
        """Get sample data from a table"""
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(
                    text(f"SELECT * FROM {table_name} LIMIT :limit"),
                    {"limit": limit}
                )
                return [dict(row._mapping) for row in result]
        except Exception as e:
            logger.warning(f"Failed to get sample data for {table_name}", error=str(e))
            return []
    
    def _get_table_description(self, table_name: str) -> str:
        """Get business description for a table based on its name"""
        descriptions = {
            'organization': 'Contains organization/company information',
            'department': 'Department information within organizations',
            'user': 'User accounts and profile information',
            'document': 'Uploaded documents and their metadata',
            'vendor': 'Vendor/supplier information and details',
            'customer': 'Customer information and details',
            'invoice': 'Invoice records with amounts and dates',
            'payment': 'Payment terms and banking information',
            'lineitem': 'Individual line items from invoices',
            'extractionaudit': 'Audit trail for data extraction processes',
            'entityalias': 'Alternative names/aliases for entities'
        }
        return descriptions.get(table_name.lower(), 'Business data table')
    
    def _get_business_context(self) -> str:
        """Get business context information for the schema"""
        return """
=== BUSINESS CONTEXT ===

This is a financial analytics database for invoice and vendor management.

Key Business Concepts:
- Organizations have multiple departments
- Documents contain invoices and related financial data
- Vendors supply goods/services and send invoices
- Invoices contain line items with quantities and prices
- Payments track due dates and payment terms
- All monetary values are in Decimal format for precision

Common Query Patterns:
- Vendor spending analysis: JOIN vendor, invoice tables
- Invoice trends: GROUP BY date periods on invoice table
- Department spending: JOIN organization, department, document, invoice
- Payment analysis: JOIN invoice, payment tables
- Line item analysis: JOIN invoice, lineitem tables

Date Fields:
- Use invoiceDate for invoice timing analysis
- Use createdAt for document processing timing
- Use dueDate for payment analysis

IMPORTANT SQL REQUIREMENTS:
- ALWAYS use double quotes around table names (e.g., "Invoice", "Vendor")
- ALWAYS use double quotes around column names with capital letters (e.g., "invoiceDate", "invoiceTotal")
- This is required because PostgreSQL identifiers are case-sensitive when quoted
"""
    
    async def get_table_names(self) -> List[str]:
        """Get list of all table names"""
        try:
            async with self.async_engine.begin() as conn:
                result = await conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_type = 'BASE TABLE'
                    ORDER BY table_name
                """))
                return [row[0] for row in result]
        except Exception as e:
            logger.error("Failed to get table names", error=str(e))
            raise SchemaError(f"Failed to retrieve table names: {str(e)}")
    
    async def validate_table_access(self, table_name: str) -> bool:
        """Validate that a table exists and is accessible"""
        try:
            table_names = await self.get_table_names()
            return table_name.lower() in [t.lower() for t in table_names]
        except Exception:
            return False
    
    async def close(self):
        """Close database connections"""
        if self.async_engine:
            await self.async_engine.dispose()
            logger.info("Schema provider connections closed")