# 🤖 Vanna AI Service

FastAPI-based AI service that converts natural language queries into SQL and executes them against the analytics database using Groq AI and Vanna framework.

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "AI Service Components"
        API[FastAPI Application]
        NL[Natural Language Parser]
        SQL[SQL Generator]
        EXEC[SQL Executor]
        FORMAT[Result Formatter]
    end
    
    subgraph "External Services"
        GROQ[Groq AI API]
        DB[(PostgreSQL Database)]
        EXPRESS[Express.js API]
    end
    
    API --> NL
    NL --> SQL
    SQL --> GROQ
    SQL --> EXEC
    EXEC --> DB
    EXEC --> FORMAT
    EXPRESS --> API
```

## 🛠️ Technology Stack

- **Runtime**: Python 3.11+
- **Framework**: FastAPI with Uvicorn
- **AI Provider**: Groq API (llama-3.1-70b-versatile)
- **SQL Framework**: Vanna AI
- **Database**: PostgreSQL with asyncpg
- **Validation**: Pydantic
- **Process Management**: PM2 (production)

## 📁 Project Structure

```
apps/services/vanna/
├── app/
│   ├── main.py              # FastAPI application entry point
│   ├── config.py            # Configuration management
│   ├── dependencies.py      # Dependency injection
│   └── routes/
│       └── nl_to_sql.py     # Natural language to SQL routes
│
├── services/
│   ├── __init__.py
│   ├── auth.py              # Authentication service
│   ├── audit_logger.py      # Audit logging
│   ├── cache.py             # Caching service
│   ├── errors.py            # Error handling
│   ├── groq_client.py       # Groq AI client
│   ├── metrics.py           # Performance metrics
│   ├── nl_parser.py         # Natural language parsing
│   ├── result_formatter.py  # Result formatting
│   ├── schema_provider.py   # Database schema provider
│   ├── sql_executor.py      # SQL execution
│   ├── sql_generator.py     # SQL generation
│   └── sql_validator.py     # SQL validation
│
├── tests/
│   ├── integration/         # Integration tests
│   └── unit/               # Unit tests
│
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL 15+
- Groq API key

### Installation
```bash
cd apps/services/vanna

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
```bash
cp .env.example .env
```

Configure your `.env` file:
```env
ENVIRONMENT=development
PORT=8001
DATABASE_URL=postgresql://username:password@localhost:5432/analytics_dashboard
GROQ_API_KEY=your_groq_api_key_here
LOG_LEVEL=INFO
CACHE_TTL=3600
MAX_QUERY_TIMEOUT=120
```

### Development
```bash
# Start development server
python -m uvicorn app.main:app --reload --port 8001

# Run with specific host
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Production
```bash
# Start with Gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8001

# Start with PM2
pm2 start "python -m uvicorn app.main:app --host 0.0.0.0 --port 8001" --name "analytics-vanna"
```

## 📚 API Documentation

### Base URL
- **Development**: `http://localhost:8001`
- **Production**: `https://your-domain.com/vanna`

### Interactive Documentation
- **Swagger UI**: `http://localhost:8001/docs`
- **ReDoc**: `http://localhost:8001/redoc`

## 🔗 API Endpoints

### 🤖 Natural Language to SQL

#### Convert Natural Language to SQL
```http
POST /nl-to-sql
Content-Type: application/json
```

**Request Body:**
```json
{
  "question": "Show me the top 5 vendors by total spend in the last 6 months",
  "context": {
    "timeframe": "6months",
    "limit": 5,
    "user_id": "optional-user-id"
  },
  "options": {
    "execute_query": true,
    "explain_query": true,
    "validate_sql": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "question": "Show me the top 5 vendors by total spend in the last 6 months",
    "generated_sql": "SELECT v.\"vendorName\", SUM(i.\"invoiceTotal\") as total_spend FROM \"Vendor\" v JOIN \"Invoice\" i ON v.id = i.\"vendorId\" WHERE i.\"invoiceDate\" >= CURRENT_DATE - INTERVAL '6 months' GROUP BY v.id, v.\"vendorName\" ORDER BY total_spend DESC LIMIT 5",
    "explanation": "This query retrieves the top 5 vendors by total spending in the last 6 months by joining the Vendor and Invoice tables, filtering by date, grouping by vendor, and ordering by total spend.",
    "results": [
      {
        "vendorName": "Tech Solutions Inc",
        "total_spend": "125000.00"
      },
      {
        "vendorName": "Office Supplies Co",
        "total_spend": "89500.50"
      }
    ],
    "execution_success": true,
    "execution_time_ms": 245,
    "row_count": 5,
    "query_complexity": "medium",
    "tables_involved": ["Vendor", "Invoice"],
    "confidence_score": 0.95
  },
  "metadata": {
    "model_used": "llama-3.1-70b-versatile",
    "processing_time_ms": 1250,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

#### Explain SQL Query
```http
POST /explain-sql
Content-Type: application/json
```

**Request Body:**
```json
{
  "sql": "SELECT v.\"vendorName\", SUM(i.\"invoiceTotal\") as total_spend FROM \"Vendor\" v JOIN \"Invoice\" i ON v.id = i.\"vendorId\" WHERE i.\"invoiceDate\" >= CURRENT_DATE - INTERVAL '6 months' GROUP BY v.id, v.\"vendorName\" ORDER BY total_spend DESC LIMIT 5"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "explanation": "This query retrieves vendor spending data by joining vendors with their invoices, filtering for the last 6 months, grouping by vendor, and returning the top 5 by total spend.",
    "complexity": "medium",
    "tables_involved": ["Vendor", "Invoice"],
    "operations": ["JOIN", "WHERE", "GROUP BY", "ORDER BY", "LIMIT"],
    "estimated_rows": 5,
    "performance_notes": "Query should perform well with proper indexing on invoiceDate and vendorId"
  }
}
```

#### Validate SQL Query
```http
POST /validate-sql
Content-Type: application/json
```

**Request Body:**
```json
{
  "sql": "SELECT * FROM Vendor WHERE vendorName = 'Test'"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "is_valid": true,
    "syntax_errors": [],
    "security_issues": [],
    "performance_warnings": [
      "SELECT * can be inefficient for large tables"
    ],
    "suggestions": [
      "Consider specifying only needed columns instead of SELECT *"
    ]
  }
}
```

### 📊 Database Schema Information

#### Get Database Schema
```http
GET /schema
```

**Response:**
```json
{
  "success": true,
  "data": {
    "tables": [
      {
        "name": "Vendor",
        "columns": [
          {
            "name": "id",
            "type": "uuid",
            "nullable": false,
            "primary_key": true
          },
          {
            "name": "vendorName",
            "type": "varchar",
            "nullable": false
          }
        ],
        "relationships": [
          {
            "type": "one_to_many",
            "target_table": "Invoice",
            "foreign_key": "vendorId"
          }
        ]
      }
    ],
    "total_tables": 15,
    "last_updated": "2024-01-15T10:00:00Z"
  }
}
```

#### Get Table Information
```http
GET /schema/{table_name}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "table_name": "Invoice",
    "columns": [
      {
        "name": "id",
        "type": "uuid",
        "nullable": false,
        "primary_key": true,
        "description": "Unique invoice identifier"
      },
      {
        "name": "invoiceTotal",
        "type": "decimal",
        "nullable": true,
        "description": "Total invoice amount including tax"
      }
    ],
    "indexes": [
      {
        "name": "idx_invoice_date",
        "columns": ["invoiceDate"],
        "type": "btree"
      }
    ],
    "row_count": 1250,
    "sample_data": [
      {
        "id": "uuid-example",
        "invoiceTotal": "2500.00",
        "invoiceDate": "2024-01-15"
      }
    ]
  }
}
```

### 🔍 Query Analysis

#### Analyze Query Performance
```http
POST /analyze-query
Content-Type: application/json
```

**Request Body:**
```json
{
  "sql": "SELECT v.vendorName, COUNT(*) FROM Vendor v JOIN Invoice i ON v.id = i.vendorId GROUP BY v.vendorName"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "execution_plan": "Hash Join -> GroupAggregate -> Seq Scan",
    "estimated_cost": 1250.45,
    "estimated_rows": 85,
    "execution_time_ms": 125,
    "index_usage": [
      {
        "table": "Invoice",
        "index": "idx_vendor_id",
        "usage": "used"
      }
    ],
    "optimization_suggestions": [
      "Consider adding index on Vendor.vendorName for better GROUP BY performance"
    ]
  }
}
```

### 📈 Service Metrics

#### Get Service Health
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "version": "1.0.0",
  "services": {
    "database": {
      "status": "connected",
      "response_time_ms": 5
    },
    "groq_api": {
      "status": "available",
      "response_time_ms": 250
    },
    "cache": {
      "status": "active",
      "hit_rate": 0.85
    }
  },
  "performance": {
    "queries_processed": 1250,
    "avg_response_time_ms": 850,
    "success_rate": 0.96
  }
}
```

#### Get Service Metrics
```http
GET /metrics
```

**Response:**
```json
{
  "success": true,
  "data": {
    "total_queries": 1250,
    "successful_queries": 1200,
    "failed_queries": 50,
    "avg_processing_time_ms": 850,
    "avg_execution_time_ms": 245,
    "cache_hit_rate": 0.85,
    "popular_query_types": [
      {
        "type": "vendor_analysis",
        "count": 450,
        "percentage": 36.0
      },
      {
        "type": "invoice_trends",
        "count": 320,
        "percentage": 25.6
      }
    ],
    "error_distribution": [
      {
        "error_type": "sql_generation_error",
        "count": 25,
        "percentage": 50.0
      },
      {
        "error_type": "execution_timeout",
        "count": 15,
        "percentage": 30.0
      }
    ]
  }
}
```

## 🔧 Service Components

### 🧠 Natural Language Parser
Processes natural language queries and extracts intent, entities, and context.

**Features:**
- Intent classification (analytics, comparison, trend analysis)
- Entity extraction (dates, vendors, amounts)
- Context understanding (timeframes, filters)
- Query complexity assessment

### 🔄 SQL Generator
Converts parsed natural language into optimized SQL queries using Groq AI.

**Features:**
- Schema-aware query generation
- Automatic JOIN optimization
- Date range handling
- Aggregation logic
- Security validation (SQL injection prevention)

### ⚡ SQL Executor
Executes generated SQL queries against the PostgreSQL database.

**Features:**
- Connection pooling
- Query timeout management
- Result set limiting
- Performance monitoring
- Error handling and recovery

### 📊 Result Formatter
Formats query results for optimal presentation in the frontend.

**Features:**
- Data type conversion
- Number formatting (currency, percentages)
- Date formatting
- Chart data preparation
- Export format generation

### 🔒 Security & Validation
Comprehensive security measures to prevent malicious queries.

**Features:**
- SQL injection prevention
- Query whitelist validation
- Resource usage limits
- Rate limiting
- Audit logging

## 🛠️ Development

### Available Scripts
```bash
# Development
python -m uvicorn app.main:app --reload --port 8001

# Testing
python -m pytest                    # Run all tests
python -m pytest tests/unit/        # Run unit tests
python -m pytest tests/integration/ # Run integration tests
python -m pytest --cov=app         # Run with coverage

# Code Quality
black .                             # Format code
flake8 .                           # Lint code
mypy app/                          # Type checking

# Database
python -c "from services.schema_provider import update_schema; update_schema()"
```

### Configuration

#### Environment Variables
```env
# Service Configuration
ENVIRONMENT=development|production
PORT=8001
LOG_LEVEL=DEBUG|INFO|WARNING|ERROR

# Database
DATABASE_URL=postgresql://user:password@host:port/database

# AI Service
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.1-70b-versatile
GROQ_TIMEOUT=30

# Performance
CACHE_TTL=3600
MAX_QUERY_TIMEOUT=120
MAX_RESULT_ROWS=10000
CONNECTION_POOL_SIZE=10

# Security
ENABLE_QUERY_VALIDATION=true
ENABLE_RATE_LIMITING=true
MAX_REQUESTS_PER_MINUTE=60
```

#### Logging Configuration
```python
# Custom logging setup
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/vanna.log'),
        logging.StreamHandler()
    ]
)
```

### Adding New Features

#### 1. Custom Query Processors
```python
# services/custom_processor.py
from typing import Dict, Any
from .nl_parser import NLParser

class CustomQueryProcessor:
    def __init__(self):
        self.parser = NLParser()
    
    def process_query(self, question: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Custom processing logic
        return {
            "processed_question": question,
            "extracted_entities": {},
            "query_intent": "custom"
        }
```

#### 2. Custom Result Formatters
```python
# services/custom_formatter.py
from typing import List, Dict, Any

class CustomResultFormatter:
    def format_results(self, results: List[Dict[str, Any]], query_type: str) -> Dict[str, Any]:
        # Custom formatting logic
        return {
            "formatted_data": results,
            "chart_config": {},
            "summary": {}
        }
```

### Testing

#### Unit Tests
```python
# tests/unit/test_sql_generator.py
import pytest
from services.sql_generator import SQLGenerator

@pytest.fixture
def sql_generator():
    return SQLGenerator()

def test_generate_vendor_query(sql_generator):
    question = "Show me top 5 vendors"
    result = sql_generator.generate_sql(question)
    
    assert "SELECT" in result["sql"]
    assert "Vendor" in result["sql"]
    assert "LIMIT 5" in result["sql"]
```

#### Integration Tests
```python
# tests/integration/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_nl_to_sql_endpoint():
    response = client.post("/nl-to-sql", json={
        "question": "Show me top vendors",
        "options": {"execute_query": False}
    })
    
    assert response.status_code == 200
    assert response.json()["success"] is True
```

## 🚀 Production Deployment

### Environment Setup
```env
ENVIRONMENT=production
PORT=8001
DATABASE_URL=postgresql://user:password@localhost:5432/analytics_dashboard
GROQ_API_KEY=your_production_groq_key
LOG_LEVEL=INFO
CACHE_TTL=7200
MAX_QUERY_TIMEOUT=60
```

### PM2 Configuration
```bash
# Start with PM2
pm2 start "python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 2" --name "analytics-vanna"

# PM2 ecosystem file
module.exports = {
  apps: [{
    name: 'analytics-vanna',
    script: 'python',
    args: '-m uvicorn app.main:app --host 0.0.0.0 --port 8001 --workers 2',
    instances: 1,
    exec_mode: 'fork',
    env: {
      ENVIRONMENT: 'production'
    }
  }]
};
```

### Performance Optimization

#### Connection Pooling
```python
# Optimized database connection pool
DATABASE_POOL_CONFIG = {
    "min_size": 5,
    "max_size": 20,
    "max_queries": 50000,
    "max_inactive_connection_lifetime": 300
}
```

#### Caching Strategy
```python
# Redis caching for frequent queries
CACHE_CONFIG = {
    "ttl": 3600,  # 1 hour
    "max_size": 1000,
    "eviction_policy": "lru"
}
```

## 🔍 Monitoring & Troubleshooting

### Health Monitoring
```bash
# Check service health
curl http://localhost:8001/health

# Monitor with PM2
pm2 monit

# View logs
pm2 logs analytics-vanna
tail -f logs/vanna.log
```

### Common Issues

1. **Groq API Timeout**
   - Check API key validity
   - Verify network connectivity
   - Review rate limiting

2. **Database Connection Issues**
   - Verify DATABASE_URL format
   - Check PostgreSQL service status
   - Review connection pool settings

3. **SQL Generation Errors**
   - Check schema synchronization
   - Review query complexity
   - Validate input parameters

4. **Performance Issues**
   - Monitor query execution times
   - Check database indexes
   - Review caching effectiveness

### Debug Mode
```bash
# Enable debug logging
LOG_LEVEL=DEBUG python -m uvicorn app.main:app --reload

# Profile performance
python -m cProfile -o profile.stats -m uvicorn app.main:app
```

## 📊 Performance Metrics

### Typical Performance
- **Query Processing**: 500-1500ms
- **SQL Generation**: 200-800ms
- **Database Execution**: 50-500ms
- **Result Formatting**: 10-100ms

### Optimization Tips
1. Use database indexes effectively
2. Implement query result caching
3. Optimize SQL generation prompts
4. Use connection pooling
5. Monitor and tune timeout settings

---

**Vanna AI Service Documentation last updated: January 2024**