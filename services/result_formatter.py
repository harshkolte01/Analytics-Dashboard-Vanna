"""
Result formatter service
Formats query results for different output formats and use cases
"""

import json
import csv
import io
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, date
from decimal import Decimal
import structlog
from .errors import ValidationError

logger = structlog.get_logger(__name__)


class ResultFormatter:
    """Formats query results for various output formats and consumption"""
    
    def __init__(self):
        self.supported_formats = ['json', 'csv', 'table', 'chart_data', 'summary']
        
    async def format_query_result(
        self,
        query_result: Dict[str, Any],
        format_type: str = 'json',
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format query result based on the specified format type
        
        Args:
            query_result: Raw query result from SQL executor
            format_type: Output format ('json', 'csv', 'table', 'chart_data', 'summary')
            options: Additional formatting options
            
        Returns:
            Formatted result dictionary
        """
        try:
            if format_type not in self.supported_formats:
                raise ValidationError(f"Unsupported format type: {format_type}")
            
            logger.info("Formatting query result", 
                       format_type=format_type,
                       row_count=query_result.get('row_count', 0))
            
            # Clean and prepare data
            cleaned_data = self._clean_data(query_result.get('data', []))
            
            # Apply formatting based on type
            if format_type == 'json':
                formatted_result = await self._format_as_json(cleaned_data, query_result, options)
            elif format_type == 'csv':
                formatted_result = await self._format_as_csv(cleaned_data, query_result, options)
            elif format_type == 'table':
                formatted_result = await self._format_as_table(cleaned_data, query_result, options)
            elif format_type == 'chart_data':
                formatted_result = await self._format_as_chart_data(cleaned_data, query_result, options)
            elif format_type == 'summary':
                formatted_result = await self._format_as_summary(cleaned_data, query_result, options)
            else:
                formatted_result = await self._format_as_json(cleaned_data, query_result, options)
            
            # Add metadata
            formatted_result['metadata'] = {
                'format_type': format_type,
                'row_count': len(cleaned_data),
                'column_count': len(query_result.get('columns', [])),
                'execution_time': query_result.get('execution_time', 0),
                'formatted_at': datetime.now().isoformat(),
                'success': query_result.get('success', False)
            }
            
            logger.info("Result formatted successfully", format_type=format_type)
            
            return formatted_result
            
        except Exception as e:
            logger.error("Failed to format result", error=str(e), format_type=format_type)
            return {
                'success': False,
                'error': str(e),
                'format_type': format_type,
                'metadata': {
                    'formatted_at': datetime.now().isoformat(),
                    'success': False
                }
            }
    
    def _clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and normalize data for formatting"""
        cleaned_data = []
        
        for row in data:
            cleaned_row = {}
            for key, value in row.items():
                # Handle different data types
                if isinstance(value, Decimal):
                    cleaned_row[key] = float(value)
                elif isinstance(value, (date, datetime)):
                    cleaned_row[key] = value.isoformat()
                elif value is None:
                    cleaned_row[key] = None
                else:
                    cleaned_row[key] = value
            
            cleaned_data.append(cleaned_row)
        
        return cleaned_data
    
    async def _format_as_json(
        self,
        data: List[Dict[str, Any]],
        query_result: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format result as JSON"""
        return {
            'success': True,
            'format': 'json',
            'data': data,
            'columns': query_result.get('columns', []),
            'query': query_result.get('query', ''),
            'explanation': query_result.get('explanation', '')
        }
    
    async def _format_as_csv(
        self,
        data: List[Dict[str, Any]],
        query_result: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format result as CSV string"""
        if not data:
            return {
                'success': True,
                'format': 'csv',
                'data': '',
                'columns': []
            }
        
        # Create CSV string
        output = io.StringIO()
        columns = query_result.get('columns', list(data[0].keys()) if data else [])
        
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        
        for row in data:
            # Convert all values to strings for CSV
            csv_row = {}
            for col in columns:
                value = row.get(col, '')
                if value is None:
                    csv_row[col] = ''
                else:
                    csv_row[col] = str(value)
            writer.writerow(csv_row)
        
        csv_content = output.getvalue()
        output.close()
        
        return {
            'success': True,
            'format': 'csv',
            'data': csv_content,
            'columns': columns,
            'filename': f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        }
    
    async def _format_as_table(
        self,
        data: List[Dict[str, Any]],
        query_result: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format result as table structure for UI display"""
        columns = query_result.get('columns', [])
        
        # Analyze column types for better display
        column_info = []
        for col in columns:
            col_info = {
                'name': col,
                'display_name': self._format_column_name(col),
                'type': self._infer_column_type(data, col),
                'sortable': True,
                'filterable': True
            }
            column_info.append(col_info)
        
        # Format rows for table display
        formatted_rows = []
        for i, row in enumerate(data):
            formatted_row = {
                'id': i,
                'data': {}
            }
            
            for col in columns:
                value = row.get(col)
                formatted_row['data'][col] = self._format_cell_value(value, col_info['type'] if col_info else 'text')
            
            formatted_rows.append(formatted_row)
        
        return {
            'success': True,
            'format': 'table',
            'columns': column_info,
            'rows': formatted_rows,
            'pagination': {
                'total_rows': len(data),
                'page_size': options.get('page_size', 50) if options else 50,
                'current_page': 1
            }
        }
    
    async def _format_as_chart_data(
        self,
        data: List[Dict[str, Any]],
        query_result: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format result for chart visualization"""
        if not data:
            return {
                'success': True,
                'format': 'chart_data',
                'chart_type': 'empty',
                'data': []
            }
        
        columns = query_result.get('columns', [])
        
        # Determine appropriate chart type based on data structure
        chart_type = self._determine_chart_type(data, columns, options)
        
        # Format data based on chart type
        if chart_type == 'bar' or chart_type == 'column':
            chart_data = self._format_for_bar_chart(data, columns)
        elif chart_type == 'line':
            chart_data = self._format_for_line_chart(data, columns)
        elif chart_type == 'pie':
            chart_data = self._format_for_pie_chart(data, columns)
        elif chart_type == 'table':
            chart_data = self._format_for_data_table(data, columns)
        else:
            chart_data = self._format_for_generic_chart(data, columns)
        
        return {
            'success': True,
            'format': 'chart_data',
            'chart_type': chart_type,
            'data': chart_data,
            'config': self._get_chart_config(chart_type, columns)
        }
    
    async def _format_as_summary(
        self,
        data: List[Dict[str, Any]],
        query_result: Dict[str, Any],
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Format result as executive summary"""
        if not data:
            return {
                'success': True,
                'format': 'summary',
                'summary': 'No data found for the query.',
                'key_metrics': [],
                'insights': []
            }
        
        columns = query_result.get('columns', [])
        
        # Generate summary statistics
        summary_stats = self._generate_summary_stats(data, columns)
        
        # Generate key insights
        insights = self._generate_insights(data, columns, summary_stats)
        
        # Create executive summary text
        summary_text = self._create_summary_text(data, columns, summary_stats, insights)
        
        return {
            'success': True,
            'format': 'summary',
            'summary': summary_text,
            'key_metrics': summary_stats,
            'insights': insights,
            'data_preview': data[:5]  # First 5 rows as preview
        }
    
    def _format_column_name(self, column_name: str) -> str:
        """Format column name for display"""
        # Convert snake_case to Title Case
        formatted = column_name.replace('_', ' ').title()
        
        # Handle common abbreviations
        replacements = {
            'Id': 'ID',
            'Url': 'URL',
            'Api': 'API',
            'Sql': 'SQL',
            'Json': 'JSON',
            'Uuid': 'UUID'
        }
        
        for old, new in replacements.items():
            formatted = formatted.replace(old, new)
        
        return formatted
    
    def _infer_column_type(self, data: List[Dict[str, Any]], column: str) -> str:
        """Infer column data type from sample data"""
        if not data:
            return 'text'
        
        # Sample first few non-null values
        sample_values = []
        for row in data[:10]:
            value = row.get(column)
            if value is not None:
                sample_values.append(value)
        
        if not sample_values:
            return 'text'
        
        # Check types
        first_value = sample_values[0]
        
        if isinstance(first_value, (int, float, Decimal)):
            return 'number'
        elif isinstance(first_value, (date, datetime)):
            return 'date'
        elif isinstance(first_value, bool):
            return 'boolean'
        else:
            return 'text'
    
    def _format_cell_value(self, value: Any, column_type: str) -> str:
        """Format individual cell value for display"""
        if value is None:
            return ''
        
        if column_type == 'number':
            if isinstance(value, float):
                return f"{value:,.2f}"
            else:
                return f"{value:,}"
        elif column_type == 'date':
            if isinstance(value, str):
                return value
            return value.strftime('%Y-%m-%d %H:%M:%S') if hasattr(value, 'strftime') else str(value)
        else:
            return str(value)
    
    def _determine_chart_type(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Determine appropriate chart type based on data structure"""
        if options and options.get('chart_type'):
            return options['chart_type']
        
        if len(columns) == 2:
            # Two columns - likely category and value
            return 'bar'
        elif len(columns) > 2 and any('date' in col.lower() or 'time' in col.lower() for col in columns):
            # Multiple columns with date/time - line chart
            return 'line'
        elif len(data) <= 10 and len(columns) == 2:
            # Small dataset with 2 columns - pie chart
            return 'pie'
        else:
            # Default to table for complex data
            return 'table'
    
    def _format_for_bar_chart(self, data: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
        """Format data for bar/column chart"""
        if len(columns) < 2:
            return []
        
        category_col = columns[0]
        value_col = columns[1]
        
        chart_data = []
        for row in data:
            chart_data.append({
                'category': str(row.get(category_col, '')),
                'value': float(row.get(value_col, 0)) if row.get(value_col) is not None else 0
            })
        
        return chart_data
    
    def _format_for_line_chart(self, data: List[Dict[str, Any]], columns: List[str]) -> Dict[str, Any]:
        """Format data for line chart"""
        # Find date/time column
        date_col = None
        for col in columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        if not date_col:
            date_col = columns[0]
        
        value_cols = [col for col in columns if col != date_col]
        
        chart_data = {
            'labels': [],
            'datasets': []
        }
        
        # Extract labels (x-axis)
        for row in data:
            label = str(row.get(date_col, ''))
            chart_data['labels'].append(label)
        
        # Extract datasets (y-axis)
        for col in value_cols:
            dataset = {
                'label': self._format_column_name(col),
                'data': []
            }
            
            for row in data:
                value = row.get(col, 0)
                dataset['data'].append(float(value) if value is not None else 0)
            
            chart_data['datasets'].append(dataset)
        
        return chart_data
    
    def _format_for_pie_chart(self, data: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
        """Format data for pie chart"""
        if len(columns) < 2:
            return []
        
        label_col = columns[0]
        value_col = columns[1]
        
        chart_data = []
        for row in data:
            chart_data.append({
                'label': str(row.get(label_col, '')),
                'value': float(row.get(value_col, 0)) if row.get(value_col) is not None else 0
            })
        
        return chart_data
    
    def _format_for_data_table(self, data: List[Dict[str, Any]], columns: List[str]) -> Dict[str, Any]:
        """Format data for data table display"""
        return {
            'columns': columns,
            'rows': data
        }
    
    def _format_for_generic_chart(self, data: List[Dict[str, Any]], columns: List[str]) -> Dict[str, Any]:
        """Format data for generic chart display"""
        return {
            'columns': columns,
            'data': data
        }
    
    def _get_chart_config(self, chart_type: str, columns: List[str]) -> Dict[str, Any]:
        """Get chart configuration based on type"""
        base_config = {
            'responsive': True,
            'maintainAspectRatio': False
        }
        
        if chart_type == 'bar':
            base_config.update({
                'scales': {
                    'y': {
                        'beginAtZero': True
                    }
                }
            })
        elif chart_type == 'line':
            base_config.update({
                'scales': {
                    'y': {
                        'beginAtZero': True
                    }
                }
            })
        elif chart_type == 'pie':
            base_config.update({
                'plugins': {
                    'legend': {
                        'position': 'right'
                    }
                }
            })
        
        return base_config
    
    def _generate_summary_stats(self, data: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
        """Generate summary statistics for the data"""
        stats = []
        
        for col in columns:
            col_data = [row.get(col) for row in data if row.get(col) is not None]
            
            if not col_data:
                continue
            
            col_stat = {
                'column': col,
                'display_name': self._format_column_name(col),
                'count': len(col_data),
                'null_count': len(data) - len(col_data)
            }
            
            # Numeric statistics
            if all(isinstance(x, (int, float, Decimal)) for x in col_data):
                numeric_data = [float(x) for x in col_data]
                col_stat.update({
                    'type': 'numeric',
                    'min': min(numeric_data),
                    'max': max(numeric_data),
                    'avg': sum(numeric_data) / len(numeric_data),
                    'sum': sum(numeric_data)
                })
            else:
                col_stat.update({
                    'type': 'categorical',
                    'unique_count': len(set(str(x) for x in col_data))
                })
            
            stats.append(col_stat)
        
        return stats
    
    def _generate_insights(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        summary_stats: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate insights from the data"""
        insights = []
        
        # Data volume insight
        insights.append(f"Query returned {len(data)} records across {len(columns)} columns.")
        
        # Numeric insights
        numeric_cols = [stat for stat in summary_stats if stat.get('type') == 'numeric']
        if numeric_cols:
            for stat in numeric_cols[:3]:  # Top 3 numeric columns
                col_name = stat['display_name']
                if stat.get('sum'):
                    insights.append(f"Total {col_name}: {stat['sum']:,.2f}")
                if stat.get('avg'):
                    insights.append(f"Average {col_name}: {stat['avg']:,.2f}")
        
        # Categorical insights
        categorical_cols = [stat for stat in summary_stats if stat.get('type') == 'categorical']
        if categorical_cols:
            for stat in categorical_cols[:2]:  # Top 2 categorical columns
                col_name = stat['display_name']
                insights.append(f"{col_name} has {stat['unique_count']} unique values.")
        
        return insights
    
    def _create_summary_text(
        self,
        data: List[Dict[str, Any]],
        columns: List[str],
        summary_stats: List[Dict[str, Any]],
        insights: List[str]
    ) -> str:
        """Create executive summary text"""
        summary_parts = [
            f"The query analysis reveals {len(data)} records with the following key findings:",
            "",
            "Key Insights:",
        ]
        
        for insight in insights[:5]:  # Top 5 insights
            summary_parts.append(f"• {insight}")
        
        return "\n".join(summary_parts)