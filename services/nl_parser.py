"""
Natural language parser service
Processes and analyzes natural language queries for better SQL generation
"""

import re
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import structlog
from .errors import ValidationError

logger = structlog.get_logger(__name__)


class NaturalLanguageParser:
    """Parses and analyzes natural language queries"""
    
    def __init__(self):
        self.date_patterns = [
            (r'\blast\s+(\d+)\s+(day|week|month|year)s?', self._parse_relative_date),
            (r'\bthis\s+(day|week|month|year)', self._parse_current_period),
            (r'\byesterday\b', lambda: datetime.now() - timedelta(days=1)),
            (r'\btoday\b', lambda: datetime.now()),
            (r'\bthis\s+week\b', self._parse_this_week),
            (r'\bthis\s+month\b', self._parse_this_month),
            (r'\bthis\s+year\b', self._parse_this_year),
            (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', self._parse_iso_date),
            (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', self._parse_us_date),
        ]
        
        self.amount_patterns = [
            (r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)', self._parse_dollar_amount),
            (r'\b(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?', self._parse_dollar_amount),
            (r'\bover\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', self._parse_amount_threshold),
            (r'\bunder\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', self._parse_amount_threshold),
            (r'\bmore\s+than\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', self._parse_amount_threshold),
            (r'\bless\s+than\s+\$?(\d+(?:,\d{3})*(?:\.\d{2})?)', self._parse_amount_threshold),
        ]
        
        self.aggregation_keywords = {
            'total': ['total', 'sum', 'aggregate'],
            'average': ['average', 'avg', 'mean'],
            'count': ['count', 'number of', 'how many'],
            'maximum': ['max', 'maximum', 'highest', 'largest'],
            'minimum': ['min', 'minimum', 'lowest', 'smallest'],
        }
        
        self.time_keywords = {
            'trend': ['trend', 'over time', 'monthly', 'weekly', 'daily', 'yearly'],
            'comparison': ['compare', 'vs', 'versus', 'compared to'],
            'growth': ['growth', 'increase', 'decrease', 'change'],
        }
        
        self.entity_keywords = {
            'vendor': ['vendor', 'supplier', 'company'],
            'invoice': ['invoice', 'bill', 'payment'],
            'department': ['department', 'division', 'team'],
            'organization': ['organization', 'org', 'company'],
            'customer': ['customer', 'client'],
        }
    
    async def parse_query(self, question: str) -> Dict[str, Any]:
        """
        Parse natural language query and extract structured information
        
        Args:
            question: Natural language question
            
        Returns:
            Parsed query information including entities, dates, amounts, etc.
        """
        try:
            logger.info("Parsing natural language query", question=question)
            
            # Clean and normalize the question
            normalized_question = self._normalize_question(question)
            
            # Extract various components
            parsed_info = {
                "original_question": question,
                "normalized_question": normalized_question,
                "entities": self._extract_entities(normalized_question),
                "dates": self._extract_dates(normalized_question),
                "amounts": self._extract_amounts(normalized_question),
                "aggregations": self._extract_aggregations(normalized_question),
                "time_analysis": self._extract_time_analysis(normalized_question),
                "filters": self._extract_filters(normalized_question),
                "intent": self._classify_intent(normalized_question),
                "complexity": self._assess_complexity(normalized_question)
            }
            
            # Generate context for SQL generation
            parsed_info["context"] = self._build_context(parsed_info)
            
            logger.info("Query parsed successfully", 
                       intent=parsed_info["intent"],
                       entities=len(parsed_info["entities"]),
                       complexity=parsed_info["complexity"])
            
            return parsed_info
            
        except Exception as e:
            logger.error("Failed to parse query", error=str(e), question=question)
            raise ValidationError(f"Failed to parse natural language query: {str(e)}")
    
    def _normalize_question(self, question: str) -> str:
        """Normalize the question for better processing"""
        # Convert to lowercase
        normalized = question.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove punctuation at the end
        normalized = re.sub(r'[?!.]+$', '', normalized)
        
        return normalized
    
    def _extract_entities(self, question: str) -> Dict[str, List[str]]:
        """Extract business entities mentioned in the question"""
        entities = {}
        
        for entity_type, keywords in self.entity_keywords.items():
            found_entities = []
            for keyword in keywords:
                if keyword in question:
                    found_entities.append(keyword)
            
            if found_entities:
                entities[entity_type] = found_entities
        
        return entities
    
    def _extract_dates(self, question: str) -> Dict[str, Any]:
        """Extract date-related information from the question"""
        dates = {
            "relative_dates": [],
            "absolute_dates": [],
            "date_ranges": []
        }
        
        for pattern, parser in self.date_patterns:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                try:
                    if callable(parser):
                        parsed_date = parser()
                        if isinstance(parsed_date, datetime):
                            dates["absolute_dates"].append({
                                "date": parsed_date.isoformat(),
                                "original_text": match.group(0)
                            })
                    else:
                        # Handle more complex parsing
                        result = parser(match)
                        if result:
                            dates["relative_dates"].append(result)
                except Exception as e:
                    logger.warning("Failed to parse date", 
                                 text=match.group(0), error=str(e))
        
        return dates
    
    def _extract_amounts(self, question: str) -> Dict[str, Any]:
        """Extract monetary amounts and thresholds from the question"""
        amounts = {
            "specific_amounts": [],
            "thresholds": [],
            "ranges": []
        }
        
        for pattern, parser in self.amount_patterns:
            matches = re.finditer(pattern, question, re.IGNORECASE)
            for match in matches:
                try:
                    parsed_amount = parser(match)
                    if parsed_amount:
                        if "over" in match.group(0) or "more than" in match.group(0):
                            amounts["thresholds"].append({
                                "type": "minimum",
                                "amount": parsed_amount,
                                "original_text": match.group(0)
                            })
                        elif "under" in match.group(0) or "less than" in match.group(0):
                            amounts["thresholds"].append({
                                "type": "maximum",
                                "amount": parsed_amount,
                                "original_text": match.group(0)
                            })
                        else:
                            amounts["specific_amounts"].append({
                                "amount": parsed_amount,
                                "original_text": match.group(0)
                            })
                except Exception as e:
                    logger.warning("Failed to parse amount", 
                                 text=match.group(0), error=str(e))
        
        return amounts
    
    def _extract_aggregations(self, question: str) -> List[str]:
        """Extract aggregation operations mentioned in the question"""
        aggregations = []
        
        for agg_type, keywords in self.aggregation_keywords.items():
            for keyword in keywords:
                if keyword in question:
                    aggregations.append(agg_type)
                    break
        
        return list(set(aggregations))  # Remove duplicates
    
    def _extract_time_analysis(self, question: str) -> List[str]:
        """Extract time-based analysis requirements"""
        time_analysis = []
        
        for analysis_type, keywords in self.time_keywords.items():
            for keyword in keywords:
                if keyword in question:
                    time_analysis.append(analysis_type)
                    break
        
        return list(set(time_analysis))
    
    def _extract_filters(self, question: str) -> Dict[str, Any]:
        """Extract filtering criteria from the question"""
        filters = {}
        
        # Extract top N requests
        top_pattern = r'\btop\s+(\d+)\b'
        top_match = re.search(top_pattern, question, re.IGNORECASE)
        if top_match:
            filters["limit"] = int(top_match.group(1))
            filters["order"] = "DESC"
        
        # Extract bottom N requests
        bottom_pattern = r'\bbottom\s+(\d+)\b'
        bottom_match = re.search(bottom_pattern, question, re.IGNORECASE)
        if bottom_match:
            filters["limit"] = int(bottom_match.group(1))
            filters["order"] = "ASC"
        
        # Extract specific entity names (basic approach)
        # This could be enhanced with NER (Named Entity Recognition)
        quoted_pattern = r'"([^"]+)"'
        quoted_matches = re.findall(quoted_pattern, question)
        if quoted_matches:
            filters["specific_entities"] = quoted_matches
        
        return filters
    
    def _classify_intent(self, question: str) -> str:
        """Classify the intent of the question"""
        # Define intent patterns
        intent_patterns = {
            "analytics": [r'\banalyze\b', r'\banalysis\b', r'\binsight\b'],
            "comparison": [r'\bcompare\b', r'\bvs\b', r'\bversus\b'],
            "trend": [r'\btrend\b', r'\bover time\b', r'\bmonthly\b', r'\bweekly\b'],
            "ranking": [r'\btop\b', r'\bbest\b', r'\bhighest\b', r'\blowest\b'],
            "summary": [r'\btotal\b', r'\bsum\b', r'\bcount\b', r'\baverage\b'],
            "search": [r'\bfind\b', r'\bshow\b', r'\blist\b', r'\bget\b'],
            "filter": [r'\bwhere\b', r'\bwith\b', r'\bhaving\b'],
        }
        
        for intent, patterns in intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, question, re.IGNORECASE):
                    return intent
        
        return "general"
    
    def _assess_complexity(self, question: str) -> str:
        """Assess the complexity of the question"""
        complexity_score = 0
        
        # Count entities
        entity_count = sum(len(entities) for entities in self._extract_entities(question).values())
        complexity_score += entity_count
        
        # Count aggregations
        aggregation_count = len(self._extract_aggregations(question))
        complexity_score += aggregation_count * 2
        
        # Check for time analysis
        time_analysis_count = len(self._extract_time_analysis(question))
        complexity_score += time_analysis_count * 2
        
        # Check for multiple conditions
        condition_keywords = ['and', 'or', 'but', 'where', 'with', 'having']
        condition_count = sum(1 for keyword in condition_keywords if keyword in question)
        complexity_score += condition_count
        
        # Classify complexity
        if complexity_score <= 2:
            return "low"
        elif complexity_score <= 6:
            return "medium"
        else:
            return "high"
    
    def _build_context(self, parsed_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build context dictionary for SQL generation"""
        context = {}
        
        # Add date context
        if parsed_info["dates"]["absolute_dates"]:
            context["date_filter"] = parsed_info["dates"]["absolute_dates"][0]["date"]
        
        # Add amount context
        if parsed_info["amounts"]["thresholds"]:
            context["amount_threshold"] = parsed_info["amounts"]["thresholds"][0]
        
        # Add entity context
        if parsed_info["entities"]:
            context["primary_entities"] = list(parsed_info["entities"].keys())
        
        # Add aggregation context
        if parsed_info["aggregations"]:
            context["required_aggregations"] = parsed_info["aggregations"]
        
        # Add filter context
        if parsed_info["filters"]:
            context.update(parsed_info["filters"])
        
        return context
    
    # Date parsing helper methods
    def _parse_relative_date(self, match):
        """Parse relative date expressions like 'last 30 days'"""
        try:
            number = int(match.group(1))
            unit = match.group(2)
            
            if unit == 'day':
                return datetime.now() - timedelta(days=number)
            elif unit == 'week':
                return datetime.now() - timedelta(weeks=number)
            elif unit == 'month':
                return datetime.now() - timedelta(days=number * 30)  # Approximate
            elif unit == 'year':
                return datetime.now() - timedelta(days=number * 365)  # Approximate
        except:
            return None
    
    def _parse_current_period(self, match):
        """Parse current period expressions like 'this month'"""
        unit = match.group(1)
        now = datetime.now()
        
        if unit == 'week':
            return self._parse_this_week()
        elif unit == 'month':
            return self._parse_this_month()
        elif unit == 'year':
            return self._parse_this_year()
        
        return now
    
    def _parse_this_week(self):
        """Get the start of this week"""
        now = datetime.now()
        return now - timedelta(days=now.weekday())
    
    def _parse_this_month(self):
        """Get the start of this month"""
        now = datetime.now()
        return now.replace(day=1)
    
    def _parse_this_year(self):
        """Get the start of this year"""
        now = datetime.now()
        return now.replace(month=1, day=1)
    
    def _parse_iso_date(self, match):
        """Parse ISO date format YYYY-MM-DD"""
        try:
            year, month, day = match.groups()
            return datetime(int(year), int(month), int(day))
        except:
            return None
    
    def _parse_us_date(self, match):
        """Parse US date format MM/DD/YYYY"""
        try:
            month, day, year = match.groups()
            return datetime(int(year), int(month), int(day))
        except:
            return None
    
    # Amount parsing helper methods
    def _parse_dollar_amount(self, match):
        """Parse dollar amounts"""
        try:
            amount_str = match.group(1).replace(',', '')
            return float(amount_str)
        except:
            return None
    
    def _parse_amount_threshold(self, match):
        """Parse amount thresholds"""
        try:
            amount_str = match.group(1).replace(',', '')
            return float(amount_str)
        except:
            return None