"""
Natural Language to SQL Agentic AI with MySQL Support
Backend API using FastAPI
"""

import os
import re
import json
from typing import TypedDict, Annotated, Sequence, Optional, List, Dict, Any
from datetime import datetime
import sqlparse
from sqlalchemy import create_engine, text, MetaData, inspect
from sqlalchemy.exc import SQLAlchemyError
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from dotenv import load_dotenv

# ==================== Configuration ====================
load_dotenv()

class Config:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your-groq-api-key-here")
    MODEL_NAME = "llama-3.1-8b-instant"
    
    # MySQL Configuration
    MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT = os.getenv("MYSQL_PORT", "3306")
    MYSQL_USER = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "testdb")
    
    @property
    def DATABASE_URL(self):
        from urllib.parse import quote_plus
        encoded_password = quote_plus(self.MYSQL_PASSWORD) if self.MYSQL_PASSWORD else ""
        
        if encoded_password:
            return f"mysql+pymysql://{self.MYSQL_USER}:{encoded_password}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
        else:
            return f"mysql+pymysql://{self.MYSQL_USER}@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
    
    MAX_RESULTS = 100
    TIMEOUT_SECONDS = 30

# ==================== Pydantic Models ====================
class QueryRequest(BaseModel):
    question: str
    database: Optional[str] = None

class QueryResponse(BaseModel):
    question: str
    reasoning: List[str]
    sql: str
    is_safe: bool
    results: Dict[str, Any]
    timestamp: str
    execution_time: float

class SchemaResponse(BaseModel):
    tables: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]

# ==================== Safety Validator ====================
class SQLSafetyValidator:
    """Validates SQL queries for safety before execution"""
    
    DANGEROUS_KEYWORDS = [
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 
        'INSERT', 'UPDATE', 'GRANT', 'REVOKE', 'EXEC',
        'EXECUTE', 'UNION', '--', '/*', '*/', 'SHUTDOWN',
        'LOAD_FILE', 'INTO OUTFILE', 'INTO DUMPFILE'
    ]
    
    @staticmethod
    def is_safe_query(query: str) -> tuple[bool, str]:
        """
        Validates if a SQL query is safe to execute
        Returns: (is_safe, reason)
        """
        query_upper = query.upper()
        
        # Check for dangerous keywords
        for keyword in SQLSafetyValidator.DANGEROUS_KEYWORDS:
            if keyword in query_upper:
                return False, f"Query contains dangerous keyword: {keyword}"
        
        # Must be a SELECT query
        if not query_upper.strip().startswith('SELECT'):
            return False, "Only SELECT queries are allowed"
        
        # Check for SQL injection patterns
        injection_patterns = [
            r";\s*DROP",
            r";\s*DELETE",
            r"'\s*OR\s*'1'\s*=\s*'1",
            r"--",
            r"/\*.*\*/"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query_upper):
                return False, f"Potential SQL injection detected"
        
        return True, "Query is safe"
    
    @staticmethod
    def sanitize_limit(query: str, max_results: int = 100) -> str:
        """Adds or enforces LIMIT clause"""
        query_upper = query.upper()
        
        if 'LIMIT' not in query_upper:
            return f"{query.rstrip(';')} LIMIT {max_results}"
        
        return query

# ==================== Schema Relevance Validator ====================
class SchemaRelevanceValidator:
    """Validates if user query is relevant to database schema"""
    
    @staticmethod
    def extract_table_names(schema_info: str) -> List[str]:
        """Extract all table names from schema info"""
        tables = []
        for line in schema_info.split('\n'):
            if line.startswith('Table: '):
                table_name = line.replace('Table: ', '').strip()
                tables.append(table_name)
        return tables
    
    @staticmethod
    def extract_column_names(schema_info: str) -> List[str]:
        """Extract all column names from schema info"""
        columns = []
        for line in schema_info.split('\n'):
            if line.strip().startswith('- '):
                
                col_info = line.strip()[2:].split(':')[0].strip()
                columns.append(col_info)
        return columns
    
    @staticmethod
    def check_schema_relevance(
        user_question: str, 
        schema_reasoning: str, 
        schema_info: str,
        available_tables: List[str]
    ) -> tuple[bool, str]:
        """
        Check if the query is actually relevant to the database schema
        Returns: (is_relevant, reason)
        """
        question_lower = user_question.lower()
        reasoning_lower = schema_reasoning.lower()
        
        
        non_db_indicators = [
            'characteristics of',
            'what is a',
            'define',
            'explain what',
            'tell me about',
            'describe a',
            'how does a',
            'why do',
            'list facts about',
            'properties of',
            'features of',
            'write a',
            'create a story',
            'poem about',
            'joke about'
        ]
        
        for indicator in non_db_indicators:
            if indicator in question_lower:
                return False, f"Query appears to be a general knowledge question, not a database query. This system only answers questions about the database."
        
        # Check if any actual table names are mentioned in the reasoning
        tables_mentioned = []
        for table in available_tables:
            if table.lower() in reasoning_lower or table.lower() in question_lower:
                tables_mentioned.append(table)
        
        # Check if reasoning mentions "no table" or similar
        no_table_indicators = [
            'no table',
            'no relevant table',
            'cannot find',
            'does not exist',
            'not available in the schema',
            'schema does not contain',
            'no appropriate table',
            'database does not have',
            'unable to find',
            'no matching table',
            'no suitable table'
        ]
        
        for indicator in no_table_indicators:
            if indicator in reasoning_lower:
                return False, f"No relevant tables found in the database schema for this query. Available tables: {', '.join(available_tables)}"
        
        # If no tables mentioned in reasoning, it's likely not a valid DB query
        if not tables_mentioned and len(available_tables) > 0:
            return False, f"Query does not reference any tables in the database schema. Available tables: {', '.join(available_tables)}"
        
        # Additional check: if schema_info is empty or has no tables
        if not available_tables or len(available_tables) == 0:
            return False, "Database schema is empty or not available"
        
        return True, f"Query is relevant to database schema. Tables identified: {', '.join(tables_mentioned)}"

# ==================== Database Manager ====================
class DatabaseManager:
    """Manages database connections and schema inspection"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=False
        )
        self.metadata = MetaData()
        
    def test_connection(self) -> tuple[bool, str]:
        """Test database connection"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True, "Connection successful"
        except Exception as e:
            return False, str(e)
    
    def get_schema_info(self) -> str:
        """Returns detailed schema information"""
        try:
            self.metadata.reflect(bind=self.engine)
            inspector = inspect(self.engine)
            schema_info = []
            
            for table_name in inspector.get_table_names():
                schema_info.append(f"\nTable: {table_name}")
                columns = inspector.get_columns(table_name)
                
                for col in columns:
                    col_type = str(col['type'])
                    nullable = "NULL" if col['nullable'] else "NOT NULL"
                    default = f"DEFAULT {col.get('default', 'None')}" if col.get('default') else ""
                    schema_info.append(f"  - {col['name']}: {col_type} ({nullable}) {default}")
                
                # Get foreign keys
                fks = inspector.get_foreign_keys(table_name)
                if fks:
                    schema_info.append("  Foreign Keys:")
                    for fk in fks:
                        schema_info.append(f"    - {fk['constrained_columns']} -> {fk['referred_table']}.{fk['referred_columns']}")
                
                # Get indexes
                indexes = inspector.get_indexes(table_name)
                if indexes:
                    schema_info.append("  Indexes:")
                    for idx in indexes:
                        schema_info.append(f"    - {idx['name']}: {idx['column_names']}")
            
            return "\n".join(schema_info)
        except Exception as e:
            return f"Error fetching schema: {str(e)}"
    
    def get_schema_json(self) -> Dict[str, Any]:
        """Returns schema as JSON for frontend"""
        try:
            self.metadata.reflect(bind=self.engine)
            inspector = inspect(self.engine)
            
            tables = []
            relationships = []
            
            for table_name in inspector.get_table_names():
                columns = []
                for col in inspector.get_columns(table_name):
                    columns.append({
                        "name": col['name'],
                        "type": str(col['type']),
                        "nullable": col['nullable'],
                        "primary_key": col.get('primary_key', False)
                    })
                
                tables.append({
                    "name": table_name,
                    "columns": columns
                })
                
                # Get relationships
                fks = inspector.get_foreign_keys(table_name)
                for fk in fks:
                    relationships.append({
                        "from_table": table_name,
                        "from_column": fk['constrained_columns'][0] if fk['constrained_columns'] else None,
                        "to_table": fk['referred_table'],
                        "to_column": fk['referred_columns'][0] if fk['referred_columns'] else None
                    })
            
            return {"tables": tables, "relationships": relationships}
        except Exception as e:
            return {"tables": [], "relationships": [], "error": str(e)}
    
    def execute_query(self, query: str) -> tuple[list, list]:
        """
        Executes a query safely and returns results
        Returns: (column_names, rows)
        """
        with self.engine.connect() as conn:
            result = conn.execute(text(query))
            columns = list(result.keys())
            rows = [dict(row._mapping) for row in result.fetchall()]
            return columns, rows

# ==================== Agent State ====================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    user_question: str
    schema_info: str
    available_tables: list[str]  # NEW: Track available tables
    reasoning_steps: list[str]
    generated_sql: str
    is_sql_safe: bool
    is_schema_relevant: bool  # NEW: Track schema relevance
    safety_message: str
    query_results: dict
    next_action: str

# ==================== SQL Agent ====================
class NLToSQLAgent:
    """Agentic AI for Natural Language to SQL conversion with reasoning"""
    
    def __init__(self, config: Config, db_manager: DatabaseManager):
        self.config = config
        self.db_manager = db_manager
        self.llm = ChatGroq(
            api_key=config.GROQ_API_KEY,
            model=config.MODEL_NAME,
            temperature=0.1
        )
        self.validator = SQLSafetyValidator()
        self.relevance_validator = SchemaRelevanceValidator()  # NEW
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Creates the agent workflow using LangGraph"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("understand_query", self._understand_query)
        workflow.add_node("reason_about_schema", self._reason_about_schema)
        workflow.add_node("validate_relevance", self._validate_relevance)  # NEW NODE
        workflow.add_node("generate_sql", self._generate_sql)
        workflow.add_node("validate_sql", self._validate_sql)
        workflow.add_node("execute_query", self._execute_query)
        workflow.add_node("format_results", self._format_results)
        workflow.add_node("handle_error", self._handle_error)
        
        workflow.set_entry_point("understand_query")
        
        workflow.add_edge("understand_query", "reason_about_schema")
        workflow.add_edge("reason_about_schema", "validate_relevance")  # NEW EDGE
        
        
        workflow.add_conditional_edges(
            "validate_relevance",
            lambda state: state["next_action"],
            {
                "generate": "generate_sql",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("generate_sql", "validate_sql")
        
        workflow.add_conditional_edges(
            "validate_sql",
            lambda state: state["next_action"],
            {
                "execute": "execute_query",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("execute_query", "format_results")
        workflow.add_edge("format_results", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    def _understand_query(self, state: AgentState) -> AgentState:
        """Step 1: Understand the user's natural language query"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at understanding natural language database queries.
            Analyze the user's question and determine:
            1. Is this a database-related question or a general knowledge question?
            2. What data they want to retrieve (if database-related)
            3. Any filters or conditions
            4. Sorting or grouping requirements
            5. Aggregations needed
            
            If the question is NOT about querying a database (e.g., "What are characteristics of a dog?", 
            "Tell me a joke", "Define artificial intelligence"), clearly state that this is a general 
            knowledge question, not a database query.
            
            Provide your analysis in a clear, structured format."""),
            ("user", "{question}")
        ])
        
        response = self.llm.invoke(
            prompt.format_messages(question=state["user_question"])
        )
        
        state["reasoning_steps"].append(f"Query Understanding: {response.content}")
        state["messages"].append(AIMessage(content=f"Understanding: {response.content}"))
        
        return state
    
    def _reason_about_schema(self, state: AgentState) -> AgentState:
        """Step 2: Reason about which tables and columns to use"""
        # Extract available tables from schema
        available_tables = self.relevance_validator.extract_table_names(state["schema_info"])
        state["available_tables"] = available_tables
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a MySQL database expert. Given the database schema and user question,
            identify which tables and columns are needed to answer the question.
            
            Database Schema:
            {schema}
            
            IMPORTANT: If the user's question is NOT related to the database schema or asks about 
            general knowledge (like "characteristics of dogs", "what is X", etc.), clearly state 
            that NO TABLES are relevant and this cannot be answered from the database.
            
            If relevant tables exist, provide reasoning about:
            1. Which specific tables to query (mention exact table names)
            2. Which columns to select (mention exact column names)
            3. How tables should be joined (if multiple tables)
            4. What WHERE conditions are needed
            
            If NO relevant tables exist, clearly state: "No relevant tables found in the database schema for this query."
            """),
            ("user", "User Question: {question}\n\nPrevious Understanding: {understanding}")
        ])
        
        response = self.llm.invoke(
            prompt.format_messages(
                schema=state["schema_info"],
                question=state["user_question"],
                understanding=state["reasoning_steps"][0]
            )
        )
        
        state["reasoning_steps"].append(f"Schema Reasoning: {response.content}")
        state["messages"].append(AIMessage(content=f"Schema Analysis: {response.content}"))
        
        return state
    
    def _validate_relevance(self, state: AgentState) -> AgentState:
        """Step 3: Validate if query is relevant to database schema - NEW STEP"""
        schema_reasoning = state["reasoning_steps"][-1]  # Get the schema reasoning
        
        is_relevant, message = self.relevance_validator.check_schema_relevance(
            user_question=state["user_question"],
            schema_reasoning=schema_reasoning,
            schema_info=state["schema_info"],
            available_tables=state["available_tables"]
        )
        
        if is_relevant:
            state["is_schema_relevant"] = True
            state["next_action"] = "generate"
            state["reasoning_steps"].append(f"✓ Schema Relevance Check: {message}")
        else:
            state["is_schema_relevant"] = False
            state["is_sql_safe"] = False
            state["safety_message"] = f"Security Validation Failed: {message}"
            state["next_action"] = "error"
            state["reasoning_steps"].append(f"✗ Schema Relevance Check Failed: {message}")
            state["generated_sql"] = ""  # No SQL will be generated
        
        return state
    
    def _generate_sql(self, state: AgentState) -> AgentState:
        """Step 4: Generate the SQL query"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert MySQL query generator. 
            Generate a SAFE, READ-ONLY SQL query based on the reasoning provided.
            
            Database Schema:
            {schema}
            
            IMPORTANT RULES:
            1. Generate ONLY SELECT queries for MySQL
            2. Use proper JOIN syntax when multiple tables are needed
            3. Use backticks for table/column names with special characters
            4. Include appropriate WHERE clauses
            5. Add ORDER BY if sorting is requested
            6. Use LIMIT to prevent excessive results
            7. Return ONLY the SQL query, no explanations
            8. Use MySQL-specific syntax and functions
            9. Give only query at a time without comments
            10. Use CONCAT() command for concatenation
            11. Only use tables and columns that exist in the schema
            
            Previous Reasoning:
            {reasoning}"""),
            ("user", "Generate MySQL query without comments and use exact table and column names: {question}")
        ])
        
        response = self.llm.invoke(
            prompt.format_messages(
                schema=state["schema_info"],
                reasoning="\n".join(state["reasoning_steps"]),
                question=state["user_question"]
            )
        )
        
        sql_query = response.content.strip()
        sql_query = re.sub(r'^```sql\s*', '', sql_query)
        sql_query = re.sub(r'^```\s*', '', sql_query)
        sql_query = re.sub(r'\s*```$', '', sql_query)
        sql_query = sql_query.strip()
        sql_query = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
        
        state["generated_sql"] = sql_query
        state["reasoning_steps"].append(f"Generated SQL: {sql_query}")
        state["messages"].append(AIMessage(content=f"Generated Query:\n```sql\n{sql_query}\n```"))
        
        return state
    
    def _validate_sql(self, state: AgentState) -> AgentState:
        """Step 5: Validate SQL for safety"""
        sql_query = state["generated_sql"]
        
        is_safe, message = self.validator.is_safe_query(sql_query)
        
        if is_safe:
            sql_query = self.validator.sanitize_limit(sql_query, self.config.MAX_RESULTS)
            state["generated_sql"] = sql_query
            state["is_sql_safe"] = True
            state["safety_message"] = "Query validated successfully"
            state["next_action"] = "execute"
            state["reasoning_steps"].append("✓ SQL Safety Validation: Passed")
        else:
            state["is_sql_safe"] = False
            state["safety_message"] = f"Security Validation Failed: {message}"
            state["next_action"] = "error"
            state["reasoning_steps"].append(f"✗ SQL Safety Validation Failed: {message}")
        
        return state
    
    def _execute_query(self, state: AgentState) -> AgentState:
        """Step 6: Execute the SQL query"""
        try:
            columns, rows = self.db_manager.execute_query(state["generated_sql"])
            
            # Convert datetime objects to strings for JSON serialization
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, datetime):
                        row[key] = value.isoformat()
            
            state["query_results"] = {
                "columns": columns,
                "rows": rows,
                "count": len(rows)
            }
            state["reasoning_steps"].append(f"✓ Query executed successfully. Retrieved {len(rows)} rows.")
        except SQLAlchemyError as e:
            state["query_results"] = {
                "error": str(e),
                "columns": [],
                "rows": []
            }
            state["reasoning_steps"].append(f"✗ Query execution error: {str(e)}")
        
        return state
    
    def _format_results(self, state: AgentState) -> AgentState:
        """Step 7: Format results for user"""
        results = state["query_results"]
        
        if "error" in results:
            message = f"Error executing query: {results['error']}"
        else:
            message = f"Query executed successfully! Retrieved {results['count']} rows."
        
        state["messages"].append(AIMessage(content=message))
        return state
    
    def _handle_error(self, state: AgentState) -> AgentState:
        """Handle errors in the workflow"""
        message = f"❌ {state['safety_message']}"
        state["messages"].append(AIMessage(content=message))
        state["query_results"] = {
            "error": state["safety_message"],
            "columns": [],
            "rows": [],
            "count": 0
        }
        return state
    
    def process_query(self, user_question: str) -> dict:
        """Main method to process a natural language query"""
        start_time = datetime.now()
        
        schema_info = self.db_manager.get_schema_info()
        available_tables = self.relevance_validator.extract_table_names(schema_info)
        
        initial_state = AgentState(
            messages=[HumanMessage(content=user_question)],
            user_question=user_question,
            schema_info=schema_info,
            available_tables=available_tables,
            reasoning_steps=[],
            generated_sql="",
            is_sql_safe=False,
            is_schema_relevant=False,  
            safety_message="",
            query_results={},
            next_action=""
        )
        
        final_state = self.workflow.invoke(initial_state)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "question": user_question,
            "reasoning": final_state["reasoning_steps"],
            "sql": final_state["generated_sql"],
            "is_safe": final_state["is_sql_safe"],
            "results": final_state["query_results"],
            "timestamp": datetime.now().isoformat(),
            "execution_time": execution_time
        }

# ==================== FastAPI Application ====================
app = FastAPI(
    title="NL to SQL Agent API",
    description="Natural Language to SQL conversion with AI reasoning and security validation",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config = Config()
db_manager = None
agent = None

@app.on_event("startup")
async def startup_event():
    """Initialize database connection and agent on startup"""
    global db_manager, agent
    
    try:
        db_manager = DatabaseManager(config.DATABASE_URL)
        is_connected, message = db_manager.test_connection()
        
        if not is_connected:
            print(f"⚠️  Warning: Database connection failed: {message}")
            print(f"Please check your MySQL configuration in environment variables")
        else:
            print(f"✅ Connected to MySQL database: {config.MYSQL_DATABASE}")
            agent = NLToSQLAgent(config, db_manager)
            print("🤖 NL to SQL Agent initialized successfully!")
            print("🛡️  Security features enabled: SQL Injection Prevention + Schema Relevance Validation")
    except Exception as e:
        print(f"❌ Startup error: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NL to SQL Agent API with Enhanced Security",
        "version": "2.0.0",
        "security_features": [
            "SQL Injection Prevention",
            "Dangerous Keyword Blocking",
            "Schema Relevance Validation",
            "General Question Detection"
        ],
        "endpoints": {
            "health": "/health",
            "schema": "/schema",
            "query": "/query (POST)",
            "test_connection": "/test-connection",
            "examples": "/examples"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "database_connected": db_manager is not None,
        "agent_initialized": agent is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/test-connection")
async def test_connection():
    """Test database connection"""
    if db_manager is None:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    is_connected, message = db_manager.test_connection()
    
    return {
        "connected": is_connected,
        "message": message,
        "database": config.MYSQL_DATABASE,
        "host": config.MYSQL_HOST
    }

@app.get("/schema", response_model=SchemaResponse)
async def get_schema():
    """Get database schema information"""
    if db_manager is None:
        raise HTTPException(status_code=500, detail="Database manager not initialized")
    
    try:
        schema_data = db_manager.get_schema_json()
        return schema_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process natural language query and return SQL + results"""
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized. Check database connection.")
    
    try:
        result = agent.process_query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/examples")
async def get_examples():
    """Get example queries"""
    return {
        "examples": [
            "Show me all records from the users table",
            "What is the total count of orders?",
            "List the top 10 customers by revenue",
            "Show me all products with price greater than 100",
            "What are the most recent 5 transactions?",
            "Calculate the average order value",
            "Show me all orders from the last 30 days"
        ]
    }

if __name__ == "__main__":
    print("🚀 Starting NL to SQL Agent API with Enhanced Security...")
    print(f"📊 Database: {config.MYSQL_DATABASE}@{config.MYSQL_HOST}")
    print(f"🔑 Groq API Key configured: {config.GROQ_API_KEY != 'your-groq-api-key-here'}")
    print("🛡️  Security Features:")
    print("   ✓ SQL Injection Prevention")
    print("   ✓ Dangerous Keyword Blocking")
    print("   ✓ Schema Relevance Validation")
    print("   ✓ General Question Detection")
    print("\n" + "="*60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")