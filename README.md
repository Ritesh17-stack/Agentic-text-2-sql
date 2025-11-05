# 🚀 Agentic Text-to-SQL

An AI-powered application that converts natural language queries into SQL using an intelligent agentic workflow. Built with FastAPI, React, LangChain, and Groq AI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![React](https://img.shields.io/badge/React-18.3.1-61DAFB.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🤖 **AI-Powered SQL Generation**: Uses Groq's Llama 3.1 8B model for intelligent SQL query generation
- 🔒 **Safety First**: Comprehensive SQL validation to prevent injection attacks and dangerous operations
- 🧠 **Transparent Reasoning**: Step-by-step AI reasoning process for better understanding
- 📊 **Schema Visualization**: Automatic database schema discovery and visualization
- ⚡ **Real-time Processing**: Fast query processing with execution time tracking
- 🎨 **Modern UI**: Beautiful, responsive React frontend with Tailwind CSS
- 🔄 **Agentic Workflow**: Multi-step LangGraph workflow for reliable SQL generation

## 🏗️ Architecture

The application follows a multi-agent workflow pattern with 6 distinct steps:

1. **Query Understanding**: Analyzes user's natural language question
2. **Schema Reasoning**: Identifies relevant tables and columns
3. **SQL Generation**: Creates optimized MySQL queries
4. **Safety Validation**: Validates queries for security
5. **Query Execution**: Safely executes queries against database
6. **Result Formatting**: Formats and returns results

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js 16+ and npm
- MySQL 5.7+ or MariaDB
- Groq API key ([Get one here](https://console.groq.com/))

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd Agentic-text-2-sql
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
# Groq API Configuration
GROQ_API_KEY=your-groq-api-key-here

# MySQL Database Configuration
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your-password
MYSQL_DATABASE=your-database-name
```

### 5. Run the Application

**Terminal 1 - Start Backend:**
```bash
# Activate venv if not already active
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

python backend.py
```

The backend will start at `http://localhost:8000`

**Terminal 2 - Start Frontend:**
```bash
cd frontend
npm run dev
```

The frontend will start at `http://localhost:5173` (or another port if 5173 is busy)

### 6. Access the Application

Open your browser and navigate to:
- Frontend: `http://localhost:5173`
- Backend API Docs: `http://localhost:8000/docs` (Swagger UI)

## 📖 Usage

### Web Interface

1. **View Schema**: The left sidebar displays your database schema automatically
2. **Ask Questions**: Type natural language questions in the query box
3. **View Results**: See query results, generated SQL, and AI reasoning
4. **Try Examples**: Click on example queries to get started quickly

### Example Queries

- "Show me all users who registered last month"
- "What is the total count of orders?"
- "List the top 10 customers by revenue"
- "Show me all products with price greater than 100"
- "Calculate the average order value"

### API Usage

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Get Database Schema
```bash
curl http://localhost:8000/schema
```

#### Process Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me all users"}'
```

## 🔒 Security Features

The application includes multiple security layers:

- **SQL Injection Prevention**: Pattern detection and validation
- **Read-Only Queries**: Only SELECT statements are allowed
- **Dangerous Keyword Blocking**: Blocks DROP, DELETE, UPDATE, etc.
- **Query Sanitization**: Automatic LIMIT clause enforcement
- **Schema-Based Validation**: Validates against actual database schema

## 📁 Project Structure

```
Agentic-text-2-sql/
├── backend.py              # FastAPI backend with agent logic
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── .gitignore             # Git ignore rules
├── README.md              # This file
├── frontend/              # React frontend application
│   ├── src/
│   │   ├── App.jsx        # Main React component
│   │   ├── main.jsx       # React entry point
│   │   └── index.css      # Styles
│   ├── package.json       # Node dependencies
│   └── vite.config.js     # Vite configuration
└── venv/                  # Python virtual environment
```

## 🔧 API Endpoints

| Endpoint          | Method | Description                             |
|-------------------|--------|-----------------------------------------|
| `/`               | GET    | API information and available endpoints |
| `/health`         | GET    | Health check and connection status      |
| `/test-connection`| GET    | Test MySQL database connection          |
| `/schema`         | GET    | Get database schema as JSON             |
| `/query`          | POST   | Process natural language query          |
| `/examples`       | GET    | Get example queries                     |

### Query Request Format

```json
{
  "question": "Show me all users",
  "database": "optional_database_name"
}
```

### Query Response Format

```json
{
  "question": "Show me all users",
  "reasoning": ["Step 1: Understanding...", "Step 2: Schema analysis..."],
  "sql": "SELECT * FROM users LIMIT 100",
  "is_safe": true,
  "results": {
    "columns": ["id", "name", "email"],
    "rows": [...],
    "count": 10
  },
  "timestamp": "2024-01-01T00:00:00",
  "execution_time": 1.23
}
```

## 🛠️ Technology Stack

### Backend
- **FastAPI**: Modern, fast web framework
- **LangChain**: LLM orchestration framework
- **LangGraph**: Agent workflow management
- **Groq AI**: High-performance LLM inference
- **SQLAlchemy**: Database ORM and connection management
- **PyMySQL**: MySQL database driver
- **Pydantic**: Data validation

### Frontend
- **React 18**: UI framework
- **Vite**: Build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **Lucide React**: Icon library

## 🧪 Development

### Running in Development Mode

Backend auto-reloads on code changes when using:
```bash
uvicorn backend:app --reload --host 0.0.0.0 --port 8000
```
                                       
Frontend uses Vite HMR (Hot Module Replacement) automatically.

### Building for Production

**Frontend:**
```bash
cd frontend
npm run build
```

**Backend:**
The backend runs as a standard Python application. For production, consider:
- Using a process manager like PM2 or Supervisor
- Setting up reverse proxy (nginx)
- Using production ASGI server (Gunicorn with Uvicorn workers)

## ⚙️ Configuration

### Backend Configuration

Edit the `Config` class in `backend.py` or use environment variables:

- `GROQ_API_KEY`: Required - Your Groq API key
- `MODEL_NAME`: Default: "llama-3.1-8b-instant"
- `MAX_RESULTS`: Default: 100 (max rows per query)
- `TIMEOUT_SECONDS`: Default: 30

### MySQL Configuration

Set via environment variables in `.env`:
- `MYSQL_HOST`
- `MYSQL_PORT`
- `MYSQL_USER`
- `MYSQL_PASSWORD`
- `MYSQL_DATABASE`

## 🐛 Troubleshooting

### Database Connection Issues

1. Verify MySQL is running: `mysql -u root -p`
2. Check environment variables in `.env`
3. Test connection: `curl http://localhost:8000/test-connection`
4. Verify database exists: `SHOW DATABASES;`

### API Key Issues

1. Verify `GROQ_API_KEY` is set in `.env`
2. Check API key is valid at [Groq Console](https://console.groq.com/)
3. Restart backend after changing `.env`

### Frontend Connection Issues

1. Verify backend is running on port 8000
2. Check CORS settings if accessing from different domain
3. Update `API_BASE_URL` in `frontend/src/App.jsx` if needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [Groq](https://groq.com/) for high-performance AI inference
- [LangChain](https://www.langchain.com/) for LLM orchestration
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [React](https://react.dev/) and [Vite](https://vitejs.dev/) for the frontend

## 📞 Support

For issues and questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Review API documentation at `/docs` endpoint
3. Open an issue on GitHub

---

**Built with ❤️ for safe and intelligent database querying**

