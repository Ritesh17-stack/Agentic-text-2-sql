import React, { useState, useEffect } from 'react';
import { Send, Database, AlertCircle, CheckCircle, Loader2, Table, Code, Brain, Shield, Clock, ChevronDown, ChevronUp } from 'lucide-react';

const API_BASE_URL = 'http://localhost:8000';

export default function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [schema, setSchema] = useState(null);
  const [health, setHealth] = useState(null);
  const [examples, setExamples] = useState([]);
  const [activeTab, setActiveTab] = useState('query');
  const [showReasoning, setShowReasoning] = useState(false);


  useEffect(() => {
    fetchHealth();
    fetchSchema();
    fetchExamples();
  }, []);

  const fetchHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`);
      const data = await response.json();
      setHealth(data);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealth({ database_connected: false, status: 'error' });
    }
  };

  const fetchSchema = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/schema`);
      const data = await response.json();
      setSchema(data);
    } catch (error) {
      console.error('Schema fetch failed:', error);
    }
  };

  const fetchExamples = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/examples`);
      const data = await response.json();
      setExamples(data.examples || []);
    } catch (error) {
      console.error('Examples fetch failed:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query })
      });

      if (!response.ok) {
        throw new Error('Query failed');
      }

      const data = await response.json();
      setResult(data);
      setActiveTab('results');
    } catch (error) {
      setResult({
        error: error.message,
        is_safe: false
      });
    } finally {
      setLoading(false);
    }
  };

  const useExample = (example) => {
    setQuery(example);
    setActiveTab('query');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <header className="bg-black bg-opacity-30 backdrop-blur-lg border-b border-purple-500 border-opacity-30">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-gradient-to-br from-purple-500 to-pink-500 p-2 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">NL to SQL Agent</h1>
                <p className="text-sm text-purple-300">AI-Powered Database Queries</p>
              </div>
            </div>

            <div className="flex items-center gap-4">
              {health && (
                <div className="flex items-center gap-2 bg-black bg-opacity-40 px-4 py-2 rounded-lg">
                  {health.database_connected ? (
                    <>
                      <CheckCircle className="w-4 h-4 text-green-400" />
                      <span className="text-sm text-green-400">Connected</span>
                    </>
                  ) : (
                    <>
                      <AlertCircle className="w-4 h-4 text-red-400" />
                      <span className="text-sm text-red-400">Disconnected</span>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Sidebar - Schema */}
          <div className="lg:col-span-1">
            <div className="bg-black bg-opacity-40 backdrop-blur-lg rounded-2xl border border-purple-500 border-opacity-30 p-6">
              <div className="flex items-center gap-2 mb-4">
                <Database className="w-5 h-5 text-purple-400" />
                <h2 className="text-lg font-semibold text-white">Database Schema</h2>
              </div>

              {schema && schema.tables ? (
                <div className="space-y-4 max-h-96 overflow-y-auto">
                  {schema.tables.map((table, idx) => (
                    <div key={idx} className="bg-purple-900 bg-opacity-30 rounded-lg p-3">
                      <h3 className="font-semibold text-purple-300 mb-2">{table.name}</h3>
                      <div className="space-y-1">
                        {table.columns.map((col, colIdx) => (
                          <div key={colIdx} className="text-xs text-gray-300 flex items-center gap-2">
                            <span className="text-purple-400">•</span>
                            <span className="font-mono">{col.name}</span>
                            <span className="text-gray-500">({col.type})</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-8 text-gray-400">
                  <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                  <p>Loading schema...</p>
                </div>
              )}
            </div>

            {/* Examples */}
            <div className="bg-black bg-opacity-40 backdrop-blur-lg rounded-2xl border border-purple-500 border-opacity-30 p-6 mt-6">
              <h2 className="text-lg font-semibold text-white mb-4">Example Queries</h2>
              <div className="space-y-2">
                {examples.map((example, idx) => (
                  <button
                    key={idx}
                    onClick={() => useExample(example)}
                    className="w-full text-left text-sm text-gray-300 hover:text-white bg-purple-900 bg-opacity-20 hover:bg-opacity-40 rounded-lg p-3 transition-all"
                  >
                    {example}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Main Area */}
          <div className="lg:col-span-2">
            {/* Tabs */}
            <div className="flex gap-2 mb-4">
              <button
                onClick={() => setActiveTab('query')}
                className={`px-6 py-3 rounded-t-xl font-semibold transition-all ${activeTab === 'query'
                  ? 'bg-purple-600 text-white'
                  : 'bg-black bg-opacity-40 text-gray-400 hover:text-white'
                  }`}
              >
                Query
              </button>
              <button
                onClick={() => setActiveTab('results')}
                className={`px-6 py-3 rounded-t-xl font-semibold transition-all ${activeTab === 'results'
                  ? 'bg-purple-600 text-white'
                  : 'bg-black bg-opacity-40 text-gray-400 hover:text-white'
                  }`}
              >
                Results
              </button>
            </div>

            {/* Query Tab */}
            {activeTab === 'query' && (
              <div className="bg-black bg-opacity-40 backdrop-blur-lg rounded-2xl border border-purple-500 border-opacity-30 p-6">
                <form onSubmit={handleSubmit}>
                  <div className="mb-4">
                    <label className="block text-sm font-medium text-purple-300 mb-2">
                      Ask a question about your database
                    </label>
                    <textarea
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="e.g., Show me all users who registered last month"
                      className="w-full h-32 bg-purple-900 bg-opacity-20 border border-purple-500 border-opacity-30 rounded-xl px-4 py-3 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500 focus:ring-2 focus:ring-purple-500 focus:ring-opacity-50 resize-none"
                      disabled={loading}
                    />
                  </div>

                  <button
                    type="submit"
                    disabled={loading || !query.trim()}
                    className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-3 px-6 rounded-xl transition-all flex items-center justify-center gap-2 disabled:cursor-not-allowed"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Send className="w-5 h-5" />
                        Generate SQL Query
                      </>
                    )}
                  </button>
                </form>

                {/* Info Cards */}
                <div className="grid grid-cols-3 gap-4 mt-6">
                  <div className="bg-purple-900 bg-opacity-20 rounded-xl p-4 border border-purple-500 border-opacity-20">
                    <Shield className="w-6 h-6 text-green-400 mb-2" />
                    <p className="text-xs text-gray-400">Safety First</p>
                    <p className="text-sm font-semibold text-white">SQL Injection Protected</p>
                  </div>
                  <div className="bg-purple-900 bg-opacity-20 rounded-xl p-4 border border-purple-500 border-opacity-20">
                    <Brain className="w-6 h-6 text-blue-400 mb-2" />
                    <p className="text-xs text-gray-400">AI Powered</p>
                    <p className="text-sm font-semibold text-white">Smart Reasoning</p>
                  </div>
                  <div className="bg-purple-900 bg-opacity-20 rounded-xl p-4 border border-purple-500 border-opacity-20">
                    <Code className="w-6 h-6 text-purple-400 mb-2" />
                    <p className="text-xs text-gray-400">Read Only</p>
                    <p className="text-sm font-semibold text-white">SELECT Queries Only</p>
                  </div>
                </div>
              </div>
            )}

            {/* Results Tab */}
            {activeTab === 'results' && (
              <div className="space-y-6">
                {result ? (
                  <>
                    {/* Status Banner */}
                    <div className={`rounded-2xl p-4 border ${result.error || !result.is_safe
                      ? 'bg-red-900 bg-opacity-20 border-red-500 border-opacity-30'
                      : 'bg-green-900 bg-opacity-20 border-green-500 border-opacity-30'
                      }`}>
                      <div className="flex items-center gap-3">
                        {result.error || !result.is_safe ? (
                          <>
                            <AlertCircle className="w-6 h-6 text-red-400" />
                            <div>
                              <p className="font-semibold text-red-300">Query Failed</p>
                              <p className="text-sm text-red-400">{result.error || 'Safety validation failed'}</p>
                            </div>
                          </>
                        ) : (
                          <>
                            <CheckCircle className="w-6 h-6 text-green-400" />
                            <div className="flex-1">
                              <p className="font-semibold text-green-300">Query Successful</p>
                              <div className="flex items-center gap-4 text-sm text-green-400">
                                <span>{result.results.count} rows returned</span>
                                <span className="flex items-center gap-1">
                                  <Clock className="w-3 h-3" />
                                  {result.execution_time?.toFixed(2)}s
                                </span>
                              </div>
                            </div>
                          </>
                        )}
                      </div>
                    </div>

                    {/* Results Table */}
                    {result.results && result.results.rows && result.results.rows.length > 0 && (
                      <div className="bg-black bg-opacity-40 backdrop-blur-lg rounded-2xl border border-purple-500 border-opacity-30 p-6">
                        <div className="flex items-center gap-2 mb-4">
                          <Table className="w-5 h-5 text-purple-400" />
                          <h3 className="text-lg font-semibold text-white">Query Results</h3>
                          <span className="ml-auto text-sm text-gray-400">
                            {result.results.count} row(s)
                          </span>
                        </div>

                        {/* Scrollable Table Container */}
                        <div className="overflow-x-auto overflow-y-auto max-h-96 rounded-lg">
                          <table className="w-full">
                            <thead>
                              <tr className="sticky top-0 bg-gradient-to-r from-purple-800 via-purple-700 to-purple-800 text-purple-100">
                                {result.results.columns.map((col, idx) => (
                                  <th
                                    key={idx}
                                    className="text-left py-3 px-4 text-sm font-semibold border-b border-purple-500 border-opacity-40"
                                  >
                                    {col}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {result.results.rows.map((row, rowIdx) => (
                                <tr
                                  key={rowIdx}
                                  className="border-b border-purple-500 border-opacity-10 hover:bg-purple-900 hover:bg-opacity-30 transition-colors"
                                >
                                  {result.results.columns.map((col, colIdx) => (
                                    <td key={colIdx} className="py-3 px-4 text-sm text-gray-300">
                                      {row[col] !== null && row[col] !== undefined ? (
                                        String(row[col])
                                      ) : (
                                        <span className="text-gray-500 italic">null</span>
                                      )}
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    )}

                    {/* Reasoning Steps */}
                    {result.reasoning && result.reasoning.length > 0 && (
                      <div className="bg-black bg-opacity-40 backdrop-blur-lg rounded-2xl border border-purple-500 border-opacity-30 p-6">
                        <div className="flex items-center gap-2 mb-4">
                          <Brain className="w-5 h-5 text-purple-400" />
                          <h3 className="text-lg font-semibold text-white">AI Reasoning Process</h3>
                          <button
                            onClick={() => setShowReasoning((prev) => !prev)}
                            className="ml-auto flex items-center gap-1 px-3 py-1 rounded-md bg-purple-700 bg-opacity-40 hover:bg-purple-600 hover:bg-opacity-60 text-sm text-purple-200 transition-all"
                          >
                            {showReasoning ? (
                              <>
                                <ChevronUp className="w-4 h-4" />
                                Hide
                              </>
                            ) : (
                              <>
                                <ChevronDown className="w-4 h-4" />
                                Show
                              </>
                            )}
                          </button>
                        </div>

                        {showReasoning && (
                          <div className="space-y-3 transition-all duration-300 ease-in-out">
                            {result.reasoning.map((step, idx) => (
                              <div
                                key={idx}
                                className="bg-purple-900 bg-opacity-20 rounded-lg p-4 border-l-4 border-purple-500"
                              >
                                <div className="flex items-start gap-3">
                                  <div className="bg-purple-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                                    {idx + 1}
                                  </div>
                                  <p className="text-sm text-gray-300 flex-1">{step}</p>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}


                    {/* Generated SQL */}
                    {result.sql && (
                      <div className="bg-black bg-opacity-40 backdrop-blur-lg rounded-2xl border border-purple-500 border-opacity-30 p-6">
                        <div className="flex items-center justify-between mb-4">
                          <div className="flex items-center gap-2">
                            <Code className="w-5 h-5 text-purple-400" />
                            <h3 className="text-lg font-semibold text-white">Generated SQL</h3>
                          </div>
                          <button
                            onClick={() => navigator.clipboard.writeText(result.sql)}
                            className="text-xs bg-purple-600 hover:bg-purple-700 text-white px-3 py-1 rounded-lg transition-colors"
                          >
                            Copy
                          </button>
                        </div>
                        <pre className="bg-slate-900 rounded-lg p-4 overflow-x-auto">
                          <code className="text-sm text-green-400 font-mono">{result.sql}</code>
                        </pre>
                      </div>
                    )}


                  </>
                ) : (
                  <div className="bg-black bg-opacity-40 backdrop-blur-lg rounded-2xl border border-purple-500 border-opacity-30 p-12 text-center">
                    <Database className="w-16 h-16 text-purple-400 mx-auto mb-4 opacity-50" />
                    <p className="text-gray-400 text-lg">No results yet</p>
                    <p className="text-gray-500 text-sm mt-2">Submit a query to see results here</p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="mt-12 py-6 text-center text-gray-500 text-sm">
        <p>Built for Safety & Reliabilty</p>
      </footer>
    </div>
  );
}