import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { 
  BarChart2, 
  CheckCircle, 
  AlertCircle, 
  Clock, 
  ChevronDown, 
  ChevronUp,
  Calendar,
  Clock3,
  Tag,
  Settings
} from 'lucide-react';
import { scriptService } from '../services/scriptService';
import { useScript } from '../contexts/ScriptContext';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  BarChart,
  Bar,
  Legend
} from 'recharts';

const Results = () => {
  const { scripts } = useScript();
  const [runHistory, setRunHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedRun, setExpandedRun] = useState(null);
  const [sortField, setSortField] = useState('timestamp');
  const [sortDirection, setSortDirection] = useState('desc');
  const [filterScript, setFilterScript] = useState('all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [metricsData, setMetricsData] = useState(null);

  // Fetch run history on mount
  useEffect(() => {
    const fetchRunHistory = async () => {
      setLoading(true);
      try {
        const history = await scriptService.getRunHistory();
        setRunHistory(history);
        setError(null);
        
        // Generate metrics data
        generateMetricsData(history);
      } catch (err) {
        console.error('Error fetching run history:', err);
        setError('Failed to fetch run history');
      } finally {
        setLoading(false);
      }
    };

    fetchRunHistory();
  }, []);

  // Generate metrics data for charts
  const generateMetricsData = (history) => {
    if (!history || history.length === 0) return;
    
    // Script distribution data
    const scriptCounts = {};
    scripts.forEach(script => {
      scriptCounts[script.id] = 0;
    });
    
    history.forEach(run => {
      if (scriptCounts[run.scriptId] !== undefined) {
        scriptCounts[run.scriptId]++;
      }
    });
    
    const scriptDistribution = Object.keys(scriptCounts).map(scriptId => ({
      name: scripts.find(s => s.id === scriptId)?.name || scriptId,
      count: scriptCounts[scriptId]
    }));
    
    // Status distribution data
    const statusCounts = {
      completed: 0,
      error: 0,
      pending: 0
    };
    
    history.forEach(run => {
      if (statusCounts[run.status] !== undefined) {
        statusCounts[run.status]++;
      }
    });
    
    const statusDistribution = Object.keys(statusCounts).map(status => ({
      name: status.charAt(0).toUpperCase() + status.slice(1),
      count: statusCounts[status]
    }));
    
    // Duration data (for completed runs)
    const durationData = history
      .filter(run => run.status === 'completed' && run.duration)
      .map(run => ({
        name: run.scriptId,
        duration: run.duration,
        timestamp: new Date(run.timestamp).toLocaleString()
      }));
    
    setMetricsData({
      scriptDistribution,
      statusDistribution,
      durationData
    });
  };

  // Sort and filter runs
  const getSortedAndFilteredRuns = () => {
    if (!runHistory) return [];
    
    return runHistory
      .filter(run => {
        if (filterScript !== 'all' && run.scriptId !== filterScript) return false;
        if (filterStatus !== 'all' && run.status !== filterStatus) return false;
        return true;
      })
      .sort((a, b) => {
        let comparison = 0;
        
        switch (sortField) {
          case 'timestamp':
            comparison = new Date(a.timestamp) - new Date(b.timestamp);
            break;
          case 'scriptId':
            comparison = a.scriptId.localeCompare(b.scriptId);
            break;
          case 'status':
            comparison = a.status.localeCompare(b.status);
            break;
          case 'duration':
            comparison = (a.duration || 0) - (b.duration || 0);
            break;
          default:
            comparison = 0;
        }
        
        return sortDirection === 'asc' ? comparison : -comparison;
      });
  };

  // Toggle sort direction or change sort field
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Get status badge for a run
  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return <span className="badge badge-success flex items-center"><CheckCircle size={14} className="mr-1" /> Completed</span>;
      case 'error':
        return <span className="badge badge-error flex items-center"><AlertCircle size={14} className="mr-1" /> Error</span>;
      default:
        return <span className="badge flex items-center"><Clock size={14} className="mr-1" /> {status}</span>;
    }
  };

  // Get sort indicator
  const getSortIndicator = (field) => {
    if (sortField !== field) return null;
    
    return sortDirection === 'asc' 
      ? <ChevronUp size={16} className="ml-1" />
      : <ChevronDown size={16} className="ml-1" />;
  };

  // Format duration
  const formatDuration = (seconds) => {
    if (!seconds) return '-';
    
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    
    if (minutes === 0) {
      return `${remainingSeconds}s`;
    }
    
    return `${minutes}m ${remainingSeconds}s`;
  };

  return (
    <div className="space-y-6">
      {/* Metrics Overview */}
      {metricsData && (
        <div className="card">
          <h2 className="text-xl font-semibold mb-4">Metrics Overview</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Script Distribution */}
            <div className="bg-white rounded-lg p-4 border border-secondary-200">
              <h3 className="text-lg font-medium mb-2">Script Distribution</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metricsData.scriptDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#0ea5e9" name="Runs" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Status Distribution */}
            <div className="bg-white rounded-lg p-4 border border-secondary-200">
              <h3 className="text-lg font-medium mb-2">Status Distribution</h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={metricsData.statusDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#0ea5e9" name="Runs" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
            
            {/* Duration Chart */}
            {metricsData.durationData.length > 0 && (
              <div className="bg-white rounded-lg p-4 border border-secondary-200 lg:col-span-2">
                <h3 className="text-lg font-medium mb-2">Run Durations</h3>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={metricsData.durationData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="timestamp" />
                      <YAxis label={{ value: 'Duration (seconds)', angle: -90, position: 'insideLeft' }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="duration" stroke="#0ea5e9" name="Duration (s)" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Run History */}
      <div className="card">
        <h2 className="text-xl font-semibold mb-4">Run History</h2>
        
        {/* Filters */}
        <div className="flex flex-wrap gap-4 mb-4">
          <div>
            <label htmlFor="filterScript" className="block text-sm font-medium text-secondary-700 mb-1">
              Script
            </label>
            <select
              id="filterScript"
              value={filterScript}
              onChange={(e) => setFilterScript(e.target.value)}
              className="form-input py-1 px-2"
            >
              <option value="all">All Scripts</option>
              {scripts.map((script) => (
                <option key={script.id} value={script.id}>
                  {script.name}
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <label htmlFor="filterStatus" className="block text-sm font-medium text-secondary-700 mb-1">
              Status
            </label>
            <select
              id="filterStatus"
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="form-input py-1 px-2"
            >
              <option value="all">All Statuses</option>
              <option value="completed">Completed</option>
              <option value="error">Error</option>
              <option value="pending">Pending</option>
            </select>
          </div>
        </div>
        
        {error && (
          <div className="bg-error-50 border border-error-200 text-error-700 px-4 py-3 rounded mb-4">
            <p>{error}</p>
          </div>
        )}
        
        {loading ? (
          <div className="flex items-center justify-center h-32">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
          </div>
        ) : getSortedAndFilteredRuns().length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-secondary-50 border-b border-secondary-200">
                  <th className="text-left py-2 px-4">
                    <button
                      className="flex items-center font-semibold"
                      onClick={() => handleSort('timestamp')}
                    >
                      Timestamp {getSortIndicator('timestamp')}
                    </button>
                  </th>
                  <th className="text-left py-2 px-4">
                    <button
                      className="flex items-center font-semibold"
                      onClick={() => handleSort('scriptId')}
                    >
                      Script {getSortIndicator('scriptId')}
                    </button>
                  </th>
                  <th className="text-left py-2 px-4">
                    <button
                      className="flex items-center font-semibold"
                      onClick={() => handleSort('status')}
                    >
                      Status {getSortIndicator('status')}
                    </button>
                  </th>
                  <th className="text-left py-2 px-4">
                    <button
                      className="flex items-center font-semibold"
                      onClick={() => handleSort('duration')}
                    >
                      Duration {getSortIndicator('duration')}
                    </button>
                  </th>
                  <th className="text-left py-2 px-4"></th>
                </tr>
              </thead>
              <tbody>
                {getSortedAndFilteredRuns().map((run, index) => (
                  <React.Fragment key={index}>
                    <tr className="border-b border-secondary-100 hover:bg-secondary-50">
                      <td className="py-3 px-4">
                        {new Date(run.timestamp).toLocaleString()}
                      </td>
                      <td className="py-3 px-4 font-medium">
                        {scripts.find(s => s.id === run.scriptId)?.name || run.scriptId}
                      </td>
                      <td className="py-3 px-4">
                        {getStatusBadge(run.status)}
                      </td>
                      <td className="py-3 px-4">
                        {formatDuration(run.duration)}
                      </td>
                      <td className="py-3 px-4 text-right">
                        <button
                          className="text-primary-600 hover:text-primary-800"
                          onClick={() => setExpandedRun(expandedRun === index ? null : index)}
                        >
                          {expandedRun === index ? 'Hide Details' : 'Show Details'}
                        </button>
                      </td>
                    </tr>
                    {expandedRun === index && (
                      <tr className="bg-secondary-50">
                        <td colSpan="5" className="py-4 px-6">
                          <div className="space-y-4">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                              <div className="flex items-start">
                                <Calendar size={18} className="text-secondary-500 mr-2 mt-0.5" />
                                <div>
                                  <p className="text-sm font-medium text-secondary-700">Timestamp</p>
                                  <p className="text-sm text-secondary-900">
                                    {new Date(run.timestamp).toLocaleString()}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-start">
                                <Clock3 size={18} className="text-secondary-500 mr-2 mt-0.5" />
                                <div>
                                  <p className="text-sm font-medium text-secondary-700">Duration</p>
                                  <p className="text-sm text-secondary-900">
                                    {formatDuration(run.duration)}
                                  </p>
                                </div>
                              </div>
                              <div className="flex items-start">
                                <Tag size={18} className="text-secondary-500 mr-2 mt-0.5" />
                                <div>
                                  <p className="text-sm font-medium text-secondary-700">Status</p>
                                  <p className="text-sm text-secondary-900">
                                    {run.status.charAt(0).toUpperCase() + run.status.slice(1)}
                                  </p>
                                </div>
                              </div>
                            </div>
                            
                            {run.config && (
                              <div>
                                <div className="flex items-center mb-2">
                                  <Settings size={18} className="text-secondary-500 mr-2" />
                                  <p className="text-sm font-medium text-secondary-700">Configuration</p>
                                </div>
                                <div className="bg-white p-3 rounded border border-secondary-200 overflow-x-auto">
                                  <pre className="text-xs font-mono">
                                    {JSON.stringify(run.config, null, 2)}
                                  </pre>
                                </div>
                              </div>
                            )}
                            
                            {run.error && (
                              <div className="bg-error-50 border border-error-200 text-error-700 px-4 py-3 rounded">
                                <p className="font-medium">Error</p>
                                <p>{run.error}</p>
                              </div>
                            )}
                            
                            <div className="flex justify-end">
                              <Link
                                to={`/scripts/${run.scriptId}`}
                                className="text-primary-600 hover:text-primary-800 text-sm font-medium"
                              >
                                Configure Script
                              </Link>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center py-8 text-secondary-500">
            No run history found
          </div>
        )}
      </div>
    </div>
  );
};

export default Results;
