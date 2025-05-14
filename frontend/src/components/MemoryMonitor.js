import React, { useState, useEffect } from 'react';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend,
  ResponsiveContainer,
  Area,
  AreaChart,
  ComposedChart,
  Bar,
  BarChart
} from 'recharts';
import { AlertCircle, RefreshCw, Trash2 } from 'lucide-react';
import api from '../services/apiService';

const MemoryMonitor = ({ scriptId }) => {
  const [memoryStats, setMemoryStats] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(10000); // 10 seconds

  // Fetch memory stats
  const fetchMemoryStats = async () => {
    if (!scriptId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await api.get(`/api/scripts/${scriptId}/memory`);
      setMemoryStats(response);
    } catch (err) {
      console.error('Error fetching memory stats:', err);
      setError('Failed to load memory usage data');
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchMemoryStats();
  }, [scriptId]);

  // Auto-refresh
  useEffect(() => {
    let intervalId;
    
    if (autoRefresh) {
      intervalId = setInterval(fetchMemoryStats, refreshInterval);
    }
    
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoRefresh, refreshInterval, scriptId]);

  // Handle manual refresh
  const handleRefresh = () => {
    fetchMemoryStats();
  };

  // Handle clearing CUDA cache
  const handleClearCache = async () => {
    try {
      await api.post(`/api/watchdog/action`, {
        action: 'clear_cuda',
        script_id: scriptId
      });
      // Refresh after clearing cache
      setTimeout(fetchMemoryStats, 1000);
    } catch (err) {
      console.error('Error clearing CUDA cache:', err);
      setError('Failed to clear CUDA cache');
    }
  };

  // Format data for charts
  const formatDataForCharts = () => {
    return memoryStats.map(stat => ({
      time: new Date(stat.timestamp).toLocaleTimeString(),
      step: stat.step,
      gpuMemoryPercent: stat.gpu_memory_percent || 0,
      gpuMemoryMB: stat.gpu_memory_used_mb || 0,
      gpuUtilization: stat.gpu_utilization || 0,
      cpuMemoryPercent: stat.cpu_memory_percent || 0
    }));
  };

  // Check for potential memory issues
  const hasMemoryIssue = () => {
    if (memoryStats.length < 3) return false;
    
    const recentStats = memoryStats.slice(-3);
    const avgMemory = recentStats.reduce((sum, stat) => sum + (stat.gpu_memory_percent || 0), 0) / recentStats.length;
    const avgUtilization = recentStats.reduce((sum, stat) => sum + (stat.gpu_utilization || 0), 0) / recentStats.length;
    
    return avgMemory > 90 && avgUtilization < 10;
  };

  const chartData = formatDataForCharts();
  const memoryIssue = hasMemoryIssue();

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Memory Usage</h2>
        <div className="flex space-x-2">
          <button 
            className="btn btn-sm btn-outline"
            onClick={handleRefresh}
            disabled={loading}
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            <span className="ml-1">Refresh</span>
          </button>
          <button 
            className="btn btn-sm btn-outline btn-warning"
            onClick={handleClearCache}
            disabled={loading}
          >
            <Trash2 size={16} />
            <span className="ml-1">Clear CUDA Cache</span>
          </button>
        </div>
      </div>
      
      {error && (
        <div className="alert alert-error mb-4">
          <AlertCircle size={16} className="mr-2" />
          {error}
        </div>
      )}
      
      {memoryIssue && (
        <div className="alert alert-warning mb-4">
          <AlertCircle size={16} className="mr-2" />
          <div>
            <div className="font-semibold">Potential memory issue detected</div>
            <div className="text-sm">High GPU memory usage with low utilization may indicate a memory leak</div>
          </div>
        </div>
      )}
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
        {/* GPU Memory Usage Chart */}
        <div className="card bg-base-100 shadow-sm">
          <div className="card-body p-4">
            <h3 className="text-lg font-medium mb-2">GPU Memory Usage</h3>
            <div className="h-64">
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 12 }}
                      interval="preserveStartEnd"
                    />
                    <YAxis 
                      yAxisId="percent"
                      domain={[0, 100]}
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <YAxis 
                      yAxisId="mb"
                      orientation="right"
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => `${value}MB`}
                    />
                    <Tooltip 
                      formatter={(value, name) => {
                        if (name === 'gpuMemoryPercent') return [`${value.toFixed(1)}%`, 'Memory Usage (%)'];
                        if (name === 'gpuMemoryMB') return [`${value.toFixed(1)}MB`, 'Memory Usage (MB)'];
                        return [value, name];
                      }}
                      labelFormatter={(label) => `Time: ${label}`}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="gpuMemoryPercent" 
                      name="Memory Usage (%)"
                      yAxisId="percent"
                      fill="#8884d8" 
                      stroke="#8884d8"
                      fillOpacity={0.3}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="gpuMemoryMB" 
                      name="Memory Usage (MB)"
                      yAxisId="mb"
                      stroke="#82ca9d" 
                      strokeWidth={2}
                      dot={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-secondary-500">
                  <div className="text-center">
                    <div className="mb-2">No memory data available</div>
                    <div className="text-sm">Run a script to see memory usage</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* GPU Utilization Chart */}
        <div className="card bg-base-100 shadow-sm">
          <div className="card-body p-4">
            <h3 className="text-lg font-medium mb-2">GPU Utilization</h3>
            <div className="h-64">
              {chartData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis 
                      dataKey="time" 
                      tick={{ fontSize: 12 }}
                      interval="preserveStartEnd"
                    />
                    <YAxis 
                      domain={[0, 100]}
                      tick={{ fontSize: 12 }}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip 
                      formatter={(value) => [`${value.toFixed(1)}%`, 'GPU Utilization']}
                      labelFormatter={(label) => `Time: ${label}`}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="gpuUtilization" 
                      name="GPU Utilization"
                      fill="#82ca9d" 
                      stroke="#82ca9d"
                      fillOpacity={0.3}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-full flex items-center justify-center text-secondary-500">
                  <div className="text-center">
                    <div className="mb-2">No utilization data available</div>
                    <div className="text-sm">Run a script to see GPU utilization</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
      
      {/* Memory Usage by Step */}
      <div className="card bg-base-100 shadow-sm">
        <div className="card-body p-4">
          <h3 className="text-lg font-medium mb-2">Memory Usage by Step</h3>
          <div className="h-64">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis 
                    dataKey="step" 
                    tick={{ fontSize: 12 }}
                    interval={0}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    domain={[0, 'dataMax + 10']}
                    tick={{ fontSize: 12 }}
                    tickFormatter={(value) => `${value}MB`}
                  />
                  <Tooltip 
                    formatter={(value) => [`${value.toFixed(1)}MB`, 'GPU Memory']}
                    labelFormatter={(label) => `Step: ${label}`}
                  />
                  <Legend />
                  <Bar 
                    dataKey="gpuMemoryMB" 
                    name="GPU Memory (MB)"
                    fill="#8884d8" 
                  />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-secondary-500">
                <div className="text-center">
                  <div className="mb-2">No step data available</div>
                  <div className="text-sm">Run a script to see memory usage by step</div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Auto-refresh controls */}
      <div className="mt-4 flex items-center">
        <div className="form-control">
          <label className="cursor-pointer label">
            <span className="label-text mr-2">Auto-refresh</span> 
            <input 
              type="checkbox" 
              className="toggle toggle-primary" 
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
          </label>
        </div>
        {autoRefresh && (
          <div className="ml-4">
            <select 
              className="select select-sm select-bordered"
              value={refreshInterval}
              onChange={(e) => setRefreshInterval(Number(e.target.value))}
            >
              <option value={5000}>Every 5 seconds</option>
              <option value={10000}>Every 10 seconds</option>
              <option value={30000}>Every 30 seconds</option>
              <option value={60000}>Every minute</option>
            </select>
          </div>
        )}
      </div>
    </div>
  );
};

export default MemoryMonitor;
