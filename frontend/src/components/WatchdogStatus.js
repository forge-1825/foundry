import React, { useState, useEffect } from 'react';
import { AlertCircle, RefreshCw, Trash2, Play, AlertTriangle, CheckCircle } from 'lucide-react';
import api from '../services/apiService';

const WatchdogStatus = () => {
  const [watchdogStatus, setWatchdogStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(10000); // 10 seconds

  // Fetch watchdog status
  const fetchWatchdogStatus = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.get('/api/watchdog/status');
      setWatchdogStatus(response);
    } catch (err) {
      console.error('Error fetching watchdog status:', err);
      setError('Failed to load watchdog status');
    } finally {
      setLoading(false);
    }
  };

  // Initial fetch
  useEffect(() => {
    fetchWatchdogStatus();
  }, []);

  // Auto-refresh
  useEffect(() => {
    let intervalId;

    if (autoRefresh) {
      intervalId = setInterval(fetchWatchdogStatus, refreshInterval);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [autoRefresh, refreshInterval]);

  // Handle manual refresh
  const handleRefresh = () => {
    fetchWatchdogStatus();
  };

  // Handle recovery actions
  const handleRecoveryAction = async (action, scriptId) => {
    try {
      const response = await api.post('/api/watchdog/action', {
        action,
        script_id: scriptId
      });

      // Show success message
      if (action === 'reset') {
        setError(null); // Clear any previous errors
        // Create a temporary success message
        const successDiv = document.createElement('div');
        successDiv.className = 'alert alert-success fixed top-4 right-4 z-50 shadow-lg';
        successDiv.innerHTML = `<div class="flex items-center"><svg xmlns="http://www.w3.org/2000/svg" class="stroke-current flex-shrink-0 h-6 w-6 mr-2" fill="none" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg><span>Process ${scriptId} removed from stuck processes!</span></div>`;
        document.body.appendChild(successDiv);

        // Remove the success message after 3 seconds
        setTimeout(() => {
          document.body.removeChild(successDiv);
        }, 3000);
      }

      // Refresh after action
      setTimeout(fetchWatchdogStatus, 1000);
    } catch (err) {
      console.error(`Error performing ${action} action:`, err);
      setError(`Failed to perform ${action} action`);
    }
  };

  // Format time ago
  const formatTimeAgo = (isoString) => {
    const date = new Date(isoString);
    const now = new Date();
    const diffMs = now - date;

    const diffSecs = Math.floor(diffMs / 1000);
    if (diffSecs < 60) return `${diffSecs} seconds ago`;

    const diffMins = Math.floor(diffSecs / 60);
    if (diffMins < 60) return `${diffMins} minutes ago`;

    const diffHours = Math.floor(diffMins / 60);
    if (diffHours < 24) return `${diffHours} hours ago`;

    const diffDays = Math.floor(diffHours / 24);
    return `${diffDays} days ago`;
  };

  // Check if there are any stuck processes
  const hasStuckProcesses = watchdogStatus &&
    watchdogStatus.stuck_processes &&
    Object.keys(watchdogStatus.stuck_processes).length > 0;

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold">Process Watchdog</h2>
        <div className="flex space-x-2">
          <button
            className="btn btn-sm btn-outline"
            onClick={handleRefresh}
            disabled={loading}
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            <span className="ml-1">Refresh</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="alert alert-error mb-4">
          <AlertCircle size={16} className="mr-2" />
          {error}
        </div>
      )}

      {!watchdogStatus ? (
        <div className="flex justify-center items-center h-32">
          <div className="animate-pulse text-secondary-500">Loading watchdog status...</div>
        </div>
      ) : (
        <>
          {/* Watchdog Status Summary */}
          <div className="card bg-base-100 shadow-sm mb-4">
            <div className="card-body p-4">
              <h3 className="text-lg font-medium mb-2">Status Summary</h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="stat">
                  <div className="stat-title">Monitored Processes</div>
                  <div className="stat-value">{watchdogStatus.monitored_processes.length}</div>
                </div>
                <div className="stat">
                  <div className="stat-title">Stuck Processes</div>
                  <div className="stat-value text-error">{Object.keys(watchdogStatus.stuck_processes).length}</div>
                </div>
                <div className="stat">
                  <div className="stat-title">Status</div>
                  <div className="stat-value flex items-center">
                    {hasStuckProcesses ? (
                      <>
                        <AlertTriangle size={24} className="text-error mr-2" />
                        <span className="text-error">Issues Detected</span>
                      </>
                    ) : (
                      <>
                        <CheckCircle size={24} className="text-success mr-2" />
                        <span className="text-success">Healthy</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Stuck Processes */}
          {hasStuckProcesses ? (
            <div className="card bg-base-100 shadow-sm mb-4">
              <div className="card-body p-4">
                <h3 className="text-lg font-medium mb-2">Stuck Processes</h3>
                <div className="overflow-x-auto">
                  <table className="table w-full">
                    <thead>
                      <tr>
                        <th>Script ID</th>
                        <th>Issue</th>
                        <th>Detected</th>
                        <th>Recovery Attempts</th>
                        <th>Actions</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(watchdogStatus.stuck_processes).map(([scriptId, info]) => (
                        <tr key={scriptId} className="hover">
                          <td className="font-medium">{scriptId}</td>
                          <td>{info.reason}</td>
                          <td>{formatTimeAgo(info.detected_at)}</td>
                          <td>{info.recovery_attempts}</td>
                          <td>
                            <div className="flex space-x-2">
                              <button
                                className="btn btn-xs btn-outline"
                                onClick={() => handleRecoveryAction('force_gc', scriptId)}
                                title="Force Garbage Collection"
                              >
                                <Trash2 size={14} />
                              </button>
                              <button
                                className="btn btn-xs btn-outline btn-warning"
                                onClick={() => handleRecoveryAction('clear_cuda', scriptId)}
                                title="Clear CUDA Cache"
                              >
                                <Trash2 size={14} />
                                <span className="ml-1">CUDA</span>
                              </button>
                              <button
                                className="btn btn-xs btn-outline btn-error"
                                onClick={() => handleRecoveryAction('restart', scriptId)}
                                title="Restart Process"
                              >
                                <Play size={14} />
                                <span className="ml-1">Restart</span>
                              </button>
                              <button
                                className="btn btn-xs btn-outline btn-success"
                                onClick={() => handleRecoveryAction('reset', scriptId)}
                                title="Remove from Stuck Processes"
                              >
                                <Trash2 size={14} />
                                <span className="ml-1">Clear</span>
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          ) : (
            <div className="alert alert-success mb-4">
              <CheckCircle size={16} className="mr-2" />
              <div>
                <div className="font-semibold">All processes are running normally</div>
                <div className="text-sm">No stuck processes detected</div>
              </div>
            </div>
          )}

          {/* Watchdog Settings */}
          <div className="card bg-base-100 shadow-sm mb-4">
            <div className="card-body p-4">
              <h3 className="text-lg font-medium mb-2">Watchdog Settings</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm font-medium">Timeout</div>
                  <div>{watchdogStatus.settings.timeout_seconds} seconds</div>
                </div>
                <div>
                  <div className="text-sm font-medium">Memory Threshold</div>
                  <div>{watchdogStatus.settings.memory_threshold}%</div>
                </div>
                <div>
                  <div className="text-sm font-medium">Utilization Threshold</div>
                  <div>{watchdogStatus.settings.utilization_threshold}%</div>
                </div>
                <div>
                  <div className="text-sm font-medium">Check Interval</div>
                  <div>{watchdogStatus.settings.check_interval_seconds} seconds</div>
                </div>
              </div>
            </div>
          </div>

          {/* Monitored Processes */}
          <div className="card bg-base-100 shadow-sm mb-4">
            <div className="card-body p-4">
              <h3 className="text-lg font-medium mb-2">Monitored Processes</h3>
              {watchdogStatus.monitored_processes.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {watchdogStatus.monitored_processes.map(scriptId => (
                    <div key={scriptId} className="badge badge-outline p-3">
                      {scriptId}
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-secondary-500">No active processes being monitored</div>
              )}
            </div>
          </div>
        </>
      )}

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

export default WatchdogStatus;
