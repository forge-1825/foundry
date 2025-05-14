import React, { useContext, useState } from 'react';
import { Link } from 'react-router-dom';
import { SystemContext } from '../contexts/SystemContext';
import { ScriptContext } from '../contexts/ScriptContext';
import { AlertCircle, RefreshCw, Server, Cpu, HardDrive, Activity } from 'lucide-react';
import MemoryMonitor from '../components/MemoryMonitor';
import WatchdogStatus from '../components/WatchdogStatus';

const Dashboard = () => {
  const {
    systemStatus,
    resourceUsage,
    watchdogStatus,
    dockerContainers,
    loading,
    error,
    isConnected,
    refreshAll
  } = useContext(SystemContext);

  const { activeScripts, scriptStatuses } = useContext(ScriptContext);
  const [selectedScript, setSelectedScript] = useState(null);

  // Handle refresh button click
  const handleRefresh = () => {
    refreshAll();
  };

  // Select a script for memory monitoring
  const handleSelectScript = (scriptId) => {
    setSelectedScript(scriptId);
  };

  // Check if there are any stuck processes
  const hasStuckProcesses = watchdogStatus &&
    watchdogStatus.stuck_processes &&
    Object.keys(watchdogStatus.stuck_processes || {}).length > 0;

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold">System Dashboard</h1>
        <div className="flex items-center">
          <button
            className="btn btn-sm btn-outline mr-2"
            onClick={handleRefresh}
            disabled={loading}
          >
            <RefreshCw size={16} className={loading ? 'animate-spin' : ''} />
            <span className="ml-1">Refresh</span>
          </button>
          {!isConnected && (
            <div className="badge badge-error gap-2">
              <AlertCircle size={12} />
              Disconnected
            </div>
          )}
          {isConnected && (
            <div className="badge badge-success gap-2">
              <Activity size={12} />
              Connected
            </div>
          )}
        </div>
      </div>

      {error && (
        <div className="alert alert-error mb-6">
          <AlertCircle size={16} className="mr-2" />
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* CPU Usage */}
        <div className="card bg-base-100 shadow-sm">
          <div className="card-body">
            <h2 className="card-title flex items-center">
              <Cpu size={20} className="mr-2" />
              CPU Usage
            </h2>
            <div className="mt-2">
              <div className="flex justify-between mb-1">
                <span>{resourceUsage?.cpu_percent?.toFixed(1) || '0.0'}%</span>
              </div>
              <progress
                className={`progress w-full ${(resourceUsage?.cpu_percent || 0) > 80 ? 'progress-error' : 'progress-primary'}`}
                value={resourceUsage?.cpu_percent || 0}
                max="100"
              ></progress>
            </div>
          </div>
        </div>

        {/* Memory Usage */}
        <div className="card bg-base-100 shadow-sm">
          <div className="card-body">
            <h2 className="card-title flex items-center">
              <HardDrive size={20} className="mr-2" />
              Memory Usage
            </h2>
            <div className="mt-2">
              <div className="flex justify-between mb-1">
                <span>{resourceUsage?.memory_percent?.toFixed(1) || '0.0'}%</span>
              </div>
              <progress
                className={`progress w-full ${(resourceUsage?.memory_percent || 0) > 80 ? 'progress-error' : 'progress-primary'}`}
                value={resourceUsage?.memory_percent || 0}
                max="100"
              ></progress>
            </div>
          </div>
        </div>

        {/* GPU Usage */}
        <div className="card bg-base-100 shadow-sm">
          <div className="card-body">
            <h2 className="card-title flex items-center">
              <Server size={20} className="mr-2" />
              GPU Usage
            </h2>
            {resourceUsage?.gpu_info ? (
              <div className="mt-2">
                <div className="flex justify-between mb-1">
                  <span>Memory: {resourceUsage.gpu_info?.memory_percent?.toFixed(1) || '0.0'}%</span>
                  <span>Util: {resourceUsage.gpu_info?.utilization?.toFixed(1) || '0.0'}%</span>
                </div>
                <progress
                  className={`progress w-full ${(resourceUsage.gpu_info?.memory_percent || 0) > 80 ? 'progress-error' : 'progress-primary'}`}
                  value={resourceUsage.gpu_info?.memory_percent || 0}
                  max="100"
                ></progress>
                <div className="mt-2">
                  <progress
                    className="progress progress-secondary w-full"
                    value={resourceUsage.gpu_info?.utilization || 0}
                    max="100"
                  ></progress>
                </div>
              </div>
            ) : (
              <div className="text-secondary-500 mt-2">No GPU detected</div>
            )}
          </div>
        </div>
      </div>

      {/* Active Scripts */}
      <div className="card bg-base-100 shadow-sm mb-6">
        <div className="card-body">
          <h2 className="card-title">Active Scripts</h2>
          {activeScripts && activeScripts.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="table w-full">
                <thead>
                  <tr>
                    <th>Script</th>
                    <th>Status</th>
                    <th>Progress</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {activeScripts.map(scriptId => {
                    const status = scriptStatuses[scriptId] || {};
                    const isStuck = watchdogStatus?.stuck_processes &&
                                   watchdogStatus.stuck_processes[scriptId];

                    return (
                      <tr key={scriptId} className={isStuck ? 'bg-error bg-opacity-10' : ''}>
                        <td>{scriptId}</td>
                        <td>
                          {isStuck ? (
                            <span className="badge badge-error">Stuck</span>
                          ) : status.status === 'running' ? (
                            <span className="badge badge-primary">Running</span>
                          ) : status.status === 'completed' ? (
                            <span className="badge badge-success">Completed</span>
                          ) : status.status === 'error' ? (
                            <span className="badge badge-error">Error</span>
                          ) : (
                            <span className="badge">Unknown</span>
                          )}
                        </td>
                        <td>
                          <div className="flex items-center">
                            <progress
                              className="progress progress-primary w-full mr-2"
                              value={status.progress_percent || 0}
                              max="100"
                            ></progress>
                            <span className="whitespace-nowrap">{(status.progress_percent || 0).toFixed(1)}%</span>
                          </div>
                        </td>
                        <td>
                          <div className="flex space-x-2">
                            <Link to={`/logs/${scriptId}`} className="btn btn-xs btn-outline">
                              Logs
                            </Link>
                            <button
                              className="btn btn-xs btn-outline"
                              onClick={() => handleSelectScript(scriptId)}
                            >
                              Memory
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-secondary-500">No active scripts</div>
          )}
        </div>
      </div>

      {/* Memory Monitor */}
      {selectedScript && (
        <div className="mb-6">
          <MemoryMonitor scriptId={selectedScript} />
        </div>
      )}

      {/* Watchdog Status */}
      <div className="mb-6">
        <WatchdogStatus />
      </div>

      {/* Docker Containers */}
      <div className="card bg-base-100 shadow-sm mb-6">
        <div className="card-body">
          <h2 className="card-title">Available Teacher Models</h2>
          <div className="flex justify-between items-center mb-2">
            <p className="text-sm text-gray-500">Models running in Docker containers</p>
            <button
              className="btn btn-xs btn-outline"
              onClick={handleRefresh}
              disabled={loading}
            >
              <RefreshCw size={12} className={loading ? 'animate-spin' : ''} />
              <span className="ml-1">Refresh</span>
            </button>
          </div>
          {dockerContainers && dockerContainers.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="table w-full">
                <thead>
                  <tr>
                    <th>Container</th>
                    <th>Type</th>
                    <th>Port</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {dockerContainers.map((container, index) => (
                    <tr key={index}>
                      <td>{container.name}</td>
                      <td>{container.type || 'Unknown'}</td>
                      <td>{container.port || 'N/A'}</td>
                      <td>
                        {container.status === 'running' ? (
                          <span className="badge badge-success">Running</span>
                        ) : (
                          <span className="badge badge-error">Stopped</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="text-secondary-500 py-4 text-center">
              <p>No Docker containers found</p>
              <p className="text-xs mt-2">Make sure your vLLM containers are running on ports 8000 and 8002</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
