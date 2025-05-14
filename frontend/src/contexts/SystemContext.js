import React, { createContext, useState, useEffect, useCallback, useContext } from 'react';
import { systemService } from '../services/systemService';
import { scriptService } from '../services/scriptService';

export const SystemContext = createContext();

// Custom hook to use the SystemContext
export const useSystem = () => {
  const context = useContext(SystemContext);
  if (!context) {
    throw new Error('useSystem must be used within a SystemProvider');
  }
  return context;
};

export const SystemProvider = ({ children }) => {
  const [systemStatus, setSystemStatus] = useState({
    cpu_percent: 0,
    memory_percent: 0,
    gpu_info: null,
    active_scripts: []
  });

  const [resourceUsage, setResourceUsage] = useState({
    cpu_percent: 0,
    memory_percent: 0,
    gpu_info: null,
    timestamp: new Date().toISOString()
  });

  const [watchdogStatus, setWatchdogStatus] = useState({
    monitored_processes: [],
    stuck_processes: {},
    settings: {
      timeout_seconds: 300,
      memory_threshold: 90,
      utilization_threshold: 10,
      check_interval_seconds: 60
    }
  });

  const [pipelineStatus, setPipelineStatus] = useState({});
  const [dockerContainers, setDockerContainers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isConnected, setIsConnected] = useState(true); // Set to true by default

  // Fetch initial system status
  const fetchSystemStatus = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const status = await systemService.getSystemStatus();
      setSystemStatus(status);

      const watchdog = await systemService.getWatchdogStatus();
      setWatchdogStatus(watchdog);

      const containers = await systemService.getDockerContainers();
      setDockerContainers(containers);

      try {
        const pipeline = await scriptService.getPipelineStatus();
        setPipelineStatus(pipeline);
      } catch (pipelineErr) {
        console.error('Error fetching pipeline status:', pipelineErr);
        // Don't set the main error for this, just log it
      }

      setIsConnected(true);
    } catch (err) {
      console.error('Error fetching system status:', err);
      setError('Failed to connect to backend services');
      setIsConnected(false);
    } finally {
      setLoading(false);
    }
  }, []);

  // Set up WebSocket connections for real-time updates
  useEffect(() => {
    // Handler for status updates
    const handleStatusUpdate = (data) => {
      if (data.type === 'status') {
        setSystemStatus({
          cpu_percent: data.data.cpu_percent,
          memory_percent: data.data.memory_percent,
          gpu_info: data.data.gpu_info,
          active_scripts: data.data.active_scripts
        });

        if (data.data.watchdog_status) {
          setWatchdogStatus(data.data.watchdog_status);
        }

        setIsConnected(true);
      }
    };

    // Handler for resource updates
    const handleResourceUpdate = (data) => {
      if (data.type === 'resources') {
        setResourceUsage({
          cpu_percent: data.data.cpu_percent,
          memory_percent: data.data.memory_percent,
          gpu_info: data.data.gpu_info,
          timestamp: data.data.timestamp
        });

        setIsConnected(true);
      }
    };

    // Set up error handlers
    const handleError = (error) => {
      console.error('WebSocket error:', error);
      setIsConnected(false);
    };

    // Connect to WebSockets
    const removeStatusListener = systemService.addStatusListener(handleStatusUpdate);
    const removeResourcesListener = systemService.addResourcesListener(handleResourceUpdate);

    // Initial data fetch
    fetchSystemStatus();

    // Clean up WebSocket connections when component unmounts
    return () => {
      removeStatusListener();
      removeResourcesListener();
    };
  }, [fetchSystemStatus]);

  // Refresh Docker containers periodically
  useEffect(() => {
    const fetchContainers = async () => {
      try {
        const containers = await systemService.getDockerContainers();
        setDockerContainers(containers);
      } catch (err) {
        console.error('Error fetching Docker containers:', err);
      }
    };

    // Fetch initially and then every 30 seconds
    fetchContainers();
    const interval = setInterval(fetchContainers, 30000);

    return () => clearInterval(interval);
  }, []);

  // Perform a watchdog action
  const performWatchdogAction = async (action, scriptId) => {
    try {
      const result = await systemService.performWatchdogAction(action, scriptId);
      // Refresh watchdog status after action
      const watchdog = await systemService.getWatchdogStatus();
      setWatchdogStatus(watchdog);
      return result;
    } catch (err) {
      console.error(`Error performing watchdog action ${action}:`, err);
      throw err;
    }
  };

  // Get memory usage for a script
  const getScriptMemory = async (scriptId) => {
    try {
      return await systemService.getScriptMemory(scriptId);
    } catch (err) {
      console.error(`Error getting memory usage for script ${scriptId}:`, err);
      throw err;
    }
  };

  // Manual refresh of all data
  const refreshAll = async () => {
    await fetchSystemStatus();
  };

  return (
    <SystemContext.Provider
      value={{
        systemStatus,
        resourceUsage,
        watchdogStatus,
        pipelineStatus,
        dockerContainers,
        loading,
        error,
        isConnected,
        refreshAll,
        performWatchdogAction,
        getScriptMemory
      }}
    >
      {children}
    </SystemContext.Provider>
  );
};
