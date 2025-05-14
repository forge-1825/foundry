import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';

const LogContext = createContext();

export const useLog = () => {
  const context = useContext(LogContext);
  if (!context) {
    throw new Error('useLog must be used within a LogProvider');
  }
  return context;
};

export const LogProvider = ({ children }) => {
  const [activeScriptId, setActiveScriptId] = useState(null);
  const [logs, setLogs] = useState({});
  const [error, setError] = useState(null);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filterLevel, setFilterLevel] = useState('all'); // 'all', 'info', 'warning', 'error'

  // Connect to the logs WebSocket for the active script
  const { lastMessage, sendMessage, connected } = useWebSocket(
    activeScriptId ? `/ws/scripts/${activeScriptId}/logs` : null
  );

  // Debug log for WebSocket connection
  useEffect(() => {
    if (activeScriptId) {
      console.log(`WebSocket ${connected ? 'connected' : 'disconnected'} for script: ${activeScriptId}`);
    }
  }, [activeScriptId, connected]);

  // Update logs when new messages arrive
  useEffect(() => {
    if (lastMessage && activeScriptId) {
      console.log(`Received new log message for script: ${activeScriptId}`);

      // Check if the message is a JSON object (progress update)
      try {
        // Try to parse as JSON
        const jsonMessage = JSON.parse(lastMessage);
        console.log('Received JSON message:', jsonMessage);

        // If it's a progress update, don't add it to the logs
        if (jsonMessage.type === 'progress') {
          console.log('Progress update received:', jsonMessage.data);
          return;
        }
      } catch (e) {
        // Not JSON, continue with normal log handling
      }

      // Skip adding JSON messages to logs
      if (lastMessage.startsWith('{') && lastMessage.endsWith('}')) {
        console.log('Skipping JSON message');
        return;
      }

      setLogs(prevLogs => {
        const updatedLogs = {
          ...prevLogs,
          [activeScriptId]: [
            ...(prevLogs[activeScriptId] || []),
            lastMessage
          ]
        };
        console.log(`Updated logs for script: ${activeScriptId}, total logs: ${updatedLogs[activeScriptId].length}`);
        return updatedLogs;
      });
    }
  }, [lastMessage, activeScriptId]);

  // Set the active script and connect to its logs
  const connectToScriptLogs = useCallback((scriptId) => {
    setActiveScriptId(scriptId);
  }, []);

  // Add logs for a specific script
  const addLogs = useCallback((scriptId, newLogs) => {
    setLogs(prevLogs => ({
      ...prevLogs,
      [scriptId]: [...(prevLogs[scriptId] || []), ...newLogs]
    }));
  }, []);

  // Clear logs for a specific script
  const clearLogs = useCallback((scriptId) => {
    setLogs(prevLogs => ({
      ...prevLogs,
      [scriptId]: []
    }));
  }, []);

  // Clear all logs
  const clearAllLogs = useCallback(() => {
    setLogs({});
  }, []);

  // Filter logs by level
  const getFilteredLogs = useCallback((scriptId) => {
    console.log(`Getting filtered logs for script: ${scriptId}, filter level: ${filterLevel}`);
    console.log(`Available logs for script: ${scriptId}:`, logs[scriptId] ? logs[scriptId].length : 0);

    if (!logs[scriptId]) {
      console.log(`No logs found for script: ${scriptId}`);
      return [];
    }

    if (filterLevel === 'all') {
      console.log(`Returning all logs for script: ${scriptId}, count: ${logs[scriptId].length}`);
      return logs[scriptId];
    }

    const filteredLogs = logs[scriptId].filter(log => {
      const logLower = log.toLowerCase();
      switch (filterLevel) {
        case 'error':
          return logLower.includes('[error]');
        case 'warning':
          return logLower.includes('[warning]') || logLower.includes('[warn]');
        case 'info':
          return logLower.includes('[info]');
        default:
          return true;
      }
    });

    console.log(`Filtered logs for script: ${scriptId}, count: ${filteredLogs.length}`);
    return filteredLogs;
  }, [logs, filterLevel]);

  // Search logs
  const searchLogs = useCallback((scriptId, searchTerm) => {
    if (!logs[scriptId] || !searchTerm) return logs[scriptId] || [];

    const term = searchTerm.toLowerCase();
    return logs[scriptId].filter(log =>
      log.toLowerCase().includes(term)
    );
  }, [logs]);

  const value = {
    activeScriptId,
    logs,
    connected,
    error,
    autoScroll,
    filterLevel,
    connectToScriptLogs,
    clearLogs,
    clearAllLogs,
    setAutoScroll,
    setFilterLevel,
    getFilteredLogs,
    searchLogs,
    addLogs
  };

  return (
    <LogContext.Provider value={value}>
      {children}
    </LogContext.Provider>
  );
};
