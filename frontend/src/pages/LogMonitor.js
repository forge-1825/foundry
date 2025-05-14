import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Search,
  Filter,
  Download,
  Trash2,
  ArrowDown,
  ArrowRight,
  PauseCircle,
  PlayCircle,
  AlertCircle,
  InfoIcon,
  AlertTriangle
} from 'lucide-react';
import { useLog } from '../contexts/LogContext';
import { useScript } from '../contexts/ScriptContext';
import { useSystem } from '../contexts/SystemContext';
import PipelineNavigation from '../components/PipelineNavigation';

const LogMonitor = () => {
  const { scriptId } = useParams();
  const navigate = useNavigate();
  const { scripts } = useScript();
  const { systemStatus } = useSystem();
  const {
    activeScriptId,
    logs,
    connected,
    autoScroll,
    filterLevel,
    connectToScriptLogs,
    clearLogs,
    setAutoScroll,
    setFilterLevel,
    getFilteredLogs,
    searchLogs,
    addLogs
  } = useLog();

  const [selectedScript, setSelectedScript] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [displayedLogs, setDisplayedLogs] = useState([]);
  const logContainerRef = useRef(null);

  // Set selected script based on URL parameter
  useEffect(() => {
    if (scripts.length > 0) {
      if (scriptId) {
        // Find the script by ID, handling special case for content_extraction_enrichment
        const script = scripts.find(s => s.id === scriptId);
        if (script) {
          setSelectedScript(script);
          connectToScriptLogs(script.id);

          // Fetch initial logs for the script
          const fetchInitialLogs = async () => {
            try {
              console.log(`Fetching logs for script: ${script.id}`);
              const response = await fetch(`/api/scripts/${script.id}/logs?limit=1000`);
              if (response.ok) {
                const data = await response.json();
                console.log(`Received ${data.logs ? data.logs.length : 0} logs for script: ${script.id}`);
                if (data.logs && data.logs.length > 0) {
                  // Add logs to the context
                  addLogs(script.id, data.logs);
                  console.log(`Added ${data.logs.length} logs to context for script: ${script.id}`);
                } else {
                  console.log(`No logs available for script: ${script.id}`);
                }
              } else {
                console.error(`Failed to fetch logs for script: ${script.id}. Status: ${response.status}`);
              }
            } catch (error) {
              console.error('Error fetching initial logs:', error);
            }
          };

          fetchInitialLogs();
        } else {
          // If script ID is invalid, navigate to the first script
          navigate(`/logs/${scripts[0].id}`);
        }
      } else {
        // If no script ID is provided, navigate to the first script
        navigate(`/logs/${scripts[0].id}`);
      }
    }
  }, [scripts, scriptId, navigate, connectToScriptLogs, addLogs]);

  // Update displayed logs when logs, filter, or search changes
  useEffect(() => {
    if (selectedScript) {
      let filteredLogs = getFilteredLogs(selectedScript.id);

      if (searchTerm) {
        filteredLogs = filteredLogs.filter(log =>
          log.toLowerCase().includes(searchTerm.toLowerCase())
        );
      }

      setDisplayedLogs(filteredLogs);
    }
  }, [selectedScript, logs, filterLevel, searchTerm, getFilteredLogs]);

  // Auto-scroll to bottom when new logs arrive
  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [displayedLogs, autoScroll]);

  // Handle script selection
  const handleScriptSelect = (script) => {
    navigate(`/logs/${script.id}`);
  };

  // Handle log download
  const handleDownloadLogs = () => {
    if (!selectedScript || !displayedLogs.length) return;

    const logText = displayedLogs.join('\n');
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${selectedScript.id}_logs_${new Date().toISOString().replace(/[:.]/g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Handle log clear
  const handleClearLogs = () => {
    if (selectedScript) {
      clearLogs(selectedScript.id);
    }
  };

  // Get log entry class based on content
  const getLogEntryClass = (log) => {
    // Skip JSON messages
    if (log.startsWith('{') && log.endsWith('}')) {
      return 'hidden';
    }

    const logLower = log.toLowerCase();
    if (logLower.includes('[error]')) {
      return 'text-error-600';
    } else if (logLower.includes('[warning]') || logLower.includes('[warn]')) {
      return 'text-warning-600';
    } else if (logLower.includes('[info]')) {
      return 'text-primary-600';
    }
    return '';
  };

  // Get log entry icon based on content
  const getLogEntryIcon = (log) => {
    const logLower = log.toLowerCase();
    if (logLower.includes('[error]')) {
      return <AlertCircle size={16} className="mr-2 text-error-600 flex-shrink-0" />;
    } else if (logLower.includes('[warning]') || logLower.includes('[warn]')) {
      return <AlertTriangle size={16} className="mr-2 text-warning-600 flex-shrink-0" />;
    } else if (logLower.includes('[info]')) {
      return <InfoIcon size={16} className="mr-2 text-primary-600 flex-shrink-0" />;
    }
    return null;
  };

  // Check if the script is currently running
  const isScriptRunning = () => {
    return systemStatus &&
           systemStatus.active_scripts &&
           systemStatus.active_scripts.includes(selectedScript?.id);
  };

  // Get the next step in the pipeline
  const getNextPipelineStep = (currentScriptId) => {
    if (!currentScriptId || !scripts.length) return null;

    // Define the pipeline steps in order
    const pipelineOrder = [
      'content_extraction_enrichment',
      'teacher_pair_generation',
      'distillation',
      'student_self_study',
      'merge_model',
      'evaluation'
    ];

    // Find the current script's index in the pipeline
    const currentIndex = pipelineOrder.indexOf(currentScriptId);

    // If it's the last step or not found, return null
    if (currentIndex === -1 || currentIndex === pipelineOrder.length - 1) return null;

    // Get the next script ID
    const nextScriptId = pipelineOrder[currentIndex + 1];

    // Find the script object
    return scripts.find(script => script.id === nextScriptId);
  };

  // Handle navigation to the next step
  const handleNextStep = (currentScriptId) => {
    const nextStep = getNextPipelineStep(currentScriptId);
    if (nextStep) {
      console.log(`Navigating to next step: ${nextStep.id}`);
      navigate(`/scripts/${nextStep.id}`);
    }
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Pipeline Navigation */}
      <PipelineNavigation />

      <div className="flex flex-col md:flex-row h-full space-y-4 md:space-y-0 md:space-x-4">
      {/* Script Selection Sidebar */}
      <div className="w-full md:w-64 bg-white rounded-lg shadow-card p-4">
        <h3 className="text-lg font-medium mb-4">Script Logs</h3>
        <ul className="space-y-2">
          {scripts.map((script) => (
            <li key={script.id}>
              <button
                onClick={() => handleScriptSelect(script)}
                className={`w-full text-left px-3 py-2 rounded-md flex items-center ${
                  selectedScript?.id === script.id
                    ? 'bg-primary-100 text-primary-800'
                    : 'hover:bg-secondary-100'
                }`}
              >
                <div className="w-6 h-6 rounded-full bg-primary-100 flex items-center justify-center mr-2">
                  <span className="text-primary-700 text-xs font-semibold">{script.step}</span>
                </div>
                <span className="text-sm">{script.name}</span>
                {systemStatus &&
                 systemStatus.active_scripts &&
                 systemStatus.active_scripts.includes(script.id) && (
                  <span className="ml-auto">
                    <span className="flex h-3 w-3">
                      <span className="animate-ping absolute h-3 w-3 rounded-full bg-primary-400 opacity-75"></span>
                      <span className="relative rounded-full h-3 w-3 bg-primary-500"></span>
                    </span>
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Log Display */}
      <div className="flex-1 flex flex-col bg-white rounded-lg shadow-card">
        {selectedScript ? (
          <>
            {/* Log Controls */}
            <div className="p-4 border-b border-secondary-200">
              <div className="flex flex-col md:flex-row md:items-center justify-between space-y-2 md:space-y-0">
                <div className="flex items-center">
                  <h2 className="text-lg font-semibold mr-3">{selectedScript.name} Logs</h2>
                  {isScriptRunning() && (
                    <span className="badge badge-info flex items-center">
                      <span className="mr-1 flex h-2 w-2">
                        <span className="animate-ping absolute h-2 w-2 rounded-full bg-primary-400 opacity-75"></span>
                        <span className="relative rounded-full h-2 w-2 bg-primary-500"></span>
                      </span>
                      Running
                    </span>
                  )}
                  {connected ? (
                    <span className="ml-2 text-xs text-success-600">Connected</span>
                  ) : (
                    <span className="ml-2 text-xs text-error-600">Disconnected</span>
                  )}
                </div>
                <div className="flex items-center space-x-2">
                  <div className="relative">
                    <input
                      type="text"
                      placeholder="Search logs..."
                      value={searchTerm}
                      onChange={(e) => setSearchTerm(e.target.value)}
                      className="pl-8 pr-2 py-1 text-sm border border-secondary-300 rounded-md focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500"
                    />
                    <Search size={16} className="absolute left-2 top-1/2 transform -translate-y-1/2 text-secondary-400" />
                  </div>
                  <div className="relative">
                    <select
                      value={filterLevel}
                      onChange={(e) => setFilterLevel(e.target.value)}
                      className="pl-8 pr-2 py-1 text-sm border border-secondary-300 rounded-md focus:outline-none focus:ring-1 focus:ring-primary-500 focus:border-primary-500 appearance-none"
                    >
                      <option value="all">All Levels</option>
                      <option value="info">Info</option>
                      <option value="warning">Warning</option>
                      <option value="error">Error</option>
                    </select>
                    <Filter size={16} className="absolute left-2 top-1/2 transform -translate-y-1/2 text-secondary-400" />
                  </div>
                  <button
                    onClick={handleDownloadLogs}
                    className="p-1 text-secondary-700 hover:text-primary-600 rounded"
                    title="Download Logs"
                  >
                    <Download size={20} />
                  </button>
                  <button
                    onClick={handleClearLogs}
                    className="p-1 text-secondary-700 hover:text-error-600 rounded"
                    title="Clear Logs"
                  >
                    <Trash2 size={20} />
                  </button>
                  <button
                    onClick={() => setAutoScroll(!autoScroll)}
                    className={`p-1 rounded ${autoScroll ? 'text-primary-600' : 'text-secondary-700'}`}
                    title={autoScroll ? "Disable Auto-scroll" : "Enable Auto-scroll"}
                  >
                    {autoScroll ? <PauseCircle size={20} /> : <PlayCircle size={20} />}
                  </button>
                </div>
              </div>
            </div>

            {/* Log Content */}
            <div
              ref={logContainerRef}
              className="flex-1 overflow-auto p-4 font-mono text-sm bg-secondary-50"
            >
              {displayedLogs.length > 0 ? (
                <div className="space-y-1">
                  {displayedLogs.map((log, index) => (
                    <div
                      key={index}
                      className={`flex items-start ${getLogEntryClass(log)}`}
                    >
                      {getLogEntryIcon(log)}
                      <pre className="whitespace-pre-wrap break-all">{log}</pre>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-secondary-500">
                  No logs available
                </div>
              )}
            </div>

            {/* Next Step Button */}
            <div className="p-4 border-t border-secondary-200 flex justify-between items-center">
              <div>
                {selectedScript && (
                  <div className="text-sm text-secondary-600">
                    Current Step: <span className="font-semibold">{selectedScript.name}</span>
                    {isScriptRunning() && (
                      <span className="ml-2 text-warning-600 text-xs">(Running...)</span>
                    )}
                  </div>
                )}
              </div>
              <div>
                {selectedScript && getNextPipelineStep(selectedScript.id) && (
                  <button
                    onClick={() => handleNextStep(selectedScript.id)}
                    className={`btn flex items-center ${isScriptRunning() ? 'btn-disabled opacity-50 cursor-not-allowed' : 'btn-primary'}`}
                    disabled={isScriptRunning()}
                    title={isScriptRunning() ? 'Wait for the current script to complete' : `Proceed to ${getNextPipelineStep(selectedScript.id).name}`}
                  >
                    {isScriptRunning() ? 'Processing...' : `Proceed to ${getNextPipelineStep(selectedScript.id).name}`}
                    <ArrowRight size={16} className="ml-2" />
                  </button>
                )}
              </div>
            </div>

            {/* Auto-scroll Indicator */}
            {autoScroll && displayedLogs.length > 0 && (
              <div className="absolute bottom-20 right-4">
                <div className="bg-primary-600 text-white px-3 py-1 rounded-full flex items-center text-xs shadow-lg">
                  <ArrowDown size={12} className="mr-1" />
                  Auto-scrolling
                </div>
              </div>
            )}
          </>
        ) : (
          <div className="flex items-center justify-center h-full">
            <p className="text-secondary-500">Select a script from the sidebar</p>
          </div>
        )}
      </div>
      </div>
    </div>
  );
};

export default LogMonitor;
