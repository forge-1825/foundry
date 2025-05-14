import React, { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import ForgeFoundryLogo from '../../ForgeFoundry.png';
import {
  LayoutDashboard,
  Code,
  Terminal,
  FolderOpen,
  BarChart2,
  Menu,
  X,
  ChevronRight,
  Play,
  CheckCircle,
  AlertCircle,
  Loader,
  MessageSquare,
  HelpCircle,
  FileText,
  Lightbulb,
  Server,
  Cpu
} from 'lucide-react';
import { useSystem } from '../../contexts/SystemContext';
import { useScript } from '../../contexts/ScriptContext';
import { scriptService } from '../../services/scriptService';

const Layout = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = React.useState(true);
  const location = useLocation();
  const { systemStatus } = useSystem();
  const { scripts, executeScript } = useScript();

  // Pipeline state
  const [pipelineStatus, setPipelineStatus] = useState({});
  const [loadingPipeline, setLoadingPipeline] = useState(false);
  const [executingScript, setExecutingScript] = useState(null);

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen);
  };

  // Fetch pipeline status
  useEffect(() => {
    const fetchPipelineStatus = async () => {
      try {
        const status = await scriptService.getPipelineStatus();
        setPipelineStatus(status);
      } catch (error) {
        console.error('Error fetching pipeline status:', error);
      }
    };

    fetchPipelineStatus();

    // Refresh pipeline status every 10 seconds
    const interval = setInterval(fetchPipelineStatus, 10000);

    return () => clearInterval(interval);
  }, []);

  // Update pipeline status when active scripts change
  useEffect(() => {
    if (systemStatus && systemStatus.active_scripts) {
      const newStatus = { ...pipelineStatus };

      // Mark scripts as running if they are in active_scripts
      systemStatus.active_scripts.forEach(scriptId => {
        if (newStatus[scriptId] !== 'completed') {
          newStatus[scriptId] = 'running';
        }
      });

      // If a script was executing but is no longer in active_scripts, mark it as completed
      if (executingScript && !systemStatus.active_scripts.includes(executingScript)) {
        newStatus[executingScript] = 'completed';
        setExecutingScript(null);
      }

      setPipelineStatus(newStatus);
    }
  }, [systemStatus, executingScript, pipelineStatus]);

  // Execute a script in the pipeline
  const handleExecuteScript = async (scriptId) => {
    try {
      setExecutingScript(scriptId);
      setLoadingPipeline(true);

      // Update status to running
      setPipelineStatus(prev => ({
        ...prev,
        [scriptId]: 'running'
      }));

      // Execute the script
      await executeScript(scriptId);

    } catch (error) {
      console.error(`Error executing script ${scriptId}:`, error);

      // Update status to error
      setPipelineStatus(prev => ({
        ...prev,
        [scriptId]: 'error'
      }));

    } finally {
      setLoadingPipeline(false);
    }
  };

  const navItems = [
    { path: '/', label: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { path: '/scripts/content_extraction_enrichment', label: 'Content Extraction & Enrichment', icon: <Code size={20} /> },
    { path: '/scripts/teacher_pair_generation', label: 'Teacher Pair Generation', icon: <Server size={20} /> },
    { path: '/scripts/distillation', label: 'Distillation Training', icon: <Cpu size={20} /> },
    { path: '/scripts/student_self_study', label: 'Student Self-Study', icon: <FileText size={20} /> },
    { path: '/scripts/merge_model', label: 'Model Merging', icon: <Server size={20} /> },
    { path: '/scripts/evaluation', label: 'Model Evaluation', icon: <BarChart2 size={20} /> },
    { path: '/logs', label: 'Logs', icon: <Terminal size={20} /> },
    { path: '/files', label: 'Files', icon: <FolderOpen size={20} /> },
    { path: '/model-interaction', label: 'Model Chat', icon: <MessageSquare size={20} /> },
    { path: '/novelty-insights', label: 'Novelty Insights', icon: <Lightbulb size={20} /> },
    { path: '/help', label: 'Help', icon: <HelpCircle size={20} /> },
  ];

  // Ensure these scripts are not accessible
  const hiddenScripts = ['manual_extractor', 'data_enrichment'];

  return (
    <div className="flex h-screen bg-secondary-50">
      {/* Sidebar */}
      <div
        className={`fixed inset-y-0 left-0 z-50 w-64 bg-primary-900 text-white transform transition-transform duration-300 ease-in-out ${
          sidebarOpen ? 'translate-x-0' : '-translate-x-full'
        } md:relative md:translate-x-0`}
      >
        <div className="flex items-center justify-between h-16 px-4 border-b border-primary-700">
          <h1 className="text-xl font-bold text-white">Model Distillation</h1>
          <button
            onClick={toggleSidebar}
            className="md:hidden text-white"
          >
            <X size={20} />
          </button>
        </div>
        <nav className="mt-4">
          <ul>
            {navItems.map((item) => (
              <li key={item.path}>
                <Link
                  to={item.path}
                  className={`flex items-center px-4 py-3 text-sm ${
                    location.pathname === item.path ||
                    (item.path !== '/' && location.pathname.startsWith(item.path))
                      ? 'bg-primary-800 text-white'
                      : 'text-primary-100 hover:bg-primary-800'
                  }`}
                >
                  <span className="mr-3">{item.icon}</span>
                  {item.label}
                </Link>
              </li>
            ))}
          </ul>
        </nav>
        {/* ForgeFoundry Logo */}
        <div className="absolute bottom-0 left-0 right-0">
          <div className="flex justify-center p-2 bg-primary-800">
            <img
              src={ForgeFoundryLogo}
              alt="ForgeFoundry Logo"
              className="h-8 w-auto object-contain"
            />
          </div>
        </div>

        {systemStatus && systemStatus.cpu_percent !== undefined && (
          <div className="absolute bottom-0 left-0 right-0 p-4 pt-12 bg-primary-800 text-xs text-primary-200">
            <div className="flex justify-between mb-1">
              <span>CPU:</span>
              <span>{systemStatus.cpu_percent?.toFixed(1) || '0.0'}%</span>
            </div>
            <div className="w-full bg-primary-700 rounded-full h-1.5 mb-2">
              <div
                className="bg-primary-300 h-1.5 rounded-full"
                style={{ width: `${systemStatus.cpu_percent || 0}%` }}
              ></div>
            </div>
            <div className="flex justify-between mb-1">
              <span>Memory:</span>
              <span>{systemStatus.memory_percent?.toFixed(1) || '0.0'}%</span>
            </div>
            <div className="w-full bg-primary-700 rounded-full h-1.5 mb-2">
              <div
                className="bg-primary-300 h-1.5 rounded-full"
                style={{ width: `${systemStatus.memory_percent || 0}%` }}
              ></div>
            </div>
            {systemStatus.gpu_info && Array.isArray(systemStatus.gpu_info) && systemStatus.gpu_info.length > 0 && (
              <>
                <div className="flex justify-between mb-1">
                  <span>GPU:</span>
                  <span>{systemStatus.gpu_info[0].memory_percent?.toFixed(1) || '0.0'}%</span>
                </div>
                <div className="w-full bg-primary-700 rounded-full h-1.5 mb-2">
                  <div
                    className="bg-primary-300 h-1.5 rounded-full"
                    style={{ width: `${systemStatus.gpu_info[0].memory_percent || 0}%` }}
                  ></div>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-white shadow-sm h-16 flex items-center px-4">
          <button
            onClick={toggleSidebar}
            className="md:hidden mr-4 text-primary-900"
          >
            <Menu size={24} />
          </button>
          <div className="flex-1">
            <h2 className="text-xl font-semibold text-primary-900">
              {navItems.find(item =>
                location.pathname === item.path ||
                (item.path !== '/' && location.pathname.startsWith(item.path))
              )?.label || 'Dashboard'}
            </h2>
          </div>
          <div className="flex items-center space-x-3">
            {systemStatus && systemStatus.active_scripts && systemStatus.active_scripts.length > 0 && (
              <div className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-info-100 text-info-800">
                <span className="w-2 h-2 mr-1 bg-info-500 rounded-full"></span>
                {systemStatus.active_scripts.length} Active {systemStatus.active_scripts.length === 1 ? 'Script' : 'Scripts'}
              </div>
            )}
          </div>
        </header>

        {/* Pipeline Navigation */}
        <div className="bg-white border-b border-secondary-200 px-4 py-2">
          <div className="flex items-center justify-between">
            <div className="text-sm font-medium text-secondary-700">Pipeline:</div>
            <div className="flex items-center space-x-1">
              {scripts
                .filter(script => !script.hidden && script.id !== 'manual_extractor' && script.id !== 'data_enrichment') // Filter out hidden scripts
                .sort((a, b) => a.step - b.step)
                .map((script, index, filteredScripts) => (
                  <React.Fragment key={script.id}>
                    <div className="relative group">
                      <button
                        className={`flex items-center px-3 py-1 rounded text-sm ${
                          pipelineStatus[script.id] === 'completed'
                            ? 'bg-success-100 text-success-800'
                            : pipelineStatus[script.id] === 'running'
                            ? 'bg-primary-100 text-primary-800 animate-pulse'
                            : pipelineStatus[script.id] === 'error'
                            ? 'bg-error-100 text-error-800'
                            : 'bg-secondary-100 text-secondary-800 hover:bg-secondary-200'
                        }`}
                        onClick={() => handleExecuteScript(script.id)}
                        disabled={loadingPipeline || pipelineStatus[script.id] === 'running'}
                      >
                        {pipelineStatus[script.id] === 'completed' ? (
                          <CheckCircle size={16} className="mr-1" />
                        ) : pipelineStatus[script.id] === 'running' ? (
                          <Loader size={16} className="mr-1 animate-spin" />
                        ) : pipelineStatus[script.id] === 'error' ? (
                          <AlertCircle size={16} className="mr-1" />
                        ) : (
                          <Play size={16} className="mr-1" />
                        )}
                        {script.name}
                      </button>

                      {/* Tooltip */}
                      <div className="absolute z-10 w-48 p-2 mt-2 text-xs bg-secondary-800 text-white rounded shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-opacity">
                        <p className="font-semibold">{script.name}</p>
                        <p className="mt-1">{script.description}</p>
                        <p className="mt-1">
                          Status: {pipelineStatus[script.id] === 'completed'
                            ? 'Completed'
                            : pipelineStatus[script.id] === 'running'
                            ? 'Running'
                            : pipelineStatus[script.id] === 'error'
                            ? 'Error'
                            : 'Not Started'}
                        </p>
                        <div className="mt-1 text-secondary-300">Click to execute</div>
                      </div>
                    </div>

                    {/* Connector between steps */}
                    {index < filteredScripts.length - 1 && (
                      <ChevronRight size={16} className="text-secondary-400" />
                    )}
                  </React.Fragment>
                ))}
            </div>
            <Link
              to="/scripts/content_extraction_enrichment"
              className="text-sm text-primary-600 hover:text-primary-800"
            >
              View All
            </Link>
          </div>
        </div>

        {/* Content */}
        <main className="flex-1 overflow-auto p-4">
          {children}
        </main>
      </div>
    </div>
  );
};

export default Layout;
