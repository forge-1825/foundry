import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Play,
  Save,
  Upload,
  Download,
  AlertCircle,
  ChevronRight,
  ChevronDown,
  Info,
  Send,
  Server
} from 'lucide-react';
import { useScript } from '../contexts/ScriptContext';
import { useSystem } from '../contexts/SystemContext';
import { scriptService } from '../services/scriptService';
import { modelService } from '../services/modelService';
import { systemService } from '../services/systemService';
import PipelineNavigation from '../components/PipelineNavigation';

const ScriptConfiguration = () => {
  const { scriptId } = useParams();
  const navigate = useNavigate();
  const { scripts, configs, updateConfig, executeScript, loading, error } = useScript();
  const { systemStatus, pipelineStatus } = useSystem();
  const [selectedScript, setSelectedScript] = useState(null);
  const [configName, setConfigName] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [showLoadDialog, setShowLoadDialog] = useState(false);
  const [savedConfigs, setSavedConfigs] = useState([]);
  const [advancedMode, setAdvancedMode] = useState(false);
  const [executionError, setExecutionError] = useState(null);
  const [selectedSource, setSelectedSource] = useState('url'); // 'url', 'source_folder', or 'docker_folder'
  const [validationErrors, setValidationErrors] = useState({});
  const [dockerContainers, setDockerContainers] = useState([]);
  const [loadingContainers, setLoadingContainers] = useState(false);

  // Model query state
  const [modelPrompt, setModelPrompt] = useState('');
  const [modelResponse, setModelResponse] = useState('');
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState(null);

  // Load saved configurations and Docker containers
  useEffect(() => {
    const fetchSavedConfigs = async () => {
      try {
        const configs = await scriptService.listConfigs();
        setSavedConfigs(configs);
      } catch (error) {
        console.error('Error fetching saved configurations:', error);
      }
    };

    const fetchDockerContainers = async () => {
      try {
        setLoadingContainers(true);
        const containers = await systemService.getDockerContainers();
        setDockerContainers(containers);
      } catch (error) {
        console.error('Error fetching Docker containers:', error);
      } finally {
        setLoadingContainers(false);
      }
    };

    fetchSavedConfigs();
    fetchDockerContainers();
  }, []);

  // Set selected script based on URL parameter
  useEffect(() => {
    if (scripts.length > 0) {
      if (scriptId) {
        const script = scripts.find(s => s.id === scriptId);
        if (script) {
          setSelectedScript(script);
        } else {
          // If script ID is invalid, navigate to the first script
          navigate(`/scripts/${scripts[0].id}`);
        }
      } else {
        // If no script ID is provided, navigate to the first script
        navigate(`/scripts/${scripts[0].id}`);
      }
    }
  }, [scripts, scriptId, navigate]);

  // Handle script selection
  const handleScriptSelect = (script) => {
    navigate(`/scripts/${script.id}`);
  };

  // Handle form input change
  const handleInputChange = (key, value) => {
    if (selectedScript) {
      updateConfig(selectedScript.id, { [key]: value });
    }
  };

  // Validate configuration before execution
  const validateConfig = () => {
    const errors = {};

    if (selectedScript && selectedScript.id === 'manual_extractor') {
      const config = configs[selectedScript.id];

      // Validate based on selected source
      if (selectedSource === 'url') {
        if (!config.url || config.url.trim() === '') {
          errors.url = 'URL is required';
        }
      } else if (selectedSource === 'source_folder') {
        if (!config.source_folder || config.source_folder.trim() === '') {
          errors.source_folder = 'Source folder path is required';
        }
      } else if (selectedSource === 'docker_folder') {
        if (!config.docker_folder || config.docker_folder.trim() === '') {
          errors.docker_folder = 'Docker folder path is required';
        }
      }

      // Validate output directory
      if (!config.output_dir || config.output_dir.trim() === '') {
        errors.output_dir = 'Output directory is required';
      }
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // Handle script execution
  const handleExecute = async () => {
    if (selectedScript) {
      setExecutionError(null);

      // Validate configuration before execution
      if (!validateConfig()) {
        setExecutionError('Please fix the validation errors before executing the script.');
        return;
      }

      // Prepare configuration based on selected source
      let executionConfig = { ...configs[selectedScript.id] };

      // For manual extractor, only include the selected source
      if (selectedScript.id === 'manual_extractor') {
        if (selectedSource === 'url') {
          executionConfig.source_folder = '';
          executionConfig.docker_folder = '';
        } else if (selectedSource === 'source_folder') {
          executionConfig.url = '';
          executionConfig.docker_folder = '';
        } else if (selectedSource === 'docker_folder') {
          executionConfig.url = '';
          executionConfig.source_folder = '';

          // Store the docker_folder value for data_enrichment
          if (executionConfig.docker_folder) {
            // Update the data_enrichment config to use the same docker folder
            updateConfig('data_enrichment', {
              source_folder: executionConfig.docker_folder
            });
          }
        }
      }

      const success = await executeScript(selectedScript.id, executionConfig);
      if (success) {
        // Navigate to logs page to show execution progress
        navigate(`/logs/${selectedScript.id}`);

        // For teacher_pair_generation, set up a WebSocket connection to listen for completion
        if (selectedScript.id === 'teacher_pair_generation') {
          // Create a WebSocket connection to listen for logs
          const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/logs/${selectedScript.id}`;
          const socket = new WebSocket(wsUrl);

          let scriptCompleted = false;
          let scriptFailed = false;

          socket.onmessage = (event) => {
            const message = event.data;

            // Check if the message indicates successful completion
            if (message.includes('Teacher Pair Generation Script Completed') ||
                message.includes('completed successfully') ||
                (message.includes('progress') && message.includes('100.0'))) {

              // Only show the dialog once
              if (!scriptCompleted) {
                scriptCompleted = true;

                // Ask user if they want to proceed to the next step in the pipeline
                const proceedToNext = window.confirm('Teacher Pair Generation completed successfully! Would you like to proceed to the next step in the distillation pipeline (Distillation Phase)?');

                if (proceedToNext) {
                  // Navigate to the Distillation page
                  navigate('/scripts/distillation');
                }

                // Close the WebSocket connection
                socket.close();
              }
            }

            // Check if the message indicates an error
            if (message.includes('[ERROR]')) {
              scriptFailed = true;
            }
          };

          // Set a timeout as a fallback
          setTimeout(() => {
            if (!scriptCompleted && !scriptFailed) {
              // If we haven't received a completion message after 30 seconds, assume it completed
              scriptCompleted = true;

              // Ask user if they want to proceed to the next step in the pipeline
              const proceedToNext = window.confirm('Teacher Pair Generation appears to have completed. Would you like to proceed to the next step in the distillation pipeline (Distillation Phase)?');

              if (proceedToNext) {
                // Navigate to the Distillation page
                navigate('/scripts/distillation');
              }

              socket.close();
            }
          }, 30000); // Show dialog after 30 seconds if no completion message is received
        }

        // For distillation, set up a WebSocket connection to listen for completion
        if (selectedScript.id === 'distillation') {
          // Create a WebSocket connection to listen for logs
          const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/logs/${selectedScript.id}`;
          const socket = new WebSocket(wsUrl);

          let scriptCompleted = false;
          let scriptFailed = false;

          socket.onmessage = (event) => {
            const message = event.data;

            // Check if the message indicates successful completion
            if (message.includes('Distillation completed successfully') ||
                message.includes('completed successfully') ||
                (message.includes('progress') && message.includes('100.0'))) {

              // Only show the dialog once
              if (!scriptCompleted) {
                scriptCompleted = true;

                // Ask user if they want to proceed to the next step in the pipeline
                const proceedToNext = window.confirm('Distillation completed successfully! Would you like to proceed to the next step in the distillation pipeline (Model Merging)?');

                if (proceedToNext) {
                  // Navigate to the Model Merging page
                  navigate('/scripts/merge_model');
                }

                // Close the WebSocket connection
                socket.close();
              }
            }

            // Check if the message indicates an error
            if (message.includes('[ERROR]')) {
              scriptFailed = true;
            }
          };

          // Set a timeout as a fallback
          setTimeout(() => {
            if (!scriptCompleted && !scriptFailed) {
              // If we haven't received a completion message after 30 seconds, assume it completed
              scriptCompleted = true;

              // Ask user if they want to proceed to the next step in the pipeline
              const proceedToNext = window.confirm('Distillation appears to have completed. Would you like to proceed to the next step in the distillation pipeline (Model Merging)?');

              if (proceedToNext) {
                // Navigate to the Model Merging page
                navigate('/scripts/merge_model');
              }

              socket.close();
            }
          }, 30000); // Show dialog after 30 seconds if no completion message is received
        }

        // For model merging, set up a WebSocket connection to listen for completion
        if (selectedScript.id === 'merge_model') {
          // Create a WebSocket connection to listen for logs
          const wsUrl = `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/logs/${selectedScript.id}`;
          const socket = new WebSocket(wsUrl);

          let scriptCompleted = false;
          let scriptFailed = false;

          socket.onmessage = (event) => {
            const message = event.data;

            // Check if the message indicates successful completion
            if (message.includes('Model merging completed successfully') ||
                message.includes('Merged model saved successfully') ||
                message.includes('completed successfully') ||
                (message.includes('progress') && message.includes('100.0'))) {

              // Only show the dialog once
              if (!scriptCompleted) {
                scriptCompleted = true;

                // Ask user if they want to proceed to the next step in the pipeline
                const proceedToNext = window.confirm('Model Merging completed successfully! Would you like to proceed to the next step in the distillation pipeline (Student Self-Study)?');

                if (proceedToNext) {
                  // Navigate to the Student Self-Study page
                  navigate('/scripts/student_self_study');
                }

                // Close the WebSocket connection
                socket.close();
              }
            }

            // Check if the message indicates an error
            if (message.includes('[ERROR]')) {
              scriptFailed = true;
            }
          };

          // Set a timeout as a fallback
          setTimeout(() => {
            if (!scriptCompleted && !scriptFailed) {
              // If we haven't received a completion message after 30 seconds, assume it completed
              scriptCompleted = true;

              // Ask user if they want to proceed to the next step in the pipeline
              const proceedToNext = window.confirm('Model Merging appears to have completed. Would you like to proceed to the next step in the distillation pipeline (Student Self-Study)?');

              if (proceedToNext) {
                // Navigate to the Student Self-Study page
                navigate('/scripts/student_self_study');
              }

              socket.close();
            }
          }, 30000); // Show dialog after 30 seconds if no completion message is received
        }
      } else {
        setExecutionError('Failed to execute script. Check the console for details.');
      }
    }
  };

  // Handle save configuration
  const handleSaveConfig = async () => {
    if (selectedScript) {
      const success = await scriptService.saveConfig({
        name: configName || `${selectedScript.id}_config_${Date.now()}`,
        scriptId: selectedScript.id,
        config: configs[selectedScript.id]
      });

      if (success) {
        setShowSaveDialog(false);
        setConfigName('');
        // Refresh saved configs
        const configs = await scriptService.listConfigs();
        setSavedConfigs(configs);
      }
    }
  };

  // Handle load configuration
  const handleLoadConfig = async (config) => {
    if (config && config.scriptId && config.config) {
      updateConfig(config.scriptId, config.config);
      setShowLoadDialog(false);
      // Navigate to the script if it's different from the current one
      if (config.scriptId !== selectedScript?.id) {
        navigate(`/scripts/${config.scriptId}`);
      }
    }
  };

  // Validate a path to ensure it exists
  const validatePath = (path) => {
    // Simple validation for now - check if path is not empty
    return path && path.trim() !== '';
  };

  // Handle source selection
  const handleSourceSelect = (source) => {
    setSelectedSource(source);
    // Clear validation errors
    setValidationErrors({});
  };

  // Render form fields based on configuration
  const renderFormFields = () => {
    if (!selectedScript || !configs[selectedScript.id]) return null;

    const config = configs[selectedScript.id];

    // Special handling for manual extractor script
    if (selectedScript.id === 'manual_extractor') {
      return (
        <div className="space-y-4">
          {/* URL field with checkbox */}
          <div className="form-group">
            <div className="flex items-center mb-2">
              <input
                id="use_url"
                type="checkbox"
                checked={selectedSource === 'url'}
                onChange={() => handleSourceSelect('url')}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor="use_url" className="ml-2 block text-sm font-medium text-secondary-700">
                Use URL
              </label>
            </div>
            <input
              id={`${selectedScript.id}_url`}
              type="text"
              value={config.url || ''}
              onChange={(e) => handleInputChange('url', e.target.value)}
              className={`form-input ${selectedSource !== 'url' ? 'opacity-50' : ''}`}
              disabled={selectedSource !== 'url'}
            />
          </div>

          {/* Source folder field with checkbox */}
          <div className="form-group">
            <div className="flex items-center mb-2">
              <input
                id="use_source_folder"
                type="checkbox"
                checked={selectedSource === 'source_folder'}
                onChange={() => handleSourceSelect('source_folder')}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor="use_source_folder" className="ml-2 block text-sm font-medium text-secondary-700">
                Select Source Folder
              </label>
            </div>
            <div className="flex space-x-2">
              <input
                id={`${selectedScript.id}_source_folder`}
                type="text"
                value={config.source_folder || ''}
                onChange={(e) => handleInputChange('source_folder', e.target.value)}
                className={`form-input flex-grow ${selectedSource !== 'source_folder' ? 'opacity-50' : ''} ${
                  validationErrors.source_folder ? 'border-error-500' : ''
                }`}
                disabled={selectedSource !== 'source_folder'}
              />
              {selectedSource === 'source_folder' && (
                <button
                  type="button"
                  className="btn btn-secondary text-xs"
                  onClick={() => {
                    // Extract the last part of the Windows path
                    const sourcePath = config.source_folder || '';
                    if (sourcePath) {
                      const pathParts = sourcePath.split('\\');
                      const lastFolder = pathParts[pathParts.length - 1];
                      // Set the Docker folder path
                      const dockerPath = `/data/${lastFolder}`;
                      handleInputChange('docker_folder', dockerPath);
                      // Switch to Docker folder source
                      handleSourceSelect('docker_folder');
                    }
                  }}
                >
                  Send to Docker
                </button>
              )}
            </div>
            {validationErrors.source_folder && (
              <p className="text-error-500 text-sm mt-1">{validationErrors.source_folder}</p>
            )}
            {selectedSource === 'source_folder' && (
              <p className="text-xs text-secondary-500 mt-1">
                Windows paths may not be accessible from Docker. Use "Send to Docker" to convert.
              </p>
            )}
          </div>

          {/* Docker folder field with checkbox */}
          <div className="form-group">
            <div className="flex items-center mb-2">
              <input
                id="use_docker_folder"
                type="checkbox"
                checked={selectedSource === 'docker_folder'}
                onChange={() => handleSourceSelect('docker_folder')}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor="use_docker_folder" className="ml-2 block text-sm font-medium text-secondary-700">
                Select Docker Folder
              </label>
            </div>
            <input
              id={`${selectedScript.id}_docker_folder`}
              type="text"
              value={config.docker_folder || '/data'}
              onChange={(e) => handleInputChange('docker_folder', e.target.value)}
              className={`form-input ${selectedSource !== 'docker_folder' ? 'opacity-50' : ''} ${
                validationErrors.docker_folder ? 'border-error-500' : ''
              }`}
              disabled={selectedSource !== 'docker_folder'}
              placeholder="/data"
            />
            {validationErrors.docker_folder && (
              <p className="text-error-500 text-sm mt-1">{validationErrors.docker_folder}</p>
            )}
            <p className="text-xs text-secondary-500 mt-1">
              Use this for folders inside the Docker container (e.g., /data)
            </p>
          </div>

          {/* Output directory field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_output_dir`} className="form-label">
              Output Directory
            </label>
            <input
              id={`${selectedScript.id}_output_dir`}
              type="text"
              value={config.output_dir || ''}
              onChange={(e) => handleInputChange('output_dir', e.target.value)}
              className={`form-input ${validationErrors.output_dir ? 'border-error-500' : ''}`}
            />
            {validationErrors.output_dir && (
              <p className="text-error-500 text-sm mt-1">{validationErrors.output_dir}</p>
            )}
          </div>

          {/* Extract links checkbox */}
          <div className="form-group">
            <div className="flex items-center">
              <input
                id={`${selectedScript.id}_extract_links`}
                type="checkbox"
                checked={config.extract_links || false}
                onChange={(e) => handleInputChange('extract_links', e.target.checked)}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor={`${selectedScript.id}_extract_links`} className="ml-2 block text-sm font-medium text-secondary-700">
                Extract Links
              </label>
            </div>
          </div>
        </div>
      );
    }

    // Special handling for data enrichment script
    if (selectedScript.id === 'data_enrichment') {
      return (
        <div className="space-y-4">
          {/* Input file field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_input_file`} className="form-label">
              Input File
            </label>
            <input
              id={`${selectedScript.id}_input_file`}
              type="text"
              value={config.input_file || ''}
              onChange={(e) => handleInputChange('input_file', e.target.value)}
              className="form-input"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Path to the extracted data JSON file from Manual Extractor
            </p>
          </div>

          {/* Output file field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_output_file`} className="form-label">
              Output File
            </label>
            <input
              id={`${selectedScript.id}_output_file`}
              type="text"
              value={config.output_file || ''}
              onChange={(e) => handleInputChange('output_file', e.target.value)}
              className="form-input"
            />
          </div>

          {/* Source folder field with Search button */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_source_folder`} className="form-label">
              Source Folder
            </label>
            <div className="flex space-x-2">
              <input
                id={`${selectedScript.id}_source_folder`}
                type="text"
                value={config.source_folder || ''}
                onChange={(e) => handleInputChange('source_folder', e.target.value)}
                className="form-input flex-grow"
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  // Create a temporary file input element
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.webkitdirectory = true; // Allow directory selection
                  input.directory = true; // Non-standard attribute for directory selection

                  // Handle file selection
                  input.onchange = (e) => {
                    if (e.target.files.length > 0) {
                      // Get the directory path from the first file
                      const file = e.target.files[0];
                      const path = file.webkitRelativePath.split('/')[0];

                      // Construct the full path (this is an approximation as browsers restrict full path access)
                      // In a real implementation, you might need to use a backend API to get the full path
                      const fullPath = path;

                      // Update the source folder input
                      handleInputChange('source_folder', fullPath);
                    }
                  };

                  // Trigger the file dialog
                  input.click();
                }}
              >
                Search
              </button>
            </div>
            <p className="text-xs text-secondary-500 mt-1">
              This will be automatically set if you use "Send to Docker" in Manual Extractor
            </p>
          </div>
        </div>
      );
    }

    // Special handling for teacher pair generation script
    if (selectedScript.id === 'teacher_pair_generation') {
      return (
        <div className="space-y-4">
          {/* Input file field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_input_file`} className="form-label">
              Input File
            </label>
            <input
              id={`${selectedScript.id}_input_file`}
              type="text"
              value={config.input_file || ''}
              onChange={(e) => handleInputChange('input_file', e.target.value)}
              className="form-input"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Path to the enriched data JSON file from Data Enrichment
            </p>
          </div>

          {/* Output file field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_output_file`} className="form-label">
              Output File
            </label>
            <input
              id={`${selectedScript.id}_output_file`}
              type="text"
              value={config.output_file || ''}
              onChange={(e) => handleInputChange('output_file', e.target.value)}
              className="form-input"
            />
          </div>

          {/* Teacher model dropdown */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_teacher_model`} className="form-label flex items-center">
              Teacher Model
              <Server size={16} className="ml-2 text-secondary-500" />
            </label>
            <div className="relative">
              <select
                id={`${selectedScript.id}_teacher_model`}
                value={config.teacher_model || ''}
                onChange={(e) => handleInputChange('teacher_model', e.target.value)}
                className="form-input appearance-none pr-10"
                disabled={loadingContainers}
              >
                {loadingContainers ? (
                  <option value="">Loading containers...</option>
                ) : (
                  <>
                    <option value="">Select a teacher model</option>
                    {dockerContainers
                      .filter(container => container.type && container.type.toLowerCase().includes('teacher'))
                      .map((container) => (
                        <option
                          key={container.name}
                          value={container.name}
                          disabled={container.status !== 'running'}
                        >
                          {container.name} - {container.type}
                        </option>
                      ))}
                    {/* Add a fallback option if no teacher models are found */}
                    {!dockerContainers.some(c => c.type && c.type.toLowerCase().includes('teacher')) && (
                      <option value="phi4_gptq_vllm">phi4_gptq_vllm - Teacher Model (Phi-4)</option>
                    )}
                  </>
                )}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-secondary-700">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
              </div>
            </div>
            <p className="text-xs text-secondary-500 mt-1">
              Select a Docker container running a vLLM server or use the default model
            </p>
          </div>
        </div>
      );
    }

    // Special handling for distillation script
    if (selectedScript.id === 'distillation') {
      return (
        <div className="space-y-4">
          {/* Input file field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_input_file`} className="form-label">
              Input File
            </label>
            <input
              id={`${selectedScript.id}_input_file`}
              type="text"
              value={config.input_file || ''}
              onChange={(e) => handleInputChange('input_file', e.target.value)}
              className="form-input"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Path to the teacher pairs JSON file from Teacher Pair Generation
            </p>
          </div>

          {/* Output directory field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_output_dir`} className="form-label">
              Output Directory
            </label>
            <input
              id={`${selectedScript.id}_output_dir`}
              type="text"
              value={config.output_dir || ''}
              onChange={(e) => handleInputChange('output_dir', e.target.value)}
              className="form-input"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Directory where the distilled model will be saved
            </p>
          </div>

          {/* Teacher model dropdown */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_teacher_model`} className="form-label flex items-center">
              Teacher Model
              <Server size={16} className="ml-2 text-secondary-500" />
            </label>
            <div className="relative">
              <select
                id={`${selectedScript.id}_teacher_model`}
                value={config.teacher_model || ''}
                onChange={(e) => handleInputChange('teacher_model', e.target.value)}
                className="form-input appearance-none pr-10"
                disabled={loadingContainers}
              >
                {loadingContainers ? (
                  <option value="">Loading containers...</option>
                ) : (
                  <>
                    <option value="">Select a teacher model</option>
                    {dockerContainers
                      .filter(container => container.type && container.type.toLowerCase().includes('teacher'))
                      .map((container) => (
                        <option
                          key={container.name}
                          value={container.name}
                          disabled={container.status !== 'running'}
                        >
                          {container.name} - {container.type}
                        </option>
                      ))}
                    {/* Add a fallback option if no teacher models are found */}
                    {!dockerContainers.some(c => c.type && c.type.toLowerCase().includes('teacher')) && (
                      <option value="phi4_gptq_vllm">phi4_gptq_vllm - Teacher Model (Phi-4)</option>
                    )}
                  </>
                )}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-secondary-700">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
              </div>
            </div>
            <p className="text-xs text-secondary-500 mt-1">
              Select a Docker container running the teacher model
            </p>
          </div>

          {/* Student model dropdown */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_student_model`} className="form-label flex items-center">
              Student Model
              <Server size={16} className="ml-2 text-secondary-500" />
            </label>
            <div className="relative">
              <select
                id={`${selectedScript.id}_student_model`}
                value={config.student_model || ''}
                onChange={(e) => handleInputChange('student_model', e.target.value)}
                className="form-input appearance-none pr-10"
                disabled={loadingContainers}
              >
                {loadingContainers ? (
                  <option value="">Loading containers...</option>
                ) : (
                  <>
                    <option value="">Select a student model</option>
                    {dockerContainers
                      .filter(container => container.type && container.type.toLowerCase().includes('student'))
                      .map((container) => (
                        <option
                          key={container.name}
                          value={container.name}
                          disabled={container.status !== 'running'}
                        >
                          {container.name} - {container.type}
                        </option>
                      ))}
                    {/* Add a fallback option if no student models are found */}
                    {!dockerContainers.some(c => c.type && c.type.toLowerCase().includes('student')) && (
                      <option value="phi3_gptq_vllm">phi3_gptq_vllm - Student Model (Phi-3)</option>
                    )}
                  </>
                )}
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-secondary-700">
                <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
                  <path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z" />
                </svg>
              </div>
            </div>
            <p className="text-xs text-secondary-500 mt-1">
              Select a Docker container running the student model
            </p>
          </div>

          {/* Training Parameters Section */}
          <div className="mt-6 mb-2">
            <h3 className="text-lg font-medium">Training Parameters</h3>
            <p className="text-xs text-secondary-500">Configure the training process for optimal results</p>
          </div>

          {/* Number of epochs field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_num_epochs`} className="form-label">
              Number of Epochs
            </label>
            <input
              id={`${selectedScript.id}_num_epochs`}
              type="number"
              value={config.num_epochs || 3}
              onChange={(e) => handleInputChange('num_epochs', parseInt(e.target.value))}
              className="form-input"
              min="1"
              max="10"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Number of training epochs (3-5 recommended)
            </p>
          </div>

          {/* Batch size field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_batch_size`} className="form-label">
              Batch Size
            </label>
            <input
              id={`${selectedScript.id}_batch_size`}
              type="number"
              value={config.batch_size || 4}
              onChange={(e) => handleInputChange('batch_size', parseInt(e.target.value))}
              className="form-input"
              min="1"
              max="32"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Training batch size (smaller values use less memory)
            </p>
          </div>

          {/* Learning rate field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_learning_rate`} className="form-label">
              Learning Rate
            </label>
            <input
              id={`${selectedScript.id}_learning_rate`}
              type="number"
              value={config.learning_rate || 0.0001}
              onChange={(e) => handleInputChange('learning_rate', parseFloat(e.target.value))}
              className="form-input"
              step="0.00001"
              min="0.00001"
              max="0.001"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Learning rate for training (0.0001 recommended)
            </p>
          </div>

          {/* Gradient accumulation steps field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_gradient_accumulation_steps`} className="form-label">
              Gradient Accumulation Steps
            </label>
            <input
              id={`${selectedScript.id}_gradient_accumulation_steps`}
              type="number"
              value={config.gradient_accumulation_steps || 8}
              onChange={(e) => handleInputChange('gradient_accumulation_steps', parseInt(e.target.value))}
              className="form-input"
              min="1"
              max="32"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Number of steps to accumulate gradients (higher values use less memory)
            </p>
          </div>

          {/* Advanced Parameters Section */}
          <div className="mt-6 mb-2">
            <h3 className="text-lg font-medium">Advanced Parameters</h3>
            <p className="text-xs text-secondary-500">Fine-tune the distillation process</p>
          </div>

          {/* Beta parameter field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_beta`} className="form-label">
              Beta (Contrastive Loss Weight)
            </label>
            <input
              id={`${selectedScript.id}_beta`}
              type="number"
              value={config.beta || 0.1}
              onChange={(e) => handleInputChange('beta', parseFloat(e.target.value))}
              className="form-input"
              step="0.01"
              min="0"
              max="1"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Weight for contrastive loss (0.1 recommended)
            </p>
          </div>

          {/* Lambda parameter field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_lambda`} className="form-label">
              Lambda (Intermediate Loss Weight)
            </label>
            <input
              id={`${selectedScript.id}_lambda`}
              type="number"
              value={config.lambda || 0.1}
              onChange={(e) => handleInputChange('lambda', parseFloat(e.target.value))}
              className="form-input"
              step="0.01"
              min="0"
              max="1"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Weight for intermediate feature loss (0.1 recommended)
            </p>
          </div>

          {/* Max sequence length field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_max_seq_length`} className="form-label">
              Max Sequence Length
            </label>
            <input
              id={`${selectedScript.id}_max_seq_length`}
              type="number"
              value={config.max_seq_length || 256}
              onChange={(e) => handleInputChange('max_seq_length', parseInt(e.target.value))}
              className="form-input"
              min="64"
              max="2048"
              step="64"
            />
            <p className="text-xs text-secondary-500 mt-1">
              Maximum sequence length for training (longer sequences use more memory)
            </p>
          </div>

          {/* Hardware Options Section */}
          <div className="mt-6 mb-2">
            <h3 className="text-lg font-medium">Hardware Options</h3>
            <p className="text-xs text-secondary-500">Configure hardware acceleration</p>
          </div>

          {/* Use GPU checkbox */}
          <div className="form-group">
            <div className="flex items-center">
              <input
                id={`${selectedScript.id}_use_gpu`}
                type="checkbox"
                checked={config.use_gpu || false}
                onChange={(e) => handleInputChange('use_gpu', e.target.checked)}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor={`${selectedScript.id}_use_gpu`} className="ml-2 block text-sm font-medium text-secondary-700">
                Use GPU Acceleration
              </label>
            </div>
            <p className="text-xs text-secondary-500 mt-1 ml-6">
              Enable GPU acceleration for faster training (if available)
            </p>
          </div>

          {/* Mixed precision checkbox */}
          <div className="form-group">
            <div className="flex items-center">
              <input
                id={`${selectedScript.id}_mixed_precision`}
                type="checkbox"
                checked={config.mixed_precision || true}
                onChange={(e) => handleInputChange('mixed_precision', e.target.checked)}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor={`${selectedScript.id}_mixed_precision`} className="ml-2 block text-sm font-medium text-secondary-700">
                Use Mixed Precision (BF16/FP16)
              </label>
            </div>
            <p className="text-xs text-secondary-500 mt-1 ml-6">
              Enable mixed precision training for better performance (requires GPU)
            </p>
          </div>

          {/* 4-bit quantization checkbox */}
          <div className="form-group">
            <div className="flex items-center">
              <input
                id={`${selectedScript.id}_use_4bit`}
                type="checkbox"
                checked={config.use_4bit || true}
                onChange={(e) => handleInputChange('use_4bit', e.target.checked)}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor={`${selectedScript.id}_use_4bit`} className="ml-2 block text-sm font-medium text-secondary-700">
                Use 4-bit Quantization
              </label>
            </div>
            <p className="text-xs text-secondary-500 mt-1 ml-6">
              Enable 4-bit quantization to reduce memory usage (recommended)
            </p>
          </div>
        </div>
      );
    }

    // Special handling for student self-study script
    if (selectedScript.id === 'student_self_study') {
      return (
        <div className="space-y-4">
          {/* PDF folder field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_pdf_folder`} className="form-label">
              PDF Folder
            </label>
            <div className="flex space-x-2">
              <input
                id={`${selectedScript.id}_pdf_folder`}
                type="text"
                value={config.pdf_folder || ''}
                onChange={(e) => handleInputChange('pdf_folder', e.target.value)}
                className="form-input flex-grow"
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.webkitdirectory = true;
                  input.directory = true;

                  input.onchange = (e) => {
                    if (e.target.files.length > 0) {
                      const file = e.target.files[0];
                      const path = file.webkitRelativePath.split('/')[0];
                      handleInputChange('pdf_folder', path);
                    }
                  };

                  input.click();
                }}
              >
                Search
              </button>
            </div>
            <p className="text-xs text-secondary-500 mt-1">
              Path to folder containing domain-specific PDF files
            </p>
          </div>

          {/* Model path field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_model_path`} className="form-label">
              Model Path
            </label>
            <div className="flex space-x-2">
              <input
                id={`${selectedScript.id}_model_path`}
                type="text"
                value={config.model_path || ''}
                onChange={(e) => handleInputChange('model_path', e.target.value)}
                className="form-input flex-grow"
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';

                  input.onchange = (e) => {
                    if (e.target.files.length > 0) {
                      const file = e.target.files[0];
                      handleInputChange('model_path', file.name);
                    }
                  };

                  input.click();
                }}
              >
                Search
              </button>
            </div>
            <p className="text-xs text-secondary-500 mt-1">
              Path to the distilled model checkpoint
            </p>
          </div>

          {/* Output directory field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_output_dir`} className="form-label">
              Output Directory
            </label>
            <div className="flex space-x-2">
              <input
                id={`${selectedScript.id}_output_dir`}
                type="text"
                value={config.output_dir || ''}
                onChange={(e) => handleInputChange('output_dir', e.target.value)}
                className="form-input flex-grow"
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.webkitdirectory = true;
                  input.directory = true;

                  input.onchange = (e) => {
                    if (e.target.files.length > 0) {
                      const file = e.target.files[0];
                      const path = file.webkitRelativePath.split('/')[0];
                      handleInputChange('output_dir', path);
                    }
                  };

                  input.click();
                }}
              >
                Search
              </button>
            </div>
          </div>

          {/* Number of questions field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_num_questions`} className="form-label">
              Questions Per Sentence
            </label>
            <input
              id={`${selectedScript.id}_num_questions`}
              type="number"
              value={config.num_questions || 3}
              onChange={(e) => handleInputChange('num_questions', parseInt(e.target.value))}
              className="form-input"
              min="1"
              max="10"
            />
          </div>

          {/* Use hierarchical context checkbox */}
          <div className="form-group">
            <div className="flex items-center">
              <input
                id={`${selectedScript.id}_use_hierarchical_context`}
                type="checkbox"
                checked={config.use_hierarchical_context || false}
                onChange={(e) => handleInputChange('use_hierarchical_context', e.target.checked)}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor={`${selectedScript.id}_use_hierarchical_context`} className="ml-2 block text-sm font-medium text-secondary-700">
                Use Hierarchical Context
              </label>
            </div>
            <p className="text-xs text-secondary-500 mt-1 ml-6">
              Enable hierarchical context encoding with paragraph summaries
            </p>
          </div>

          {/* Include reasoning checkbox */}
          <div className="form-group">
            <div className="flex items-center">
              <input
                id={`${selectedScript.id}_include_reasoning`}
                type="checkbox"
                checked={config.include_reasoning || false}
                onChange={(e) => handleInputChange('include_reasoning', e.target.checked)}
                className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
              />
              <label htmlFor={`${selectedScript.id}_include_reasoning`} className="ml-2 block text-sm font-medium text-secondary-700">
                Include Chain-of-Thought Reasoning
              </label>
            </div>
            <p className="text-xs text-secondary-500 mt-1 ml-6">
              Include step-by-step reasoning in prompts
            </p>
          </div>
        </div>
      );
    }

    // Special handling for model merging script
    if (selectedScript.id === 'merge_model') {
      return (
        <div className="space-y-4">
          {/* Adapter path field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_adapter_path`} className="form-label">
              Adapter Path
            </label>
            <div className="flex space-x-2">
              <input
                id={`${selectedScript.id}_adapter_path`}
                type="text"
                value={config.adapter_path || ''}
                onChange={(e) => handleInputChange('adapter_path', e.target.value)}
                className="form-input flex-grow"
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.webkitdirectory = true;
                  input.directory = true;

                  input.onchange = (e) => {
                    if (e.target.files.length > 0) {
                      const file = e.target.files[0];
                      const path = file.webkitRelativePath.split('/')[0];
                      handleInputChange('adapter_path', path);
                    }
                  };

                  input.click();
                }}
              >
                Search
              </button>
            </div>
            <p className="text-xs text-secondary-500 mt-1">
              Path to the trained LoRA adapters (usually the best checkpoint from distillation)
            </p>
          </div>

          {/* Output path field */}
          <div className="form-group">
            <label htmlFor={`${selectedScript.id}_output_path`} className="form-label">
              Output Path
            </label>
            <div className="flex space-x-2">
              <input
                id={`${selectedScript.id}_output_path`}
                type="text"
                value={config.output_path || ''}
                onChange={(e) => handleInputChange('output_path', e.target.value)}
                className="form-input flex-grow"
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  const input = document.createElement('input');
                  input.type = 'file';
                  input.webkitdirectory = true;
                  input.directory = true;

                  input.onchange = (e) => {
                    if (e.target.files.length > 0) {
                      const file = e.target.files[0];
                      const path = file.webkitRelativePath.split('/')[0];
                      handleInputChange('output_path', path);
                    }
                  };

                  input.click();
                }}
              >
                Search
              </button>
            </div>
            <p className="text-xs text-secondary-500 mt-1">
              Path to save the merged model
            </p>
          </div>

          {/* View Logs button */}
          <div className="form-group mt-6">
            <div className="bg-secondary-100 p-4 rounded-lg">
              <h3 className="text-lg font-medium mb-2">Script Output</h3>
              <p className="text-sm text-secondary-600 mb-4">
                The Model Merging script will output detailed logs about the merging process. You can view these logs in real-time to monitor the progress.
              </p>
              <button
                type="button"
                className="btn btn-primary flex items-center"
                onClick={() => navigate(`/logs/${selectedScript.id}`)}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M3 5a2 2 0 012-2h10a2 2 0 012 2v10a2 2 0 01-2 2H5a2 2 0 01-2-2V5zm11 1H6v8l4-2 4 2V6z" clipRule="evenodd" />
                </svg>
                View Logs
              </button>
            </div>
          </div>
        </div>
      );
    }

    // Default rendering for other scripts
    return (
      <div className="space-y-4">
        {Object.entries(config).map(([key, value]) => {
          // Skip rendering advanced fields if not in advanced mode
          const isAdvanced = key.startsWith('advanced_');
          if (isAdvanced && !advancedMode) return null;

          const fieldId = `${selectedScript.id}_${key}`;
          const fieldLabel = key.replace(/_/g, ' ').replace(/^advanced /, '');

          // Render different input types based on value type
          if (typeof value === 'boolean') {
            return (
              <div key={key} className="form-group">
                <div className="flex items-center">
                  <input
                    id={fieldId}
                    type="checkbox"
                    checked={value}
                    onChange={(e) => handleInputChange(key, e.target.checked)}
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                  />
                  <label htmlFor={fieldId} className="ml-2 block text-sm font-medium text-secondary-700">
                    {fieldLabel}
                  </label>
                </div>
              </div>
            );
          } else if (typeof value === 'number') {
            return (
              <div key={key} className="form-group">
                <label htmlFor={fieldId} className="form-label">
                  {fieldLabel}
                </label>
                <input
                  id={fieldId}
                  type="number"
                  value={value}
                  onChange={(e) => handleInputChange(key, parseFloat(e.target.value))}
                  className="form-input"
                />
              </div>
            );
          } else {
            return (
              <div key={key} className="form-group">
                <label htmlFor={fieldId} className="form-label">
                  {fieldLabel}
                </label>
                <input
                  id={fieldId}
                  type="text"
                  value={value}
                  onChange={(e) => handleInputChange(key, e.target.value)}
                  className={`form-input ${validationErrors[key] ? 'border-error-500' : ''}`}
                />
                {validationErrors[key] && (
                  <p className="text-error-500 text-sm mt-1">{validationErrors[key]}</p>
                )}
              </div>
            );
          }
        })}
      </div>
    );
  };

  // Handle model query
  const handleModelQuery = async () => {
    if (!modelPrompt.trim()) {
      setQueryError('Please enter a prompt');
      return;
    }

    if (!configs.evaluation?.model_path) {
      setQueryError('Please specify a model path');
      return;
    }

    setQueryLoading(true);
    setQueryError(null);

    try {
      const response = await modelService.queryModel(
        configs.evaluation.model_path,
        modelPrompt
      );
      setModelResponse(response.response);
    } catch (error) {
      console.error('Error querying model:', error);
      setQueryError(error.message || 'Failed to query model');
    } finally {
      setQueryLoading(false);
    }
  };

  // Check if the script is currently running
  const isScriptRunning = () => {
    return systemStatus &&
           systemStatus.active_scripts &&
           systemStatus.active_scripts.includes(selectedScript?.id);
  };

  // Render model query interface
  const renderModelQueryInterface = () => {
    if (selectedScript?.id !== 'evaluation') return null;

    return (
      <div className="mt-8 border-t border-secondary-200 pt-6">
        <h3 className="text-lg font-medium mb-4">Query Model</h3>
        <p className="text-secondary-600 mb-4">
          Test the distilled model by sending a prompt and viewing the response.
        </p>

        {queryError && (
          <div className="bg-error-50 border border-error-200 text-error-700 px-4 py-3 rounded mb-4 flex items-start">
            <AlertCircle size={20} className="mr-2 mt-0.5 flex-shrink-0" />
            <p>{queryError}</p>
          </div>
        )}

        <div className="space-y-4">
          <div className="form-group">
            <label htmlFor="model-prompt" className="form-label">Prompt</label>
            <textarea
              id="model-prompt"
              value={modelPrompt}
              onChange={(e) => setModelPrompt(e.target.value)}
              className="form-input h-32"
              placeholder="Enter your prompt here..."
            />
          </div>

          <div className="flex justify-end">
            <button
              type="button"
              className="btn btn-primary flex items-center"
              onClick={handleModelQuery}
              disabled={queryLoading}
            >
              {queryLoading ? (
                <span className="animate-pulse">Processing...</span>
              ) : (
                <>
                  <Send size={16} className="mr-2" />
                  Send Query
                </>
              )}
            </button>
          </div>

          {modelResponse && (
            <div className="mt-4">
              <label className="form-label">Model Response</label>
              <div className="bg-secondary-50 border border-secondary-200 rounded-md p-4 whitespace-pre-wrap">
                {modelResponse}
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex flex-col h-full space-y-4">
      {/* Pipeline Navigation */}
      <PipelineNavigation />

      <div className="flex flex-col md:flex-row h-full space-y-4 md:space-y-0 md:space-x-4">
      {/* Script Selection Sidebar */}
      <div className="w-full md:w-64 bg-white rounded-lg shadow-card p-4">
        <h3 className="text-lg font-medium mb-4">Pipeline Steps</h3>
        <ul className="space-y-2">
          {scripts
            .filter(script => !script.hidden && script.id !== 'manual_extractor' && script.id !== 'data_enrichment')
            .sort((a, b) => a.step - b.step)
            .map((script, index) => (
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
                    <span className="text-primary-700 text-xs font-semibold">{index + 1}</span>
                  </div>
                  <span className="text-sm">{script.name}</span>
                  {pipelineStatus && pipelineStatus[script.id] === 'completed' && (
                    <span className="ml-auto text-success-600">✓</span>
                  )}
                </button>
              </li>
            ))}
        </ul>
      </div>

      {/* Configuration Form */}
      <div className="flex-1">
        {selectedScript ? (
          <div className="bg-white rounded-lg shadow-card p-6">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-xl font-semibold">{selectedScript.name} Configuration</h2>
              <div className="flex space-x-2">
                <button
                  className="btn btn-secondary flex items-center"
                  onClick={() => setShowSaveDialog(true)}
                >
                  <Save size={16} className="mr-1" />
                  Save
                </button>
                <button
                  className="btn btn-secondary flex items-center"
                  onClick={() => setShowLoadDialog(true)}
                >
                  <Upload size={16} className="mr-1" />
                  Load
                </button>
              </div>
            </div>

            <p className="text-secondary-600 mb-6">{selectedScript.description}</p>

            {error && (
              <div className="bg-error-50 border border-error-200 text-error-700 px-4 py-3 rounded mb-4 flex items-start">
                <AlertCircle size={20} className="mr-2 mt-0.5 flex-shrink-0" />
                <p>{error}</p>
              </div>
            )}

            {executionError && (
              <div className="bg-error-50 border border-error-200 text-error-700 px-4 py-3 rounded mb-4 flex items-start">
                <AlertCircle size={20} className="mr-2 mt-0.5 flex-shrink-0" />
                <p>{executionError}</p>
              </div>
            )}

            <form className="space-y-6">
              {renderFormFields()}

              {/* Advanced Options Toggle */}
              <div className="border-t border-secondary-200 pt-4">
                <button
                  type="button"
                  className="flex items-center text-secondary-700 hover:text-secondary-900"
                  onClick={() => setAdvancedMode(!advancedMode)}
                >
                  {advancedMode ? (
                    <ChevronDown size={20} className="mr-1" />
                  ) : (
                    <ChevronRight size={20} className="mr-1" />
                  )}
                  Advanced Options
                </button>
              </div>

              {/* Execution Button */}
              <div className="flex justify-end pt-4 border-t border-secondary-200">
                <button
                  type="button"
                  className="btn btn-primary flex items-center"
                  onClick={handleExecute}
                  disabled={loading || isScriptRunning()}
                >
                  <Play size={16} className="mr-2" />
                  Execute {selectedScript.name}
                </button>
              </div>

              {/* Model Query Interface (only for evaluation script) */}
              {renderModelQueryInterface()}
            </form>
          </div>
        ) : (
          <div className="bg-white rounded-lg shadow-card p-6 flex items-center justify-center h-full">
            <p className="text-secondary-500">Select a script from the sidebar</p>
          </div>
        )}
      </div>

      {/* Save Configuration Dialog */}
      {showSaveDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Save Configuration</h3>
            <div className="mb-4">
              <label htmlFor="configName" className="form-label">Configuration Name</label>
              <input
                id="configName"
                type="text"
                value={configName}
                onChange={(e) => setConfigName(e.target.value)}
                placeholder={`${selectedScript.id}_config_${Date.now()}`}
                className="form-input"
              />
            </div>
            <div className="flex justify-end space-x-2">
              <button
                className="btn btn-secondary"
                onClick={() => setShowSaveDialog(false)}
              >
                Cancel
              </button>
              <button
                className="btn btn-primary flex items-center"
                onClick={handleSaveConfig}
              >
                <Save size={16} className="mr-1" />
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Load Configuration Dialog */}
      {showLoadDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-lg p-6 w-full max-w-md">
            <h3 className="text-lg font-semibold mb-4">Load Configuration</h3>
            {savedConfigs.length > 0 ? (
              <div className="max-h-96 overflow-y-auto">
                <ul className="divide-y divide-secondary-200">
                  {savedConfigs.map((config, index) => (
                    <li key={index} className="py-3">
                      <button
                        className="w-full text-left flex items-center justify-between hover:bg-secondary-50 p-2 rounded"
                        onClick={() => handleLoadConfig(config)}
                      >
                        <div>
                          <p className="font-medium">{config.name}</p>
                          <p className="text-sm text-secondary-500">
                            {config.scriptId} - {new Date(config.created).toLocaleString()}
                          </p>
                        </div>
                        <Download size={16} className="text-primary-600" />
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <p className="text-center py-4 text-secondary-500">No saved configurations found</p>
            )}
            <div className="flex justify-end mt-4">
              <button
                className="btn btn-secondary"
                onClick={() => setShowLoadDialog(false)}
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
};

export default ScriptConfiguration;
