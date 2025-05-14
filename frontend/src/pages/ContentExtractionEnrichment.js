import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
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
  Server,
  FileText,
  Database,
  Cpu
} from 'lucide-react';
import { useScript } from '../contexts/ScriptContext';
import { useSystem } from '../contexts/SystemContext';
import { scriptService } from '../services/scriptService';
import PipelineNavigation from '../components/PipelineNavigation';
import PipelineSelector from '../components/PipelineSelector';

const ContentExtractionEnrichment = () => {
  const navigate = useNavigate();
  const { scripts, configs, updateConfig, executeScript, loading, error } = useScript();
  const { systemStatus } = useSystem();

  // State for the combined configuration
  const [combinedConfig, setCombinedConfig] = useState({
    // Manual Extractor config
    url: '',
    source_folder: '',
    docker_folder: '/data',
    output_dir: '',
    extract_links: false,

    // Data Enrichment config
    enable_enrichment: true,
    input_file: '',
    output_file: '',
    topic: 'cybersecurity',

    // Advanced options for enrichment
    enable_entity_extraction: true,
    enable_summarization: true,
    enable_keyword_extraction: true,
    use_gpu: true,

    // Code Dataset Integration
    process_code_dataset: true,
    code_dataset: 'Shuu12121/python-codesearch-dataset-open',
    code_dataset_split: 'train',
    code_dataset_max_samples: 5000,

    // CVE Data Integration
    process_cve_data: true,
    cve_data_folder: 'cvelistV5-main',

    // Domain Context
    use_domain_context: false,
    domain_context_file: ''
  });

  // State for validation errors
  const [validationErrors, setValidationErrors] = useState({});

  // State for selected source type (url, source_folder, docker_folder)
  const [selectedSource, setSelectedSource] = useState('url');

  // State for execution status
  const [executionStatus, setExecutionStatus] = useState({
    isRunning: false,
    step: null,
    message: null,
    error: null
  });

  // State for advanced mode
  const [advancedMode, setAdvancedMode] = useState(false);

  // State for Docker containers
  const [dockerContainers, setDockerContainers] = useState([]);
  const [loadingContainers, setLoadingContainers] = useState(false);

  // State for pipeline selector
  const [showPipelineSelector, setShowPipelineSelector] = useState(false);

  // Load Docker containers
  useEffect(() => {
    const fetchDockerContainers = async () => {
      setLoadingContainers(true);
      try {
        const containers = await scriptService.getDockerContainers();
        setDockerContainers(containers);
      } catch (err) {
        console.error('Error fetching Docker containers:', err);
      } finally {
        setLoadingContainers(false);
      }
    };

    fetchDockerContainers();
  }, []);

  // Handle source selection
  const handleSourceSelect = (source) => {
    setSelectedSource(source);

    // Clear validation errors for the other sources
    const newErrors = { ...validationErrors };
    if (source !== 'url') delete newErrors.url;
    if (source !== 'source_folder') delete newErrors.source_folder;
    if (source !== 'docker_folder') delete newErrors.docker_folder;

    setValidationErrors(newErrors);
  };

  // Handle input change
  const handleInputChange = (field, value) => {
    setCombinedConfig(prev => ({
      ...prev,
      [field]: value
    }));

    // Clear validation error for this field
    if (validationErrors[field]) {
      setValidationErrors(prev => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }

    // Auto-update output paths based on input
    if (field === 'output_dir' && value) {
      // Update the input_file and output_file for enrichment
      setCombinedConfig(prev => ({
        ...prev,
        input_file: `${value}/extracted_data.json`,
        output_file: `${value}/enriched_data.json`
      }));
    }
  };

  // Validate the configuration
  const validateConfig = () => {
    const errors = {};

    // Validate based on selected source
    if (selectedSource === 'url') {
      if (!combinedConfig.url) {
        errors.url = 'URL is required';
      } else if (!combinedConfig.url.startsWith('http')) {
        errors.url = 'URL must start with http:// or https://';
      }
    } else if (selectedSource === 'source_folder') {
      if (!combinedConfig.source_folder) {
        errors.source_folder = 'Source folder is required';
      }
    } else if (selectedSource === 'docker_folder') {
      if (!combinedConfig.docker_folder) {
        errors.docker_folder = 'Docker folder is required';
      }
    }

    // Validate output directory
    if (!combinedConfig.output_dir) {
      errors.output_dir = 'Output directory is required';
    }

    // Validate enrichment options if enabled
    if (combinedConfig.enable_enrichment) {
      if (!combinedConfig.output_file) {
        errors.output_file = 'Output file is required for enrichment';
      }
    }

    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  // Handle execution
  const handleExecute = async () => {
    if (!validateConfig()) {
      return;
    }

    setExecutionStatus({
      isRunning: true,
      step: 'extraction',
      message: 'Running content extraction...',
      error: null
    });

    try {
      // Prepare extraction config
      const extractionConfig = {
        url: selectedSource === 'url' ? combinedConfig.url : '',
        source_folder: selectedSource === 'source_folder' ? combinedConfig.source_folder : '',
        docker_folder: selectedSource === 'docker_folder' ? combinedConfig.docker_folder : '',
        output_dir: combinedConfig.output_dir,
        extract_links: combinedConfig.extract_links
      };

      // Prepare combined config for content_extraction_enrichment
      const combinedScriptConfig = {
        url: selectedSource === 'url' ? combinedConfig.url : '',
        source_folder: selectedSource === 'source_folder' ? combinedConfig.source_folder : '',
        docker_folder: selectedSource === 'docker_folder' ? combinedConfig.docker_folder : '',
        output_dir: combinedConfig.output_dir,
        extract_links: combinedConfig.extract_links,
        enable_enrichment: combinedConfig.enable_enrichment,
        input_file: combinedConfig.input_file || `${combinedConfig.output_dir}/extracted_data.json`,
        output_file: combinedConfig.output_file || `${combinedConfig.output_dir}/enriched_data.json`,
        topic: combinedConfig.topic,
        enable_entity_extraction: combinedConfig.enable_entity_extraction,
        enable_summarization: combinedConfig.enable_summarization,
        enable_keyword_extraction: combinedConfig.enable_keyword_extraction,
        use_gpu: combinedConfig.use_gpu,

        // Code Dataset Integration parameters
        process_code_dataset: combinedConfig.process_code_dataset,
        code_dataset: combinedConfig.code_dataset,
        code_dataset_split: combinedConfig.code_dataset_split,
        code_dataset_max_samples: combinedConfig.code_dataset_max_samples,

        // CVE Data Integration parameters
        process_cve_data: combinedConfig.process_cve_data,
        cve_data_folder: combinedConfig.cve_data_folder,

        // Domain Context parameters
        use_domain_context: combinedConfig.use_domain_context,
        domain_context_file: combinedConfig.domain_context_file
      };

      // Execute the combined script
      console.log('Executing content_extraction_enrichment script with config:', combinedScriptConfig);
      const success = await executeScript('content_extraction_enrichment', combinedScriptConfig);

      if (!success) {
        throw new Error('Content extraction and enrichment failed');
      }

      // All steps completed successfully
      setExecutionStatus({
        isRunning: false,
        step: 'completed',
        message: 'All steps completed successfully. Choose which pipeline to use next.',
        error: null
      });

      // Show the pipeline selector instead of navigating to logs
      console.log('Showing pipeline selector');
      setShowPipelineSelector(true);
    } catch (err) {
      console.error('Execution error:', err);
      setExecutionStatus({
        isRunning: false,
        step: 'error',
        message: null,
        error: err.message || 'An error occurred during execution'
      });
    }
  };

  // Handle sending to Docker
  const handleSendToDocker = () => {
    if (!combinedConfig.source_folder) {
      setValidationErrors(prev => ({
        ...prev,
        source_folder: 'Source folder is required'
      }));
      return;
    }

    // Update Docker folder with the source folder path
    handleInputChange('docker_folder', '/data');

    // Switch to Docker folder source
    handleSourceSelect('docker_folder');
  };

  return (
    <div className="p-6">
      {/* Pipeline Navigation */}
      <PipelineNavigation />

      {/* Pipeline Selector Modal */}
      {showPipelineSelector && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <PipelineSelector
            onClose={() => {
              setShowPipelineSelector(false);
              navigate('/logs/content_extraction_enrichment');
            }}
            outputDir={combinedConfig.output_dir}
          />
        </div>
      )}

      <div className="bg-white rounded-lg shadow-card p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold">Content Extraction & Enrichment</h2>
          <div className="flex space-x-2">
            <button
              className="btn btn-primary flex items-center"
              onClick={handleExecute}
              disabled={executionStatus.isRunning}
            >
              {executionStatus.isRunning ? (
                <>
                  <div className="animate-spin mr-2 h-4 w-4 border-2 border-white rounded-full border-t-transparent"></div>
                  Running...
                </>
              ) : (
                <>
                  <Play size={16} className="mr-1" />
                  Execute
                </>
              )}
            </button>
          </div>
        </div>

        {/* Execution status messages */}
        {executionStatus.message && (
          <div className={`mb-4 p-3 rounded ${
            executionStatus.step === 'completed'
              ? 'bg-success-100 text-success-800'
              : 'bg-primary-100 text-primary-800'
          }`}>
            <p className="flex items-center">
              <Info size={16} className="mr-2" />
              {executionStatus.message}
            </p>
          </div>
        )}

        {executionStatus.error && (
          <div className="mb-4 p-3 rounded bg-error-100 text-error-800">
            <p className="flex items-center">
              <AlertCircle size={16} className="mr-2" />
              {executionStatus.error}
            </p>
          </div>
        )}

        {/* Content Extraction Section */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-4 flex items-center">
            <FileText size={18} className="mr-2 text-primary-600" />
            Content Extraction
          </h3>

          <div className="space-y-4">
            {/* URL input with radio button */}
            <div className="form-group">
              <div className="flex items-center mb-2">
                <input
                  id="use_url"
                  type="radio"
                  checked={selectedSource === 'url'}
                  onChange={() => handleSourceSelect('url')}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300"
                />
                <label htmlFor="use_url" className="ml-2 block text-sm font-medium text-secondary-700">
                  Extract from URL
                </label>
              </div>
              <input
                id="url"
                type="text"
                value={combinedConfig.url}
                onChange={(e) => handleInputChange('url', e.target.value)}
                className={`form-input ${selectedSource !== 'url' ? 'opacity-50' : ''} ${
                  validationErrors.url ? 'border-error-500' : ''
                }`}
                disabled={selectedSource !== 'url'}
                placeholder="https://example.com/docs"
              />
              {validationErrors.url && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.url}</p>
              )}
            </div>

            {/* Source folder with radio button */}
            <div className="form-group">
              <div className="flex items-center mb-2">
                <input
                  id="use_source_folder"
                  type="radio"
                  checked={selectedSource === 'source_folder'}
                  onChange={() => handleSourceSelect('source_folder')}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300"
                />
                <label htmlFor="use_source_folder" className="ml-2 block text-sm font-medium text-secondary-700">
                  Extract from Local Folder
                </label>
              </div>
              <div className="flex space-x-2">
                <input
                  id="source_folder"
                  type="text"
                  value={combinedConfig.source_folder}
                  onChange={(e) => handleInputChange('source_folder', e.target.value)}
                  className={`form-input flex-grow ${selectedSource !== 'source_folder' ? 'opacity-50' : ''} ${
                    validationErrors.source_folder ? 'border-error-500' : ''
                  }`}
                  disabled={selectedSource !== 'source_folder'}
                  placeholder="C:\\Path\\To\\Folder"
                />
                <button
                  type="button"
                  className="btn btn-secondary"
                  disabled={selectedSource !== 'source_folder'}
                  onClick={() => {
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.webkitdirectory = true;
                    input.directory = true;

                    input.onchange = (e) => {
                      if (e.target.files.length > 0) {
                        const file = e.target.files[0];
                        const path = file.webkitRelativePath.split('/')[0];
                        handleInputChange('source_folder', path);
                      }
                    };

                    input.click();
                  }}
                >
                  Browse
                </button>
                {selectedSource === 'source_folder' && (
                  <button
                    type="button"
                    className="btn btn-primary flex items-center"
                    onClick={handleSendToDocker}
                  >
                    <Send size={16} className="mr-1" />
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

            {/* Docker folder with radio button */}
            <div className="form-group">
              <div className="flex items-center mb-2">
                <input
                  id="use_docker_folder"
                  type="radio"
                  checked={selectedSource === 'docker_folder'}
                  onChange={() => handleSourceSelect('docker_folder')}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300"
                />
                <label htmlFor="use_docker_folder" className="ml-2 block text-sm font-medium text-secondary-700">
                  Extract from Docker Folder
                </label>
              </div>
              <input
                id="docker_folder"
                type="text"
                value={combinedConfig.docker_folder}
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
              <label htmlFor="output_dir" className="form-label">
                Output Directory
              </label>
              <input
                id="output_dir"
                type="text"
                value={combinedConfig.output_dir}
                onChange={(e) => handleInputChange('output_dir', e.target.value)}
                className={`form-input ${validationErrors.output_dir ? 'border-error-500' : ''}`}
                placeholder="Output"
              />
              {validationErrors.output_dir && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.output_dir}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                <strong>Base directory for the extraction and enrichment steps.</strong> The enrichment step will automatically use the extracted data from this directory.
                <span className="text-primary-600">Important for pipeline: </span>
                This directory feeds into subsequent pipeline steps (Teacher Pair Generation).
              </p>
              <div className="mt-2 p-2 bg-info-50 rounded text-xs text-info-700">
                <strong>Note:</strong> The Output Directory is the base directory for all extraction and enrichment operations.
                Files will be saved here and used by subsequent pipeline steps. Make sure this directory is accessible to all pipeline components.
              </div>
            </div>

            {/* Extract links checkbox */}
            <div className="form-group">
              <div className="flex items-center">
                <input
                  id="extract_links"
                  type="checkbox"
                  checked={combinedConfig.extract_links}
                  onChange={(e) => handleInputChange('extract_links', e.target.checked)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                />
                <label htmlFor="extract_links" className="ml-2 block text-sm font-medium text-secondary-700">
                  Extract Links
                </label>
              </div>
              <p className="text-xs text-secondary-500 mt-1 ml-6">
                Extract and process links found in the content
              </p>
            </div>
          </div>
        </div>

        {/* Data Enrichment Section */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-4 flex items-center">
            <Database size={18} className="mr-2 text-primary-600" />
            Data Enrichment
          </h3>

          <div className="space-y-4">
            {/* Enable enrichment checkbox */}
            <div className="form-group">
              <div className="flex items-center">
                <input
                  id="enable_enrichment"
                  type="checkbox"
                  checked={combinedConfig.enable_enrichment}
                  onChange={(e) => handleInputChange('enable_enrichment', e.target.checked)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                />
                <label htmlFor="enable_enrichment" className="ml-2 block text-sm font-medium text-secondary-700">
                  Enable Data Enrichment
                </label>
              </div>
              <p className="text-xs text-secondary-500 mt-1 ml-6">
                Process extracted content to add summaries, entities, and keywords
              </p>
            </div>

            {combinedConfig.enable_enrichment && (
              <>
                {/* Input file field */}
                <div className="form-group">
                  <label htmlFor="input_file" className="form-label">
                    Input File
                  </label>
                  <input
                    id="input_file"
                    type="text"
                    value={combinedConfig.input_file}
                    onChange={(e) => handleInputChange('input_file', e.target.value)}
                    className="form-input"
                    placeholder="Output/extracted_data.json"
                  />
                  <p className="text-xs text-secondary-500 mt-1">
                    Path to the extracted data JSON file (auto-filled from extraction step).
                    This is automatically set based on the Output Directory above.
                  </p>
                </div>

                {/* Output file field */}
                <div className="form-group">
                  <label htmlFor="output_file" className="form-label">
                    Output File
                  </label>
                  <input
                    id="output_file"
                    type="text"
                    value={combinedConfig.output_file}
                    onChange={(e) => handleInputChange('output_file', e.target.value)}
                    className={`form-input ${validationErrors.output_file ? 'border-error-500' : ''}`}
                    placeholder="Output/enriched_data.json"
                  />
                  {validationErrors.output_file && (
                    <p className="text-error-500 text-sm mt-1">{validationErrors.output_file}</p>
                  )}
                  <p className="text-xs text-secondary-500 mt-1">
                    Path where the enriched data will be saved. This file will be used as input for the Teacher Pair Generation step in the pipeline.
                  </p>
                </div>

                {/* Topic field */}
                <div className="form-group">
                  <label htmlFor="topic" className="form-label">
                    Topic
                  </label>
                  <input
                    id="topic"
                    type="text"
                    value={combinedConfig.topic}
                    onChange={(e) => handleInputChange('topic', e.target.value)}
                    className="form-input"
                    placeholder="cybersecurity"
                  />
                  <p className="text-xs text-secondary-500 mt-1">
                    The main topic for data enrichment (e.g., cybersecurity, network scanning, vulnerability assessment)
                  </p>
                </div>

                {/* Domain Context File */}
                <div className="form-group mt-4 p-3 bg-secondary-50 rounded-md">
                  <div className="flex items-center mb-3">
                    <input
                      id="use_domain_context"
                      type="checkbox"
                      checked={combinedConfig.use_domain_context}
                      onChange={(e) => handleInputChange('use_domain_context', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                    />
                    <label htmlFor="use_domain_context" className="ml-2 block text-sm font-medium text-secondary-700">
                      Use Domain Context File
                    </label>
                  </div>
                  <p className="text-xs text-secondary-500 mb-3">
                    Use a domain-specific context file to improve enrichment quality
                  </p>

                  {combinedConfig.use_domain_context && (
                    <div className="space-y-3 pl-6 border-l-2 border-primary-100">
                      <div className="form-group">
                        <label htmlFor="domain_context_file" className="form-label text-sm">
                          Domain Context File
                        </label>
                        <input
                          id="domain_context_file"
                          type="text"
                          value={combinedConfig.domain_context_file}
                          onChange={(e) => handleInputChange('domain_context_file', e.target.value)}
                          className="form-input"
                          placeholder="domain_context.json"
                        />
                        <p className="text-xs text-secondary-500 mt-1">
                          Path to a JSON file containing domain-specific context information
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* CVE Data Integration */}
                <div className="form-group mt-4 p-3 bg-secondary-50 rounded-md">
                  <div className="flex items-center mb-3">
                    <input
                      id="process_cve_data"
                      type="checkbox"
                      checked={combinedConfig.process_cve_data}
                      onChange={(e) => handleInputChange('process_cve_data', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                    />
                    <label htmlFor="process_cve_data" className="ml-2 block text-sm font-medium text-secondary-700">
                      Process CVE Data
                    </label>
                  </div>
                  <p className="text-xs text-secondary-500 mb-3">
                    Process Common Vulnerabilities and Exposures (CVE) data
                  </p>

                  {combinedConfig.process_cve_data && (
                    <div className="space-y-3 pl-6 border-l-2 border-primary-100">
                      <div className="form-group">
                        <label htmlFor="cve_data_folder" className="form-label text-sm">
                          CVE Data Folder
                        </label>
                        <input
                          id="cve_data_folder"
                          type="text"
                          value={combinedConfig.cve_data_folder}
                          onChange={(e) => handleInputChange('cve_data_folder', e.target.value)}
                          className="form-input"
                          placeholder="cvelistV5-main"
                        />
                        <p className="text-xs text-secondary-500 mt-1">
                          Folder containing CVE JSON files (typically cvelistV5-main)
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Code Dataset Integration */}
                <div className="form-group mt-4 p-3 bg-secondary-50 rounded-md">
                  <h4 className="text-md font-medium mb-3">Code Dataset Integration</h4>

                  <div className="flex items-center mb-3">
                    <input
                      id="process_code_dataset"
                      type="checkbox"
                      checked={combinedConfig.process_code_dataset}
                      onChange={(e) => handleInputChange('process_code_dataset', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                    />
                    <label htmlFor="process_code_dataset" className="ml-2 block text-sm font-medium text-secondary-700">
                      Process Code Dataset
                    </label>
                  </div>
                  <p className="text-xs text-secondary-500 mb-3">
                    Automatically download and process code datasets from Hugging Face
                  </p>

                  {combinedConfig.process_code_dataset && (
                    <div className="space-y-3 pl-6 border-l-2 border-primary-100">
                      {/* Dataset name */}
                      <div className="form-group">
                        <label htmlFor="code_dataset" className="form-label text-sm">
                          Dataset Name
                        </label>
                        <input
                          id="code_dataset"
                          type="text"
                          value={combinedConfig.code_dataset}
                          onChange={(e) => handleInputChange('code_dataset', e.target.value)}
                          className="form-input"
                          placeholder="Shuu12121/python-codesearch-dataset-open"
                        />
                        <p className="text-xs text-secondary-500 mt-1">
                          Hugging Face dataset identifier (e.g., Shuu12121/python-codesearch-dataset-open)
                        </p>
                      </div>

                      {/* Dataset split */}
                      <div className="form-group">
                        <label htmlFor="code_dataset_split" className="form-label text-sm">
                          Dataset Split
                        </label>
                        <select
                          id="code_dataset_split"
                          value={combinedConfig.code_dataset_split}
                          onChange={(e) => handleInputChange('code_dataset_split', e.target.value)}
                          className="form-select"
                        >
                          <option value="train">train</option>
                          <option value="validation">validation</option>
                          <option value="test">test</option>
                        </select>
                        <p className="text-xs text-secondary-500 mt-1">
                          Which split of the dataset to use
                        </p>
                      </div>

                      {/* Max samples */}
                      <div className="form-group">
                        <label htmlFor="code_dataset_max_samples" className="form-label text-sm">
                          Max Samples
                        </label>
                        <input
                          id="code_dataset_max_samples"
                          type="number"
                          value={combinedConfig.code_dataset_max_samples}
                          onChange={(e) => handleInputChange('code_dataset_max_samples', parseInt(e.target.value) || 1000)}
                          className="form-input"
                          min="100"
                          max="50000"
                        />
                        <p className="text-xs text-secondary-500 mt-1">
                          Maximum number of samples to process (100-50000)
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                {/* Advanced options toggle */}
                <div className="form-group">
                  <button
                    type="button"
                    className="flex items-center text-sm font-medium text-primary-600 hover:text-primary-800"
                    onClick={() => setAdvancedMode(!advancedMode)}
                  >
                    {advancedMode ? (
                      <ChevronDown size={16} className="mr-1" />
                    ) : (
                      <ChevronRight size={16} className="mr-1" />
                    )}
                    Advanced Options
                  </button>

                  {advancedMode && (
                    <div className="mt-3 pl-6 space-y-3 border-l-2 border-primary-100">
                      {/* Entity extraction checkbox */}
                      <div className="form-group">
                        <div className="flex items-center">
                          <input
                            id="enable_entity_extraction"
                            type="checkbox"
                            checked={combinedConfig.enable_entity_extraction}
                            onChange={(e) => handleInputChange('enable_entity_extraction', e.target.checked)}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                          />
                          <label htmlFor="enable_entity_extraction" className="ml-2 block text-sm font-medium text-secondary-700">
                            Enable Entity Extraction
                          </label>
                        </div>
                        <p className="text-xs text-secondary-500 mt-1 ml-6">
                          Extract named entities (people, organizations, locations, etc.)
                        </p>
                      </div>

                      {/* Summarization checkbox */}
                      <div className="form-group">
                        <div className="flex items-center">
                          <input
                            id="enable_summarization"
                            type="checkbox"
                            checked={combinedConfig.enable_summarization}
                            onChange={(e) => handleInputChange('enable_summarization', e.target.checked)}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                          />
                          <label htmlFor="enable_summarization" className="ml-2 block text-sm font-medium text-secondary-700">
                            Enable Summarization
                          </label>
                        </div>
                        <p className="text-xs text-secondary-500 mt-1 ml-6">
                          Generate concise summaries of the content
                        </p>
                      </div>

                      {/* Keyword extraction checkbox */}
                      <div className="form-group">
                        <div className="flex items-center">
                          <input
                            id="enable_keyword_extraction"
                            type="checkbox"
                            checked={combinedConfig.enable_keyword_extraction}
                            onChange={(e) => handleInputChange('enable_keyword_extraction', e.target.checked)}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                          />
                          <label htmlFor="enable_keyword_extraction" className="ml-2 block text-sm font-medium text-secondary-700">
                            Enable Keyword Extraction
                          </label>
                        </div>
                        <p className="text-xs text-secondary-500 mt-1 ml-6">
                          Extract important keywords from the content
                        </p>
                      </div>

                      {/* GPU acceleration checkbox */}
                      <div className="form-group">
                        <div className="flex items-center">
                          <input
                            id="use_gpu"
                            type="checkbox"
                            checked={combinedConfig.use_gpu}
                            onChange={(e) => handleInputChange('use_gpu', e.target.checked)}
                            className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                          />
                          <label htmlFor="use_gpu" className="ml-2 block text-sm font-medium text-secondary-700 flex items-center">
                            Use GPU Acceleration
                            <Cpu size={16} className="ml-2 text-secondary-500" />
                          </label>
                        </div>
                        <p className="text-xs text-secondary-500 mt-1 ml-6">
                          Use GPU for faster processing (if available)
                        </p>
                      </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        </div>

        {/* System Status Section */}
        <div className="mt-8 border-t pt-4">
          <h3 className="text-sm font-medium text-secondary-700 mb-2">System Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-secondary-50 p-3 rounded">
              <p className="text-xs text-secondary-600">CPU Usage</p>
              <div className="flex items-center mt-1">
                <div className="w-full bg-secondary-200 rounded-full h-2">
                  <div
                    className="bg-primary-600 h-2 rounded-full"
                    style={{ width: `${systemStatus?.cpu_percent || 0}%` }}
                  ></div>
                </div>
                <span className="ml-2 text-xs font-medium">{systemStatus?.cpu_percent || 0}%</span>
              </div>
            </div>

            <div className="bg-secondary-50 p-3 rounded">
              <p className="text-xs text-secondary-600">Memory Usage</p>
              <div className="flex items-center mt-1">
                <div className="w-full bg-secondary-200 rounded-full h-2">
                  <div
                    className="bg-primary-600 h-2 rounded-full"
                    style={{ width: `${systemStatus?.memory_percent || 0}%` }}
                  ></div>
                </div>
                <span className="ml-2 text-xs font-medium">{systemStatus?.memory_percent || 0}%</span>
              </div>
            </div>

            <div className="bg-secondary-50 p-3 rounded">
              <p className="text-xs text-secondary-600">GPU Status</p>
              {systemStatus?.gpu_info ? (
                <div className="flex items-center mt-1">
                  <div className="w-full bg-secondary-200 rounded-full h-2">
                    <div
                      className="bg-primary-600 h-2 rounded-full"
                      style={{ width: `${systemStatus.gpu_info[0]?.memory_percent || 0}%` }}
                    ></div>
                  </div>
                  <span className="ml-2 text-xs font-medium">{systemStatus.gpu_info[0]?.memory_percent.toFixed(1) || 0}%</span>
                </div>
              ) : (
                <p className="text-xs text-secondary-500 mt-1">No GPU detected</p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ContentExtractionEnrichment;
