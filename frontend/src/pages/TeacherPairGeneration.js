import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Play,
  AlertCircle,
  Info,
  Server,
  Database,
  FileText
} from 'lucide-react';
import { useScript } from '../contexts/ScriptContext';
import { useSystem } from '../contexts/SystemContext';
import { scriptService } from '../services/scriptService';
import PipelineNavigation from '../components/PipelineNavigation';
import ModelAvailabilityStatus from '../components/ModelAvailabilityStatus';

const TeacherPairGeneration = () => {
  const navigate = useNavigate();
  const { scripts, configs, updateConfig, executeScript, loading, error } = useScript();
  const { systemStatus, dockerContainers } = useSystem();

  // State for configuration
  const [config, setConfig] = useState({
    input_file: '',
    output_file: '',
    teacher_model: ''
  });

  // State for execution status
  const [executionStatus, setExecutionStatus] = useState({
    isRunning: false,
    step: null,
    message: null,
    error: null
  });

  // State for validation errors
  const [validationErrors, setValidationErrors] = useState({});

  // State for advanced mode
  const [advancedMode, setAdvancedMode] = useState(false);

  // State for available teacher models
  const [availableModels, setAvailableModels] = useState([]);

  // Load Docker containers and set available models
  useEffect(() => {
    const fetchDockerContainers = async () => {
      try {
        const containers = await scriptService.getDockerContainers();

        // Filter for teacher model containers (typically containing 'phi' or 'teacher' in the name)
        const teacherModels = containers.filter(container =>
          container.status === 'running' &&
          (container.name.toLowerCase().includes('phi') ||
           container.name.toLowerCase().includes('teacher') ||
           container.type?.toLowerCase().includes('teacher'))
        );

        setAvailableModels(teacherModels);

        // If we have models and no model is selected yet, select the first one
        if (teacherModels.length > 0 && !config.teacher_model) {
          setConfig(prev => ({
            ...prev,
            teacher_model: teacherModels[0].name
          }));
        }
      } catch (err) {
        console.error('Error fetching Docker containers:', err);
      }
    };

    fetchDockerContainers();

    // Also load saved config if available
    const savedConfig = configs['teacher_pair_generation'];
    if (savedConfig) {
      setConfig(prev => ({
        ...prev,
        ...savedConfig
      }));
    }
  }, [configs]);

  // Handle input change
  const handleInputChange = (field, value) => {
    setConfig(prev => ({
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
  };

  // Validate configuration
  const validateConfig = () => {
    const errors = {};

    if (!config.input_file) {
      errors.input_file = 'Input file is required';
    }

    if (!config.output_file) {
      errors.output_file = 'Output file is required';
    }

    if (!config.teacher_model) {
      errors.teacher_model = 'Teacher model is required';
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
      step: 'generation',
      message: 'Running teacher pair generation...',
      error: null
    });

    try {
      // Update the config in the context
      updateConfig('teacher_pair_generation', config);

      // Execute the script
      const success = await executeScript('teacher_pair_generation', config);

      if (!success) {
        throw new Error('Teacher pair generation failed');
      }

      // All steps completed successfully
      setExecutionStatus({
        isRunning: false,
        step: 'completed',
        message: 'Teacher pair generation completed successfully',
        error: null
      });

      // Always navigate to logs page to show results
      console.log('Navigating to logs page for teacher_pair_generation');
      navigate('/logs/teacher_pair_generation');
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

  return (
    <div className="p-6">
      {/* Pipeline Navigation */}
      <PipelineNavigation />

      {/* Model Availability Status */}
      <div className="mb-4">
        <ModelAvailabilityStatus 
          onRefresh={async () => {
            // Refresh available models
            const containers = await scriptService.getDockerContainers();
            const teacherModels = containers.filter(container =>
              container.status === 'running' &&
              (container.name.toLowerCase().includes('phi') ||
               container.name.toLowerCase().includes('teacher') ||
               container.type?.toLowerCase().includes('teacher'))
            );
            setAvailableModels(teacherModels);
          }}
        />
      </div>

      <div className="bg-white rounded-lg shadow-card p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold">Teacher Pair Generation</h2>
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

        {/* Teacher Pair Generation Configuration */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-4 flex items-center">
            <Server size={18} className="mr-2 text-primary-600" />
            Teacher Pair Generation Configuration
          </h3>

          <div className="space-y-4">
            {/* Input file field */}
            <div className="form-group">
              <label htmlFor="input_file" className="form-label">
                Input File
              </label>
              <input
                id="input_file"
                type="text"
                value={config.input_file}
                onChange={(e) => handleInputChange('input_file', e.target.value)}
                className={`form-input ${validationErrors.input_file ? 'border-error-500' : ''}`}
                placeholder="Output/enriched_data.json"
              />
              {validationErrors.input_file && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.input_file}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Path to the enriched data JSON file from the Content Extraction & Enrichment step.
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
                value={config.output_file}
                onChange={(e) => handleInputChange('output_file', e.target.value)}
                className={`form-input ${validationErrors.output_file ? 'border-error-500' : ''}`}
                placeholder="Output/teacher_pairs.json"
              />
              {validationErrors.output_file && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.output_file}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Path where the teacher pairs will be saved. This file will be used as input for the Distillation Training step.
              </p>
            </div>

            {/* Teacher model selection */}
            <div className="form-group">
              <label htmlFor="teacher_model" className="form-label">
                Teacher Model
              </label>
              <select
                id="teacher_model"
                value={config.teacher_model}
                onChange={(e) => handleInputChange('teacher_model', e.target.value)}
                className={`form-select ${validationErrors.teacher_model ? 'border-error-500' : ''}`}
              >
                <option value="">Select a teacher model</option>
                {availableModels.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name} {model.type ? `(${model.type})` : ''}
                  </option>
                ))}
              </select>
              {validationErrors.teacher_model && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.teacher_model}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Select the teacher model to use for generating pairs. Only running Docker containers are shown.
              </p>
            </div>

            {/* Pipeline information */}
            <div className="mt-6 p-4 bg-info-50 rounded-lg">
              <h4 className="text-md font-medium mb-2 flex items-center text-info-800">
                <FileText size={16} className="mr-2" />
                Pipeline Information
              </h4>
              <p className="text-sm text-info-700">
                This step generates teaching pairs using a teacher model (like Phi-4) based on the enriched data from the previous step.
                The output will be used in the next step (Distillation Training) to train the student model.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TeacherPairGeneration;
