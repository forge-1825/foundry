import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Play,
  AlertCircle,
  Info,
  Cpu,
  Database,
  FileText,
  Server
} from 'lucide-react';
import { useScript } from '../contexts/ScriptContext';
import { useSystem } from '../contexts/SystemContext';
import { scriptService } from '../services/scriptService';
import PipelineNavigation from '../components/PipelineNavigation';

const DistillationPhase = () => {
  const navigate = useNavigate();
  const { scripts, configs, updateConfig, executeScript, loading, error } = useScript();
  const { systemStatus, dockerContainers } = useSystem();

  // State for configuration
  const [config, setConfig] = useState({
    teacher_pairs: '',
    student_model: '',
    output_dir: '',
    beta: 0.1,
    lambda: 0.1,
    epochs: 15
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

  // State for available models
  const [availableTeacherModels, setAvailableTeacherModels] = useState([]);
  const [availableStudentModels, setAvailableStudentModels] = useState([]);

  // Load Docker containers and set available models
  useEffect(() => {
    const fetchDockerContainers = async () => {
      try {
        const containers = await scriptService.getDockerContainers();

        // Filter for teacher model containers
        const teacherModels = containers.filter(container =>
          container.status === 'running' &&
          (container.name.toLowerCase().includes('phi4') ||
           container.name.toLowerCase().includes('teacher') ||
           container.type?.toLowerCase().includes('teacher'))
        );

        // Filter for student model containers
        const studentModels = containers.filter(container =>
          container.status === 'running' &&
          (container.name.toLowerCase().includes('phi3') ||
           container.name.toLowerCase().includes('student') ||
           container.type?.toLowerCase().includes('student'))
        );

        setAvailableTeacherModels(teacherModels);
        setAvailableStudentModels(studentModels);

        // If we have models and no model is selected yet, select the first one
        if (studentModels.length > 0 && !config.student_model) {
          setConfig(prev => ({
            ...prev,
            student_model: studentModels[0].name
          }));
        }
      } catch (err) {
        console.error('Error fetching Docker containers:', err);
      }
    };

    fetchDockerContainers();

    // Also load saved config if available
    const savedConfig = configs['distillation'];
    if (savedConfig) {
      setConfig(prev => ({
        ...prev,
        ...savedConfig
      }));
    }
  }, [configs]);

  // Handle input change
  const handleInputChange = (field, value) => {
    // Convert numeric values
    if (['beta', 'lambda', 'epochs'].includes(field)) {
      value = parseFloat(value);
      if (isNaN(value)) {
        value = 0;
      }
    }

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

  // Toggle advanced mode
  const toggleAdvancedMode = () => {
    setAdvancedMode(!advancedMode);
  };

  // Validate configuration
  const validateConfig = () => {
    const errors = {};

    if (!config.teacher_pairs) {
      errors.teacher_pairs = 'Teacher pairs file is required';
    }

    if (!config.student_model) {
      errors.student_model = 'Student model is required';
    }

    if (!config.output_dir) {
      errors.output_dir = 'Output directory is required';
    }

    if (config.epochs <= 0) {
      errors.epochs = 'Epochs must be greater than 0';
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
      step: 'distillation',
      message: 'Running distillation training...',
      error: null
    });

    try {
      // Update the config in the context
      updateConfig('distillation', config);

      // Execute the script
      const success = await executeScript('distillation', config);

      if (!success) {
        throw new Error('Distillation training failed');
      }

      // All steps completed successfully
      setExecutionStatus({
        isRunning: false,
        step: 'completed',
        message: 'Distillation training completed successfully',
        error: null
      });

      // Always navigate to logs page to show results
      console.log('Navigating to logs page for distillation');
      navigate('/logs/distillation');
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

      <div className="bg-white rounded-lg shadow-card p-6">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-xl font-semibold">Distillation Training</h2>
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

        {/* Distillation Configuration */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-4 flex items-center">
            <Cpu size={18} className="mr-2 text-primary-600" />
            Distillation Configuration
          </h3>

          <div className="space-y-4">
            {/* Teacher pairs file field */}
            <div className="form-group">
              <label htmlFor="teacher_pairs" className="form-label">
                Teacher Pairs File
              </label>
              <input
                id="teacher_pairs"
                type="text"
                value={config.teacher_pairs}
                onChange={(e) => handleInputChange('teacher_pairs', e.target.value)}
                className={`form-input ${validationErrors.teacher_pairs ? 'border-error-500' : ''}`}
                placeholder="Output/teacher_pairs.json"
              />
              {validationErrors.teacher_pairs && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.teacher_pairs}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Path to the teacher pairs JSON file from the Teacher Pair Generation step.
              </p>
            </div>

            {/* Student model selection */}
            <div className="form-group">
              <label htmlFor="student_model" className="form-label">
                Student Model
              </label>
              <select
                id="student_model"
                value={config.student_model}
                onChange={(e) => handleInputChange('student_model', e.target.value)}
                className={`form-select ${validationErrors.student_model ? 'border-error-500' : ''}`}
              >
                <option value="">Select a student model</option>
                {availableStudentModels.map((model) => (
                  <option key={model.name} value={model.name}>
                    {model.name} {model.type ? `(${model.type})` : ''}
                  </option>
                ))}
              </select>
              {validationErrors.student_model && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.student_model}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Select the student model to train. Only running Docker containers are shown.
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
                value={config.output_dir}
                onChange={(e) => handleInputChange('output_dir', e.target.value)}
                className={`form-input ${validationErrors.output_dir ? 'border-error-500' : ''}`}
                placeholder="Output/distilled_model"
              />
              {validationErrors.output_dir && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.output_dir}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Directory where the trained model will be saved.
              </p>
            </div>

            {/* Advanced options toggle */}
            <div className="mt-6">
              <button
                type="button"
                className="flex items-center text-primary-600 hover:text-primary-800"
                onClick={toggleAdvancedMode}
              >
                {advancedMode ? (
                  <AlertCircle size={16} className="mr-2" />
                ) : (
                  <Info size={16} className="mr-2" />
                )}
                {advancedMode ? 'Hide Advanced Options' : 'Show Advanced Options'}
              </button>
            </div>

            {/* Advanced options */}
            {advancedMode && (
              <div className="mt-4 p-4 bg-secondary-50 rounded-lg space-y-4">
                <h4 className="text-md font-medium mb-2">Advanced Training Parameters</h4>

                {/* Beta parameter */}
                <div className="form-group">
                  <label htmlFor="beta" className="form-label">
                    Beta (Knowledge Distillation Weight)
                  </label>
                  <input
                    id="beta"
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={config.beta}
                    onChange={(e) => handleInputChange('beta', e.target.value)}
                    className="form-input"
                  />
                  <p className="text-xs text-secondary-500 mt-1">
                    Weight for the knowledge distillation loss (0-1). Higher values emphasize matching the teacher's outputs.
                  </p>
                </div>

                {/* Lambda parameter */}
                <div className="form-group">
                  <label htmlFor="lambda" className="form-label">
                    Lambda (Regularization Weight)
                  </label>
                  <input
                    id="lambda"
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={config.lambda}
                    onChange={(e) => handleInputChange('lambda', e.target.value)}
                    className="form-input"
                  />
                  <p className="text-xs text-secondary-500 mt-1">
                    Weight for the regularization loss (0-1). Higher values prevent overfitting.
                  </p>
                </div>

                {/* Epochs parameter */}
                <div className="form-group">
                  <label htmlFor="epochs" className="form-label">
                    Training Epochs
                  </label>
                  <input
                    id="epochs"
                    type="number"
                    step="1"
                    min="1"
                    value={config.epochs}
                    onChange={(e) => handleInputChange('epochs', e.target.value)}
                    className={`form-input ${validationErrors.epochs ? 'border-error-500' : ''}`}
                  />
                  {validationErrors.epochs && (
                    <p className="text-error-500 text-sm mt-1">{validationErrors.epochs}</p>
                  )}
                  <p className="text-xs text-secondary-500 mt-1">
                    Number of training epochs. More epochs may improve results but take longer to train.
                  </p>
                </div>
              </div>
            )}

            {/* Pipeline information */}
            <div className="mt-6 p-4 bg-info-50 rounded-lg">
              <h4 className="text-md font-medium mb-2 flex items-center text-info-800">
                <FileText size={16} className="mr-2" />
                Pipeline Information
              </h4>
              <p className="text-sm text-info-700">
                This step trains the student model using the teacher pairs generated in the previous step.
                The trained model will be saved to the specified output directory and can be used in the next step (Student Self-Study).
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DistillationPhase;
