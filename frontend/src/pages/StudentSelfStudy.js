import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Play,
  AlertCircle,
  Info,
  Cpu,
  Database,
  FileText,
  Server,
  ChevronRight,
  ChevronDown,
  Book,
  Settings,
  Zap,
  Brain
} from 'lucide-react';
import { useScript } from '../contexts/ScriptContext';
import { useSystem } from '../contexts/SystemContext';
import { scriptService } from '../services/scriptService';
import PipelineNavigation from '../components/PipelineNavigation';

const StudentSelfStudy = () => {
  const navigate = useNavigate();
  const { scripts, configs, updateConfig, executeScript, loading, error } = useScript();
  const { systemStatus, dockerContainers } = useSystem();

  // State for configuration
  const [config, setConfig] = useState({
    // Basic configuration
    pdf_folder: '',
    model_path: 'Output/merged',
    output_dir: 'Output/self_study_results',

    // Teacher model configuration
    use_teacher: true,
    teacher_model: 'llama3_teacher_vllm',
    teacher_port: 8000,

    // Learning parameters
    topics_of_interest: 'cybersecurity, network scanning, vulnerability assessment',
    num_questions: 20,

    // Advanced options
    iterative_refinement: true,
    use_hierarchical_context: true,
    include_reasoning: true,
    use_rag: true,
    use_taboutt: true,

    // Performance tuning
    min_sentence_length: 5,
    max_sentence_length: 100,
    max_paragraph_size: 5,
    max_sentences: 10,

    // Display options
    verbose: true,
    show_thoughts: false,
    use_8bit: false
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

  // State for performance tuning mode
  const [performanceTuningMode, setPerformanceTuningMode] = useState(false);

  // State for available models
  const [availableTeacherModels, setAvailableTeacherModels] = useState([]);

  // Load Docker containers and set available models
  useEffect(() => {
    const fetchDockerContainers = async () => {
      try {
        const containers = await scriptService.getDockerContainers();

        // Filter for teacher model containers
        const teacherModels = containers.filter(container =>
          container.status === 'running' &&
          (container.name.toLowerCase().includes('llama') ||
           container.name.toLowerCase().includes('teacher') ||
           container.type?.toLowerCase().includes('teacher'))
        );

        setAvailableTeacherModels(teacherModels);

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
    const savedConfig = configs['student_self_study'];
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
    if (['num_questions', 'teacher_port', 'min_sentence_length', 'max_sentence_length', 'max_paragraph_size', 'max_sentences'].includes(field)) {
      value = parseInt(value);
      if (isNaN(value)) {
        value = 0;
      }
    }

    // Convert boolean values
    if (['use_teacher', 'iterative_refinement', 'use_hierarchical_context', 'include_reasoning', 'use_rag', 'use_taboutt', 'verbose', 'show_thoughts', 'use_8bit'].includes(field)) {
      if (typeof value === 'string') {
        value = value === 'true';
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

  // Toggle performance tuning mode
  const togglePerformanceTuningMode = () => {
    setPerformanceTuningMode(!performanceTuningMode);
  };

  // Validate configuration
  const validateConfig = () => {
    const errors = {};

    if (!config.pdf_folder) {
      errors.pdf_folder = 'PDF folder is required';
    }

    if (!config.model_path) {
      errors.model_path = 'Model path is required';
    }

    if (!config.output_dir) {
      errors.output_dir = 'Output directory is required';
    }

    if (config.use_teacher && !config.teacher_model) {
      errors.teacher_model = 'Teacher model is required when using teacher verification';
    }

    if (config.num_questions <= 0) {
      errors.num_questions = 'Number of questions must be greater than 0';
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
      step: 'self_study',
      message: 'Running student self-study...',
      error: null
    });

    try {
      // Update the config in the context
      updateConfig('student_self_study', config);

      // Execute the script
      const success = await executeScript('student_self_study', config);

      if (!success) {
        throw new Error('Student self-study failed');
      }

      // All steps completed successfully
      setExecutionStatus({
        isRunning: false,
        step: 'completed',
        message: 'Student self-study completed successfully',
        error: null
      });

      // Navigate to logs page to show results
      console.log('Navigating to logs page for student_self_study');
      navigate('/logs/student_self_study');
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
          <h2 className="text-xl font-semibold">Student Self-Study</h2>
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

        {/* Basic Configuration */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-4 flex items-center">
            <Book size={18} className="mr-2 text-primary-600" />
            Basic Configuration
          </h3>

          <div className="space-y-4">
            {/* PDF folder field */}
            <div className="form-group">
              <label htmlFor="pdf_folder" className="form-label">
                PDF Folder
              </label>
              <input
                id="pdf_folder"
                type="text"
                value={config.pdf_folder}
                onChange={(e) => handleInputChange('pdf_folder', e.target.value)}
                className={`form-input ${validationErrors.pdf_folder ? 'border-error-500' : ''}`}
                placeholder="AgentGreen"
              />
              {validationErrors.pdf_folder && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.pdf_folder}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Folder containing PDF files for the student to learn from.
              </p>
            </div>

            {/* Model path field */}
            <div className="form-group">
              <label htmlFor="model_path" className="form-label">
                Model Path
              </label>
              <input
                id="model_path"
                type="text"
                value={config.model_path}
                onChange={(e) => handleInputChange('model_path', e.target.value)}
                className={`form-input ${validationErrors.model_path ? 'border-error-500' : ''}`}
                placeholder="Output/merged"
              />
              {validationErrors.model_path && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.model_path}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Path to the merged model from the previous step.
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
                placeholder="Output/self_study_results"
              />
              {validationErrors.output_dir && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.output_dir}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Directory where the self-study results will be saved.
              </p>
            </div>

            {/* Topics of interest field */}
            <div className="form-group">
              <label htmlFor="topics_of_interest" className="form-label">
                Topics of Interest
              </label>
              <input
                id="topics_of_interest"
                type="text"
                value={config.topics_of_interest}
                onChange={(e) => handleInputChange('topics_of_interest', e.target.value)}
                className="form-input"
                placeholder="cybersecurity, network scanning, vulnerability assessment"
              />
              <p className="text-xs text-secondary-500 mt-1">
                Comma-separated list of topics the student should focus on during self-study.
              </p>
            </div>

            {/* Number of questions field */}
            <div className="form-group">
              <label htmlFor="num_questions" className="form-label">
                Number of Questions
              </label>
              <input
                id="num_questions"
                type="number"
                value={config.num_questions}
                onChange={(e) => handleInputChange('num_questions', e.target.value)}
                className={`form-input ${validationErrors.num_questions ? 'border-error-500' : ''}`}
                min="1"
                max="100"
              />
              {validationErrors.num_questions && (
                <p className="text-error-500 text-sm mt-1">{validationErrors.num_questions}</p>
              )}
              <p className="text-xs text-secondary-500 mt-1">
                Number of questions to generate per sentence (1-100).
              </p>
            </div>
          </div>
        </div>

        {/* Teacher Model Configuration */}
        <div className="mb-6">
          <h3 className="text-lg font-medium mb-4 flex items-center">
            <Server size={18} className="mr-2 text-primary-600" />
            Teacher Model Configuration
          </h3>

          <div className="space-y-4">
            {/* Use teacher checkbox */}
            <div className="form-group">
              <div className="flex items-center">
                <input
                  id="use_teacher"
                  type="checkbox"
                  checked={config.use_teacher}
                  onChange={(e) => handleInputChange('use_teacher', e.target.checked)}
                  className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                />
                <label htmlFor="use_teacher" className="ml-2 block text-sm font-medium text-secondary-700">
                  Use Teacher Model for Verification
                </label>
              </div>
              <p className="text-xs text-secondary-500 mt-1 ml-6">
                Use the teacher model to verify and improve the student's answers.
              </p>
            </div>

            {config.use_teacher && (
              <>
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
                    {availableTeacherModels.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.name} {model.type ? `(${model.type})` : ''}
                      </option>
                    ))}
                  </select>
                  {validationErrors.teacher_model && (
                    <p className="text-error-500 text-sm mt-1">{validationErrors.teacher_model}</p>
                  )}
                  <p className="text-xs text-secondary-500 mt-1">
                    Select the teacher model to use for verification. Only running Docker containers are shown.
                  </p>
                </div>

                {/* Teacher port field */}
                <div className="form-group">
                  <label htmlFor="teacher_port" className="form-label">
                    Teacher Port
                  </label>
                  <input
                    id="teacher_port"
                    type="number"
                    value={config.teacher_port}
                    onChange={(e) => handleInputChange('teacher_port', e.target.value)}
                    className="form-input"
                    min="1"
                    max="65535"
                  />
                  <p className="text-xs text-secondary-500 mt-1">
                    Port for the teacher model vLLM server (default: 8000).
                  </p>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Advanced Options */}
        <div className="mb-6">
          <button
            type="button"
            className="flex items-center text-primary-600 hover:text-primary-800 mb-4"
            onClick={toggleAdvancedMode}
          >
            {advancedMode ? (
              <ChevronDown size={16} className="mr-1" />
            ) : (
              <ChevronRight size={16} className="mr-1" />
            )}
            <Settings size={18} className="mr-2" />
            Advanced Options
          </button>

          {advancedMode && (
            <div className="space-y-4 pl-6 border-l-2 border-primary-100">
              {/* Iterative refinement checkbox */}
              <div className="form-group">
                <div className="flex items-center">
                  <input
                    id="iterative_refinement"
                    type="checkbox"
                    checked={config.iterative_refinement}
                    onChange={(e) => handleInputChange('iterative_refinement', e.target.checked)}
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                  />
                  <label htmlFor="iterative_refinement" className="ml-2 block text-sm font-medium text-secondary-700">
                    Use Iterative Refinement
                  </label>
                </div>
                <p className="text-xs text-secondary-500 mt-1 ml-6">
                  Iteratively refine answers for better quality.
                </p>
              </div>

              {/* Hierarchical context checkbox */}
              <div className="form-group">
                <div className="flex items-center">
                  <input
                    id="use_hierarchical_context"
                    type="checkbox"
                    checked={config.use_hierarchical_context}
                    onChange={(e) => handleInputChange('use_hierarchical_context', e.target.checked)}
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                  />
                  <label htmlFor="use_hierarchical_context" className="ml-2 block text-sm font-medium text-secondary-700">
                    Use Hierarchical Context
                  </label>
                </div>
                <p className="text-xs text-secondary-500 mt-1 ml-6">
                  Use hierarchical context encoding with paragraph summaries.
                </p>
              </div>

              {/* Include reasoning checkbox */}
              <div className="form-group">
                <div className="flex items-center">
                  <input
                    id="include_reasoning"
                    type="checkbox"
                    checked={config.include_reasoning}
                    onChange={(e) => handleInputChange('include_reasoning', e.target.checked)}
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                  />
                  <label htmlFor="include_reasoning" className="ml-2 block text-sm font-medium text-secondary-700">
                    Include Chain-of-Thought Reasoning
                  </label>
                </div>
                <p className="text-xs text-secondary-500 mt-1 ml-6">
                  Include step-by-step reasoning in prompts.
                </p>
              </div>

              {/* Use RAG checkbox */}
              <div className="form-group">
                <div className="flex items-center">
                  <input
                    id="use_rag"
                    type="checkbox"
                    checked={config.use_rag}
                    onChange={(e) => handleInputChange('use_rag', e.target.checked)}
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                  />
                  <label htmlFor="use_rag" className="ml-2 block text-sm font-medium text-secondary-700 flex items-center">
                    Use RAG Integration
                    <Zap size={16} className="ml-2 text-yellow-500" />
                  </label>
                </div>
                <p className="text-xs text-secondary-500 mt-1 ml-6">
                  Use Retrieval-Augmented Generation for better context understanding.
                </p>
              </div>

              {/* Use TaboutT checkbox */}
              <div className="form-group">
                <div className="flex items-center">
                  <input
                    id="use_taboutt"
                    type="checkbox"
                    checked={config.use_taboutt}
                    onChange={(e) => handleInputChange('use_taboutt', e.target.checked)}
                    className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                  />
                  <label htmlFor="use_taboutt" className="ml-2 block text-sm font-medium text-secondary-700 flex items-center">
                    Use TaboutT Framework
                    <Brain size={16} className="ml-2 text-purple-500" />
                  </label>
                </div>
                <p className="text-xs text-secondary-500 mt-1 ml-6">
                  Use TaboutT framework for enhanced reasoning and planning.
                </p>
              </div>

              {/* Performance Tuning */}
              <div className="mt-4">
                <button
                  type="button"
                  className="flex items-center text-primary-600 hover:text-primary-800 mb-2"
                  onClick={togglePerformanceTuningMode}
                >
                  {performanceTuningMode ? (
                    <ChevronDown size={16} className="mr-1" />
                  ) : (
                    <ChevronRight size={16} className="mr-1" />
                  )}
                  <Cpu size={16} className="mr-2" />
                  Performance Tuning
                </button>

                {performanceTuningMode && (
                  <div className="space-y-3 pl-6 border-l-2 border-primary-100 mt-2">
                    {/* Min sentence length */}
                    <div className="form-group">
                      <label htmlFor="min_sentence_length" className="form-label text-sm">
                        Min Sentence Length (words)
                      </label>
                      <input
                        id="min_sentence_length"
                        type="number"
                        value={config.min_sentence_length}
                        onChange={(e) => handleInputChange('min_sentence_length', e.target.value)}
                        className="form-input"
                        min="1"
                        max="50"
                      />
                    </div>

                    {/* Max sentence length */}
                    <div className="form-group">
                      <label htmlFor="max_sentence_length" className="form-label text-sm">
                        Max Sentence Length (words)
                      </label>
                      <input
                        id="max_sentence_length"
                        type="number"
                        value={config.max_sentence_length}
                        onChange={(e) => handleInputChange('max_sentence_length', e.target.value)}
                        className="form-input"
                        min={config.min_sentence_length}
                        max="1000"
                      />
                    </div>

                    {/* Max paragraph size */}
                    <div className="form-group">
                      <label htmlFor="max_paragraph_size" className="form-label text-sm">
                        Max Paragraph Size (sentences)
                      </label>
                      <input
                        id="max_paragraph_size"
                        type="number"
                        value={config.max_paragraph_size}
                        onChange={(e) => handleInputChange('max_paragraph_size', e.target.value)}
                        className="form-input"
                        min="1"
                        max="20"
                      />
                    </div>

                    {/* Max sentences */}
                    <div className="form-group">
                      <label htmlFor="max_sentences" className="form-label text-sm">
                        Max Sentences per PDF
                      </label>
                      <input
                        id="max_sentences"
                        type="number"
                        value={config.max_sentences}
                        onChange={(e) => handleInputChange('max_sentences', e.target.value)}
                        className="form-input"
                        min="1"
                        max="100"
                      />
                    </div>
                  </div>
                )}
              </div>

              {/* Display Options */}
              <div className="mt-4">
                <h4 className="text-md font-medium mb-2">Display Options</h4>

                {/* Verbose checkbox */}
                <div className="form-group">
                  <div className="flex items-center">
                    <input
                      id="verbose"
                      type="checkbox"
                      checked={config.verbose}
                      onChange={(e) => handleInputChange('verbose', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                    />
                    <label htmlFor="verbose" className="ml-2 block text-sm font-medium text-secondary-700">
                      Verbose Output
                    </label>
                  </div>
                </div>

                {/* Show thoughts checkbox */}
                <div className="form-group">
                  <div className="flex items-center">
                    <input
                      id="show_thoughts"
                      type="checkbox"
                      checked={config.show_thoughts}
                      onChange={(e) => handleInputChange('show_thoughts', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                    />
                    <label htmlFor="show_thoughts" className="ml-2 block text-sm font-medium text-secondary-700">
                      Show Student's Thought Process
                    </label>
                  </div>
                </div>

                {/* Use 8-bit checkbox */}
                <div className="form-group">
                  <div className="flex items-center">
                    <input
                      id="use_8bit"
                      type="checkbox"
                      checked={config.use_8bit}
                      onChange={(e) => handleInputChange('use_8bit', e.target.checked)}
                      className="h-4 w-4 text-primary-600 focus:ring-primary-500 border-secondary-300 rounded"
                    />
                    <label htmlFor="use_8bit" className="ml-2 block text-sm font-medium text-secondary-700">
                      Load Model in 8-bit Precision
                    </label>
                  </div>
                  <p className="text-xs text-secondary-500 mt-1 ml-6">
                    Use 8-bit precision to save memory (may reduce quality).
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Pipeline information */}
        <div className="mt-6 p-4 bg-info-50 rounded-lg">
          <h4 className="text-md font-medium mb-2 flex items-center text-info-800">
            <FileText size={16} className="mr-2" />
            Pipeline Information
          </h4>
          <p className="text-sm text-info-700">
            This step allows the student model to further learn from the data through self-study.
            The student will generate questions and answers based on the PDF content, and the teacher model
            can verify and improve the answers if enabled. The results will be saved to the specified output directory.
          </p>
        </div>
      </div>
    </div>
  );
};

export default StudentSelfStudy;
