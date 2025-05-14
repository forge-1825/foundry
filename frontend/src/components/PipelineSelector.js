import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ArrowRight,
  Check,
  Info,
  AlertCircle,
  ChevronRight,
  ChevronDown,
  Layers,
  Brain,
  GitMerge
} from 'lucide-react';
import { useScript } from '../contexts/ScriptContext';

const PipelineSelector = ({ onClose, outputDir }) => {
  const navigate = useNavigate();
  const { executeScript } = useScript();
  const [selectedPipeline, setSelectedPipeline] = useState('standard');
  const [selectedPRDPhase, setSelectedPRDPhase] = useState('1');
  const [selectedPRDModel, setSelectedPRDModel] = useState('3');
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionStatus, setExecutionStatus] = useState({
    status: null,
    message: null,
    error: null
  });

  const pipelines = [
    {
      id: 'standard',
      name: 'Standard Pipeline',
      description: 'Teacher Pair Generation + Standard Distillation',
      details: 'The standard pipeline generates teacher-student pairs and uses them for distillation training. This is the traditional approach for knowledge distillation.'
    },
    {
      id: 'prd',
      name: 'PRD Pipeline',
      description: 'Preference Reasoning Datasets + DPO',
      details: 'The PRD pipeline uses Preference Reasoning Datasets methodology with Direct Preference Optimization. This approach can produce better results for reasoning tasks.'
    }
  ];

  const prdPhases = [
    {
      id: '1',
      name: 'Phase 1: Reasoning Only',
      description: 'Focus on basic reasoning capabilities',
      icon: <Brain size={16} className="mr-2 text-primary-600" />
    },
    {
      id: '2',
      name: 'Phase 2: Reasoning + Error Analysis',
      description: 'Add error analysis capabilities',
      icon: <AlertCircle size={16} className="mr-2 text-primary-600" />
    },
    {
      id: '3',
      name: 'Phase 3: Full PRD',
      description: 'Complete PRD with reasoning, error analysis, and constructive feedback',
      icon: <Layers size={16} className="mr-2 text-primary-600" />
    },
    {
      id: '4',
      name: 'Phase 4: Model Merging',
      description: 'Merge a PRD-trained model with the base model',
      icon: <GitMerge size={16} className="mr-2 text-primary-600" />
    }
  ];

  const prdModels = [
    {
      id: '1',
      name: 'Phase 1 Model',
      path: 'distilled_model_prd_phase1',
      description: 'Basic reasoning capabilities'
    },
    {
      id: '2',
      name: 'Phase 2 Model',
      path: 'distilled_model_prd_phase2',
      description: 'Reasoning with error analysis'
    },
    {
      id: '3',
      name: 'Phase 3 Model',
      path: 'distilled_model_prd_phase3',
      description: 'Full PRD capabilities (recommended)'
    }
  ];

  const handlePipelineSelect = (pipelineId) => {
    setSelectedPipeline(pipelineId);
  };

  const handlePRDPhaseSelect = (phaseId) => {
    setSelectedPRDPhase(phaseId);

    // If selecting Phase 4, ensure a model is selected
    if (phaseId === '4' && !selectedPRDModel) {
      setSelectedPRDModel('3'); // Default to Phase 3 model
    }
  };

  const handlePRDModelSelect = (modelId) => {
    setSelectedPRDModel(modelId);
  };

  const handleExecute = async () => {
    setIsExecuting(true);

    // Prepare status message based on selection
    let statusMessage = '';
    if (selectedPipeline === 'standard') {
      statusMessage = 'Starting Standard pipeline...';
    } else {
      const phase = prdPhases.find(p => p.id === selectedPRDPhase);
      statusMessage = `Starting PRD pipeline - ${phase.name}...`;
    }

    setExecutionStatus({
      status: 'running',
      message: statusMessage,
      error: null
    });

    try {
      // Execute the pipeline selector script
      const config = {
        pipeline_type: selectedPipeline,
        output_dir: outputDir || 'Output'
      };

      // Add PRD-specific parameters if PRD pipeline is selected
      if (selectedPipeline === 'prd') {
        config.prd_phase = selectedPRDPhase;

        // Add model selection for Phase 4
        if (selectedPRDPhase === '4') {
          const selectedModel = prdModels.find(m => m.id === selectedPRDModel);
          config.prd_model_path = selectedModel.path;
        }
      }

      const success = await executeScript('post_enrichment_pipeline_selector', config);

      if (!success) {
        throw new Error('Pipeline selection failed');
      }

      // Prepare completion message
      let completionMessage = '';
      if (selectedPipeline === 'standard') {
        completionMessage = 'Standard pipeline selected successfully.';
      } else {
        const phase = prdPhases.find(p => p.id === selectedPRDPhase);
        completionMessage = `PRD pipeline - ${phase.name} selected successfully.`;
      }

      setExecutionStatus({
        status: 'completed',
        message: completionMessage,
        error: null
      });

      // Navigate to the appropriate next step
      setTimeout(() => {
        // For standard pipeline, always go to teacher_pair_generation
        if (selectedPipeline === 'standard') {
          navigate('/scripts/teacher_pair_generation');
        } else {
          // For PRD pipeline, navigation depends on the phase
          if (selectedPRDPhase === '4') {
            // Phase 4 is model merging
            navigate('/scripts/merge_model');
          } else {
            // Phases 1-3 start with teacher pair generation
            navigate('/scripts/teacher_pair_generation');
          }
        }
      }, 1500);
    } catch (err) {
      console.error('Pipeline selection error:', err);
      setExecutionStatus({
        status: 'error',
        message: null,
        error: err.message || 'An error occurred during pipeline selection'
      });
    } finally {
      setIsExecuting(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 max-w-2xl mx-auto">
      <h2 className="text-xl font-semibold mb-4">Select Pipeline</h2>

      <p className="text-secondary-600 mb-6">
        Choose which pipeline to use for the next steps in the distillation process.
      </p>

      {/* Pipeline options */}
      <div className="space-y-4 mb-6">
        {pipelines.map((pipeline) => (
          <div
            key={pipeline.id}
            className={`border rounded-lg p-4 cursor-pointer transition-colors ${
              selectedPipeline === pipeline.id
                ? 'border-primary-500 bg-primary-50'
                : 'border-secondary-200 hover:border-secondary-300'
            }`}
            onClick={() => handlePipelineSelect(pipeline.id)}
          >
            <div className="flex items-center">
              <div className={`w-5 h-5 rounded-full border flex items-center justify-center mr-3 ${
                selectedPipeline === pipeline.id
                  ? 'border-primary-500 bg-primary-500 text-white'
                  : 'border-secondary-300'
              }`}>
                {selectedPipeline === pipeline.id && <Check size={12} />}
              </div>
              <div className="flex-grow">
                <h3 className="font-medium">{pipeline.name}</h3>
                <p className="text-sm text-secondary-600">{pipeline.description}</p>
              </div>
            </div>

            {selectedPipeline === pipeline.id && (
              <div className="mt-3 pl-8 text-sm text-secondary-700 border-l-2 border-primary-100 ml-2 py-1">
                {pipeline.details}
              </div>
            )}

            {/* PRD Phase Selection */}
            {selectedPipeline === 'prd' && pipeline.id === 'prd' && (
              <div className="mt-4 pt-4 border-t border-secondary-200">
                <h4 className="text-sm font-medium mb-3 flex items-center">
                  <Layers size={16} className="mr-2 text-primary-600" />
                  PRD Phase Selection
                </h4>
                <div className="space-y-3">
                  {prdPhases.map((phase) => (
                    <div
                      key={phase.id}
                      className={`flex items-center p-2 rounded cursor-pointer ${
                        selectedPRDPhase === phase.id
                          ? 'bg-primary-100'
                          : 'hover:bg-secondary-50'
                      }`}
                      onClick={(e) => {
                        e.stopPropagation();
                        handlePRDPhaseSelect(phase.id);
                      }}
                    >
                      <div className={`w-4 h-4 rounded-full border flex items-center justify-center mr-2 ${
                        selectedPRDPhase === phase.id
                          ? 'border-primary-500 bg-primary-500 text-white'
                          : 'border-secondary-300'
                      }`}>
                        {selectedPRDPhase === phase.id && <Check size={10} />}
                      </div>
                      <div className="flex-grow">
                        <div className="flex items-center">
                          {phase.icon}
                          <span className="font-medium text-sm">{phase.name}</span>
                        </div>
                        <p className="text-xs text-secondary-600 ml-6">{phase.description}</p>
                      </div>
                    </div>
                  ))}
                </div>

                {/* PRD Model Selection for Phase 4 */}
                {selectedPRDPhase === '4' && (
                  <div className="mt-4 pt-3 border-t border-secondary-200">
                    <h4 className="text-sm font-medium mb-3 flex items-center">
                      <Brain size={16} className="mr-2 text-primary-600" />
                      Select PRD Model to Merge
                    </h4>
                    <div className="space-y-2 pl-2">
                      {prdModels.map((model) => (
                        <div
                          key={model.id}
                          className={`flex items-center p-2 rounded cursor-pointer ${
                            selectedPRDModel === model.id
                              ? 'bg-primary-100'
                              : 'hover:bg-secondary-50'
                          }`}
                          onClick={(e) => {
                            e.stopPropagation();
                            handlePRDModelSelect(model.id);
                          }}
                        >
                          <div className={`w-4 h-4 rounded-full border flex items-center justify-center mr-2 ${
                            selectedPRDModel === model.id
                              ? 'border-primary-500 bg-primary-500 text-white'
                              : 'border-secondary-300'
                          }`}>
                            {selectedPRDModel === model.id && <Check size={10} />}
                          </div>
                          <div>
                            <span className="font-medium text-sm">{model.name}</span>
                            <p className="text-xs text-secondary-600">{model.description}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Status messages */}
      {executionStatus.message && (
        <div className={`mb-4 p-3 rounded ${
          executionStatus.status === 'completed'
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

      {/* Action buttons */}
      <div className="flex justify-end space-x-3 mt-6">
        <button
          className="btn btn-secondary"
          onClick={onClose}
          disabled={isExecuting}
        >
          Cancel
        </button>
        <button
          className="btn btn-primary flex items-center"
          onClick={handleExecute}
          disabled={isExecuting}
        >
          {isExecuting ? (
            <>
              <div className="animate-spin mr-2 h-4 w-4 border-2 border-white rounded-full border-t-transparent"></div>
              Processing...
            </>
          ) : (
            <>
              Continue <ArrowRight size={16} className="ml-1" />
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default PipelineSelector;
