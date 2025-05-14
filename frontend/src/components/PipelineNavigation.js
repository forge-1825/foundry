import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { CheckCircle, Circle, ArrowRight } from 'lucide-react';
import { useSystem } from '../contexts/SystemContext';

const PipelineNavigation = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { systemStatus } = useSystem();

  // Define the pipeline steps
  const pipelineSteps = [
    {
      id: 'content_extraction_enrichment',
      name: 'Content Extraction & Enrichment',
      path: '/scripts/content_extraction_enrichment',
      step: 1
    },
    {
      id: 'teacher_pair_generation',
      name: 'Teacher Pair Generation',
      path: '/scripts/teacher_pair_generation',
      step: 2
    },
    {
      id: 'distillation',
      name: 'Distillation Training',
      path: '/scripts/distillation',
      step: 3
    },
    {
      id: 'merge_model',
      name: 'Model Merging',
      path: '/scripts/merge_model',
      step: 4
    },
    {
      id: 'student_self_study',
      name: 'Student Self-Study',
      path: '/scripts/student_self_study',
      step: 5
    },
    {
      id: 'evaluation',
      name: 'Model Evaluation',
      path: '/scripts/evaluation',
      step: 6
    }
  ];

  // Ensure these scripts are not accessible
  const hiddenScripts = ['manual_extractor', 'data_enrichment'];

  // Determine the current step based on the URL
  const getCurrentStepId = () => {
    const path = location.pathname;

    // Handle log pages
    if (path.startsWith('/logs/')) {
      const scriptId = path.split('/logs/')[1];

      // Redirect hidden scripts to content_extraction_enrichment
      if (hiddenScripts.includes(scriptId)) {
        return 'content_extraction_enrichment';
      }

      return scriptId;
    }

    // Handle script pages
    if (path.startsWith('/scripts/')) {
      const scriptId = path.split('/scripts/')[1];

      // Redirect hidden scripts to content_extraction_enrichment
      if (hiddenScripts.includes(scriptId)) {
        return 'content_extraction_enrichment';
      }

      return scriptId;
    }

    return null;
  };

  const currentStepId = getCurrentStepId();

  // Check if a step is active (currently running)
  const isStepActive = (stepId) => {
    return systemStatus &&
           systemStatus.active_scripts &&
           systemStatus.active_scripts.includes(stepId);
  };

  // Navigate to a step
  const goToStep = (step) => {
    navigate(step.path);
  };

  return (
    <div className="bg-white shadow-md rounded-lg p-4 mb-6">
      <h2 className="text-lg font-semibold mb-4">Distillation Pipeline</h2>
      <div className="flex flex-col space-y-2">
        {pipelineSteps.map((step, index) => {
          const isCurrentStep = step.id === currentStepId;
          const isActive = isStepActive(step.id);

          return (
            <div key={step.id} className="flex items-center">
              <div
                className={`flex items-center p-2 rounded-md cursor-pointer flex-grow ${
                  isCurrentStep ? 'bg-primary-100 text-primary-800 font-medium' : 'hover:bg-secondary-100'
                }`}
                onClick={() => goToStep(step)}
              >
                <div className="flex items-center justify-center w-8 h-8 rounded-full mr-3">
                  {isActive ? (
                    <div className="relative">
                      <Circle className="text-primary-500" />
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-2 h-2 bg-primary-500 rounded-full animate-pulse"></div>
                      </div>
                    </div>
                  ) : (
                    <div className="text-secondary-400">
                      {isCurrentStep ? (
                        <Circle className="text-primary-500" />
                      ) : (
                        <span className="flex items-center justify-center w-6 h-6 border border-secondary-300 rounded-full text-xs">
                          {step.step}
                        </span>
                      )}
                    </div>
                  )}
                </div>
                <span>{step.name}</span>
              </div>

              {index < pipelineSteps.length - 1 && (
                <div className="ml-4 mr-4">
                  <ArrowRight size={16} className="text-secondary-400" />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default PipelineNavigation;
