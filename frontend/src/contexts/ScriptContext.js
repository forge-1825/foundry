import React, { createContext, useContext, useState, useEffect } from 'react';
import { scriptService } from '../services/scriptService';

export const ScriptContext = createContext();

export const useScript = () => {
  const context = useContext(ScriptContext);
  if (!context) {
    throw new Error('useScript must be used within a ScriptProvider');
  }
  return context;
};

export const ScriptProvider = ({ children }) => {
  const [scripts, setScripts] = useState([]);
  const [configs, setConfigs] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Hardcoded scripts for testing
  const hardcodedScripts = [
    {
      id: "manual_extractor",
      name: "Manual Extractor",
      description: "Extract content from technical documents to PDF and JSON",
      step: 1,
      hidden: true
    },
    {
      id: "data_enrichment",
      name: "Data Enrichment",
      description: "Clean text, extract entities, and generate summaries using GPU acceleration",
      step: 2,
      hidden: true
    },
    {
      id: "content_extraction_enrichment",
      name: "Content Extraction & Enrichment",
      description: "Extract and enrich content from various sources in a single step",
      step: 1
    },
    {
      id: "teacher_pair_generation",
      name: "Teacher Pair Generation",
      description: "Generate teacher model outputs using Phi-4 via vLLM",
      step: 2
    },
    {
      id: "distillation",
      name: "Distillation Training",
      description: "Train student model using enhanced prompt engineering and parameters",
      step: 3
    },
    {
      id: "student_self_study",
      name: "Student Self-Study",
      description: "Enable the distilled model to learn from additional domain-specific content",
      step: 4
    },
    {
      id: "merge_model",
      name: "Model Merging",
      description: "Merge the base model with trained LoRA adapters for deployment",
      step: 5
    },
    {
      id: "evaluation",
      name: "Model Evaluation",
      description: "Test distilled model on sample prompts",
      step: 6
    }
  ];

  // Hardcoded configs for testing
  const hardcodedConfigs = {
    "manual_extractor": {
      "url": "https://example.com/docs",
      "source_folder": "",
      "output_dir": "/data/extracted"
    },
    "data_enrichment": {
      "input_file": "/data/extracted/extracted_data.json",
      "output_file": "/data/enriched/enriched_data.json",
      "source_folder": ""
    },
    "content_extraction_enrichment": {
      "url": "https://example.com/docs",
      "source_folder": "",
      "docker_folder": "/data",
      "output_dir": "/data/extracted",
      "extract_links": false,
      "enable_enrichment": true,
      "input_file": "/data/extracted/extracted_data.json",
      "output_file": "/data/enriched/enriched_data.json",
      "enable_entity_extraction": true,
      "enable_summarization": true,
      "enable_keyword_extraction": true,
      "use_gpu": true
    },
    "teacher_pair_generation": {
      "input_file": "/data/enriched/enriched_data.json",
      "output_file": "/data/teacher_pairs/teacher_pairs.json",
      "teacher_model": "jakiAJK/microsoft-phi-4_GPTQ-int4"
    },
    "distillation": {
      "teacher_pairs": "/data/teacher_pairs/teacher_pairs.json",
      "student_model": "WhiteRabbitNeo-13B-v1",
      "output_dir": "/data/distilled_model",
      "beta": 0.1,
      "lambda": 0.1,
      "epochs": 15
    },
    "student_self_study": {
      "pdf_folder": "/data/domain_pdfs",
      "model_path": "/data/distilled_model/best_checkpoint",
      "output_dir": "/data/self_study_results",
      "num_questions": 3,
      "use_hierarchical_context": true,
      "include_reasoning": true
    },
    "merge_model": {
      "adapter_path": "/data/distilled_model/best_checkpoint",
      "output_path": "/data/merged_distilled_model"
    },
    "evaluation": {
      "model_path": "/data/merged_distilled_model",
      "test_prompt": "Extract detailed technical requirements for a new IoT device in the healthcare domain:"
    }
  };

  // Fetch available scripts on mount
  useEffect(() => {
    const fetchScripts = async () => {
      setLoading(true);
      try {
        // Try to fetch from API first
        const data = await scriptService.listScripts();
        setScripts(data);

        // Initialize configs with default values
        const initialConfigs = {};
        for (const script of data) {
          const config = await scriptService.getScriptConfig(script.id);
          initialConfigs[script.id] = config;
        }
        setConfigs(initialConfigs);
        setError(null);
      } catch (err) {
        console.error('Error fetching scripts:', err);
        console.log('Using hardcoded scripts and configs instead');

        // Use hardcoded values if API fails
        setScripts(hardcodedScripts);
        setConfigs(hardcodedConfigs);
        setError('Using hardcoded scripts (API connection failed)');
      } finally {
        setLoading(false);
      }
    };

    fetchScripts();
  }, []);

  // Update a script configuration
  const updateConfig = (scriptId, newConfig) => {
    setConfigs(prevConfigs => ({
      ...prevConfigs,
      [scriptId]: {
        ...prevConfigs[scriptId],
        ...newConfig
      }
    }));
  };

  // Execute a script with the current configuration or a custom configuration
  const executeScript = async (scriptId, customConfig = null) => {
    setLoading(true);
    try {
      const config = customConfig || configs[scriptId];
      await scriptService.executeScript(scriptId, config);
      return true;
    } catch (err) {
      console.error(`Error executing script ${scriptId}:`, err);
      setError(`Failed to execute script: ${err.message}`);
      return false;
    } finally {
      setLoading(false);
    }
  };

  // Save a configuration
  const saveConfig = async (scriptId, name) => {
    try {
      const config = configs[scriptId];
      const configData = {
        name: name || `${scriptId}_config_${Date.now()}`,
        scriptId,
        config
      };
      await scriptService.saveConfig(configData);
      return true;
    } catch (err) {
      console.error('Error saving configuration:', err);
      setError(`Failed to save configuration: ${err.message}`);
      return false;
    }
  };

  // Load a saved configuration
  const loadConfig = async (configName) => {
    try {
      const config = await scriptService.getConfig(configName);
      if (config && config.scriptId && config.config) {
        updateConfig(config.scriptId, config.config);
        return true;
      }
      return false;
    } catch (err) {
      console.error('Error loading configuration:', err);
      setError(`Failed to load configuration: ${err.message}`);
      return false;
    }
  };

  const value = {
    scripts,
    configs,
    loading,
    error,
    updateConfig,
    executeScript,
    saveConfig,
    loadConfig
  };

  return (
    <ScriptContext.Provider value={value}>
      {children}
    </ScriptContext.Provider>
  );
};
