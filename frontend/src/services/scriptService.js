import api from './apiService';

export const scriptService = {
  // Get a list of all available scripts
  listScripts: async () => {
    return api.get('/api/scripts');
  },

  // Get configuration template for a script
  getScriptConfig: async (scriptId) => {
    return api.get(`/api/scripts/${scriptId}/config`);
  },

  // Execute a script with the given configuration
  executeScript: async (scriptId, config) => {
    return api.post(`/api/scripts/${scriptId}/execute`, config);
  },

  // Get the current pipeline status
  getPipelineStatus: async () => {
    return api.get('/api/pipeline/status');
  },

  // Execute the entire pipeline
  executePipeline: async () => {
    return api.post('/api/pipeline/execute');
  },

  // Save a configuration
  saveConfig: async (configData) => {
    return api.post('/api/configs/save', configData);
  },

  // List all saved configurations
  listConfigs: async () => {
    return api.get('/api/configs/list');
  },

  // Get a specific configuration
  getConfig: async (configName) => {
    return api.get(`/api/configs/${configName}`);
  },

  // Record a script run in the history
  recordRun: async (runData) => {
    return api.post('/api/runs/record', runData);
  },

  // Get history of script runs
  getRunHistory: async () => {
    return api.get('/api/runs/history');
  },

  // Check model availability
  checkModelAvailability: async () => {
    return api.get('/api/models/check-availability');
  },

  // Get Docker containers
  getDockerContainers: async () => {
    return api.get('/api/docker/containers');
  },

  // Get vLLM models
  getVLLMModels: async () => {
    return api.get('/api/vllm/models');
  }
};
