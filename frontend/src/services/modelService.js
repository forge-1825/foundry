import api from './apiService';
import axios from 'axios';

export const modelService = {
  // Query a model with a prompt directly using vLLM API
  queryModel: async (modelPath, prompt, maxTokens = 500, temperature = 0.7) => {
    try {
      // Check if the model path is a port specification
      if (modelPath.startsWith('port:')) {
        const port = modelPath.split(':')[1];
        const vllmUrl = `http://localhost:${port}/v1/completions`;

        // First, try to get available models to determine the correct model ID
        let modelId = '/model'; // Default model ID
        try {
          const modelsResponse = await axios.get(`http://localhost:${port}/v1/models`);
          if (modelsResponse.data && modelsResponse.data.data && modelsResponse.data.data.length > 0) {
            modelId = modelsResponse.data.data[0].id;
          }
        } catch (modelError) {
          console.warn(`Could not get models from port ${port}, using default model ID:`, modelError);
        }

        // Query the vLLM API directly
        console.log(`Querying vLLM API directly at ${vllmUrl} with model ${modelId}`);
        const vllmResponse = await axios.post(vllmUrl, {
          model: modelId,
          prompt: prompt,
          max_tokens: maxTokens,
          temperature: temperature
        });

        if (vllmResponse.data && vllmResponse.data.choices && vllmResponse.data.choices.length > 0) {
          return { response: vllmResponse.data.choices[0].text };
        } else {
          throw new Error('No response from vLLM API');
        }
      } else {
        // Fall back to the backend API for non-port model paths
        const response = await api.post('/api/model/query', {
          model_path: modelPath,
          prompt: prompt,
          max_tokens: maxTokens,
          temperature: temperature
        });
        return response;
      }
    } catch (error) {
      console.error('Error querying model:', error);
      throw error;
    }
  },

  // Get available models directly from vLLM API
  getAvailableModels: async () => {
    try {
      const models = [];
      const portsToCheck = [8000, 8001, 8002]; // Standard ports for vLLM models

      // Try to get models directly from vLLM API on each port
      for (const port of portsToCheck) {
        try {
          console.log(`Checking for vLLM models on port ${port}...`);
          const response = await axios.get(`http://localhost:${port}/v1/models`);

          if (response.data && response.data.data && response.data.data.length > 0) {
            // Determine model type based on port
            let modelType = "Unknown Model";
            if (port === 8000) modelType = "Teacher Model";
            if (port === 8001) modelType = "Teacher Model (WhiteRabbitNeo)";
            if (port === 8002) modelType = "Student Model";

            // Add each model from this port
            for (const model of response.data.data) {
              const modelName = model.id.includes('/') ? model.id.split('/').pop() : model.id;
              models.push({
                name: `${modelName} (port ${port})`,
                path: `port:${port}`,
                description: `${modelType} running on port ${port}`,
                type: modelType,
                port: port,
                modelId: model.id
              });
            }

            console.log(`Found ${response.data.data.length} models on port ${port}`);
          }
        } catch (error) {
          console.warn(`No vLLM API found on port ${port}:`, error.message);
        }
      }

      // If models were found, return them
      if (models.length > 0) {
        console.log('Found vLLM models:', models);
        return models;
      }

      // Try to get models from backend API as fallback
      try {
        // First, try to get vLLM models from backend
        const response = await api.get('/api/vllm/models');
        if (response && response.length > 0) {
          console.log('Found vLLM models from backend API:', response);
          return response;
        }
      } catch (vllmError) {
        console.warn('Error fetching vLLM models from backend:', vllmError);
      }

      // If no vLLM models, try to get Docker containers from backend
      try {
        const containers = await api.get('/api/docker/containers');
        if (containers && containers.length > 0) {
          // Filter for containers with ports
          const modelContainers = containers.filter(container => container.port);

          if (modelContainers.length > 0) {
            console.log('Found Docker containers with ports:', modelContainers);
            return modelContainers.map(container => ({
              name: container.name,
              path: `port:${container.port}`,
              description: `${container.type || 'Model'} on port ${container.port}`,
              type: container.type || 'Unknown',
              port: container.port
            }));
          }
        }
      } catch (dockerError) {
        console.warn('Error fetching Docker containers:', dockerError);
      }

      // Fallback to hardcoded models
      console.log('Using fallback hardcoded models');
      return [
        {
          name: "Student Model (port 8002)",
          path: "port:8002",
          description: "Student model running on port 8002",
          type: "Student Model",
          port: 8002
        },
        {
          name: "Teacher Model (port 8000)",
          path: "port:8000",
          description: "Teacher model running on port 8000",
          type: "Teacher Model",
          port: 8000
        }
      ];
    } catch (error) {
      console.error('Error fetching available models:', error);
      throw error;
    }
  }
};

export default modelService;
