import api from './apiService';

// WebSocket connection management
let statusSocket = null;
let resourcesSocket = null;
let statusListeners = [];
let resourcesListeners = [];

export const systemService = {
  // Get current system status
  getSystemStatus: async () => {
    return api.get('/api/system/status');
  },

  // Get available Docker containers directly using vLLM API
  getDockerContainers: async () => {
    try {
      // First try to get containers from backend API
      try {
        const backendContainers = await api.get('/api/docker/containers');
        if (backendContainers && backendContainers.length > 0) {
          console.log('Got containers from backend:', backendContainers);
          // Transform the backend data to match the expected format
          return backendContainers.map(container => ({
            name: container.Name || container.Names || 'Unknown',
            image: container.Image || 'Unknown',
            status: container.State === 'running' ? 'running' : 'stopped',
            type: container.Type || 'Unknown',
            port: container.Port || (container.Ports && container.Ports[0] ? container.Ports[0].PublicPort : null),
            id: container.ID || container.Id
          }));
        }
      } catch (backendError) {
        console.warn('Error fetching Docker containers from backend:', backendError);
      }

      // If backend fails, try direct vLLM API detection
      const containers = [];
      const portsToCheck = [8000, 8001, 8002, 8003]; // Standard ports for vLLM models

      // Try to get models directly from vLLM API on each port
      for (const port of portsToCheck) {
        try {
          console.log(`Checking for vLLM models on port ${port}...`);
          const response = await fetch(`http://localhost:${port}/v1/models`, { signal: AbortSignal.timeout(2000) });
          const data = await response.json();

          if (data && data.data && data.data.length > 0) {
            // Use model names from the API if available
            const modelIds = data.data.map(model => model.id);
            let containerName = modelIds.length > 0 ? modelIds[0] : `vllm_model_${port}`;
            let containerType = "Model";
            let containerImage = "vllm/vllm-openai";

            if (port === 8000) {
              containerType = "Teacher Model";
            } else if (port === 8001) {
              containerType = "Student Model";
            } else if (port === 8002) {
              containerType = "Student Model";
            } else if (port === 8003) {
              containerType = "Distilled Model";
            }

            // Add container info
            containers.push({
              name: containerName,
              image: containerImage,
              status: "running",
              type: containerType,
              port: port,
              models: modelIds
            });

            console.log(`Found vLLM container on port ${port} with models:`, modelIds);
          }
        } catch (error) {
          console.warn(`No vLLM API found on port ${port}:`, error.message);
        }
      }

      // If containers were found, return them
      if (containers.length > 0) {
        console.log('Found vLLM containers via direct detection:', containers);
        return containers;
      }

      // No containers found
      console.log('No Docker containers found');
      return [];
    } catch (error) {
      console.error('Error fetching Docker containers:', error);
      throw error;
    }
  },

  // Get watchdog status
  getWatchdogStatus: async () => {
    return api.get('/api/watchdog/status');
  },

  // Perform watchdog action
  performWatchdogAction: async (action, scriptId) => {
    return api.post('/api/watchdog/action', {
      action,
      script_id: scriptId
    });
  },

  // Get memory usage for a script
  getScriptMemory: async (scriptId) => {
    return api.get(`/api/scripts/${scriptId}/memory`);
  },

  // Connect to status WebSocket for real-time system status updates
  connectToStatusSocket: (onMessage, onError) => {
    // Close existing connection if any
    if (statusSocket && statusSocket.readyState !== WebSocket.CLOSED) {
      statusSocket.close();
    }

    // Create new WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    statusSocket = new WebSocket(`${protocol}//${host}/ws/status`);

    // Set up event handlers
    statusSocket.onopen = () => {
      console.log('Status WebSocket connection established');
    };

    statusSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Notify all listeners
        statusListeners.forEach(listener => listener(data));
        // Call the specific callback if provided
        if (onMessage) onMessage(data);
      } catch (error) {
        console.error('Error parsing status WebSocket message:', error);
      }
    };

    statusSocket.onerror = (error) => {
      console.error('Status WebSocket error:', error);
      if (onError) onError(error);
    };

    statusSocket.onclose = () => {
      console.log('Status WebSocket connection closed');
      // Attempt to reconnect after a delay
      setTimeout(() => {
        if (statusListeners.length > 0) {
          systemService.connectToStatusSocket();
        }
      }, 5000);
    };

    // Return the socket for direct access if needed
    return statusSocket;
  },

  // Connect to resources WebSocket for real-time resource monitoring
  connectToResourcesSocket: (onMessage, onError) => {
    // Close existing connection if any
    if (resourcesSocket && resourcesSocket.readyState !== WebSocket.CLOSED) {
      resourcesSocket.close();
    }

    // Create new WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host;
    resourcesSocket = new WebSocket(`${protocol}//${host}/ws/resources`);

    // Set up event handlers
    resourcesSocket.onopen = () => {
      console.log('Resources WebSocket connection established');
    };

    resourcesSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Notify all listeners
        resourcesListeners.forEach(listener => listener(data));
        // Call the specific callback if provided
        if (onMessage) onMessage(data);
      } catch (error) {
        console.error('Error parsing resources WebSocket message:', error);
      }
    };

    resourcesSocket.onerror = (error) => {
      console.error('Resources WebSocket error:', error);
      if (onError) onError(error);
    };

    resourcesSocket.onclose = () => {
      console.log('Resources WebSocket connection closed');
      // Attempt to reconnect after a delay
      setTimeout(() => {
        if (resourcesListeners.length > 0) {
          systemService.connectToResourcesSocket();
        }
      }, 5000);
    };

    // Return the socket for direct access if needed
    return resourcesSocket;
  },

  // Add a status listener
  addStatusListener: (listener) => {
    statusListeners.push(listener);
    // Start the WebSocket connection if it's not already active
    if (!statusSocket || statusSocket.readyState === WebSocket.CLOSED) {
      systemService.connectToStatusSocket();
    }
    return () => {
      // Return a function to remove the listener
      statusListeners = statusListeners.filter(l => l !== listener);
      // Close the connection if there are no more listeners
      if (statusListeners.length === 0 && statusSocket) {
        statusSocket.close();
      }
    };
  },

  // Add a resources listener
  addResourcesListener: (listener) => {
    resourcesListeners.push(listener);
    // Start the WebSocket connection if it's not already active
    if (!resourcesSocket || resourcesSocket.readyState === WebSocket.CLOSED) {
      systemService.connectToResourcesSocket();
    }
    return () => {
      // Return a function to remove the listener
      resourcesListeners = resourcesListeners.filter(l => l !== listener);
      // Close the connection if there are no more listeners
      if (resourcesListeners.length === 0 && resourcesSocket) {
        resourcesSocket.close();
      }
    };
  },

  // Disconnect from all WebSockets
  disconnectAll: () => {
    if (statusSocket) {
      statusSocket.close();
      statusSocket = null;
    }
    if (resourcesSocket) {
      resourcesSocket.close();
      resourcesSocket = null;
    }
    statusListeners = [];
    resourcesListeners = [];
  }
};
