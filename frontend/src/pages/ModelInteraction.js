import React, { useState, useEffect, useContext } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Textarea } from '../components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select';
import { Alert, AlertTitle, AlertDescription } from '../components/ui/alert';
import { Loader2, AlertCircle, CheckCircle2, RefreshCw } from 'lucide-react';
import { SystemContext } from '../contexts/SystemContext';
import { modelService } from '../services/modelService';
import { systemService } from '../services/systemService';

const ModelInteraction = () => {
  const [prompt, setPrompt] = useState('');
  const [response, setResponse] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dockerContainers, setDockerContainers] = useState([]);
  const [maxTokens, setMaxTokens] = useState(500);
  const [temperature, setTemperature] = useState(0.7);
  const { systemStatus } = useContext(SystemContext);

  // Function to fetch available models
  const fetchModels = async () => {
    try {
      setIsLoading(true);
      const models = await modelService.getAvailableModels();
      setAvailableModels(models);
      if (models.length > 0 && !selectedModel) {
        setSelectedModel(models[0].path);
      }
      setError(null);
    } catch (err) {
      console.error('Error fetching models:', err);
      setError('Failed to load available models');
    } finally {
      setIsLoading(false);
    }
  };

  // Function to fetch Docker containers
  const fetchDockerContainers = async () => {
    try {
      const containers = await systemService.getDockerContainers();
      setDockerContainers(containers);
    } catch (err) {
      console.error('Error fetching Docker containers:', err);
    }
  };

  // Fetch available models and Docker containers on component mount
  useEffect(() => {
    fetchModels();
    fetchDockerContainers();

    // Set up polling for Docker container status
    const intervalId = setInterval(fetchDockerContainers, 10000);
    return () => clearInterval(intervalId);
  }, []);

  const handlePromptChange = (e) => {
    setPrompt(e.target.value);
  };

  const handleModelChange = (value) => {
    setSelectedModel(value);
  };

  const handleMaxTokensChange = (e) => {
    setMaxTokens(parseInt(e.target.value, 10));
  };

  const handleTemperatureChange = (e) => {
    setTemperature(parseFloat(e.target.value));
  };

  const handleSubmit = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    if (!selectedModel) {
      setError('Please select a model');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResponse('');

    try {
      console.log(`Querying model ${selectedModel} with max_tokens=${maxTokens}, temperature=${temperature}`);
      const result = await modelService.queryModel(selectedModel, prompt, maxTokens, temperature);
      setResponse(result.response);
    } catch (err) {
      console.error('Error querying model:', err);
      setError('Failed to get response from model: ' + (err.message || 'Unknown error'));
    } finally {
      setIsLoading(false);
    }
  };

  const getContainerStatus = (containerName) => {
    const container = dockerContainers.find(c => c.name === containerName);
    if (!container) {
      return { status: 'not-found', label: 'Not Found' };
    }

    if (container.status === 'running') {
      return { status: 'running', label: 'Connected' };
    } else {
      return { status: 'stopped', label: 'Disconnected' };
    }
  };

  const phi4Status = getContainerStatus('phi4_gptq_vllm');
  const whiterabbitStatus = getContainerStatus('whiterabbitneo_vllm');

  return (
    <div className="container mx-auto py-6 space-y-6">
      <h1 className="text-3xl font-bold">Model Interaction</h1>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Chat with Model</CardTitle>
              <CardDescription>
                Select a model and enter a prompt to get a response
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <label className="block text-sm font-medium mb-1">Select Model</label>
                <Select value={selectedModel} onValueChange={handleModelChange}>
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select a model" />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((model) => (
                      <SelectItem key={model.path} value={model.path}>
                        {model.name} {model.port ? `(Port ${model.port})` : ''}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="flex justify-between items-center mt-2">
                  {selectedModel && (
                    <div className="text-sm text-gray-500">
                      {availableModels.find(m => m.path === selectedModel)?.description || 'No description available'}
                    </div>
                  )}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={fetchModels}
                    disabled={isLoading}
                    title="Refresh model list"
                  >
                    <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                  </Button>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium mb-1">Prompt</label>
                <Textarea
                  placeholder="Enter your prompt here..."
                  value={prompt}
                  onChange={handlePromptChange}
                  rows={5}
                  className="w-full"
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium mb-1">Max Tokens</label>
                  <div className="flex items-center">
                    <input
                      type="range"
                      min="10"
                      max="2000"
                      step="10"
                      value={maxTokens}
                      onChange={handleMaxTokensChange}
                      className="w-full mr-2"
                    />
                    <span className="text-sm font-medium w-12 text-right">{maxTokens}</span>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-1">Temperature</label>
                  <div className="flex items-center">
                    <input
                      type="range"
                      min="0"
                      max="1"
                      step="0.1"
                      value={temperature}
                      onChange={handleTemperatureChange}
                      className="w-full mr-2"
                    />
                    <span className="text-sm font-medium w-12 text-right">{temperature.toFixed(1)}</span>
                  </div>
                </div>
              </div>

              <Button
                onClick={handleSubmit}
                disabled={isLoading || !selectedModel}
                className="w-full"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Processing...
                  </>
                ) : 'Submit'}
              </Button>

              {error && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {response && (
                <div>
                  <label className="block text-sm font-medium mb-1">Response</label>
                  <div className="p-4 border rounded-md bg-muted whitespace-pre-wrap">
                    {response}
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div>
          <Card>
            <CardHeader>
              <CardTitle>System Status</CardTitle>
              <CardDescription>
                Status of required Docker containers
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                {dockerContainers.length > 0 ? (
                  dockerContainers
                    .filter(container => container.type && container.type.toLowerCase().includes('model'))
                    .map((container, index) => (
                      <div key={index} className="flex items-center justify-between">
                        <span title={container.name}>
                          {container.type || container.name}
                          {container.port && <span className="text-xs text-gray-500 ml-1">(Port {container.port})</span>}
                        </span>
                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                          container.status === 'running'
                            ? 'bg-green-100 text-green-800'
                            : 'bg-red-100 text-red-800'
                        }`}>
                          {container.status === 'running' ? (
                            <CheckCircle2 className="mr-1 h-3 w-3" />
                          ) : (
                            <AlertCircle className="mr-1 h-3 w-3" />
                          )}
                          {container.status === 'running' ? 'Connected' : 'Disconnected'}
                        </span>
                      </div>
                    ))
                ) : (
                  <div className="flex flex-col items-center justify-center py-4">
                    <AlertCircle className="h-8 w-8 text-yellow-500 mb-2" />
                    <p className="text-center text-sm">No model containers detected</p>
                    <Button
                      variant="outline"
                      size="sm"
                      className="mt-2"
                      onClick={() => systemService.getDockerContainers().then(setDockerContainers)}
                    >
                      Refresh
                    </Button>
                  </div>
                )}
              </div>

              {systemStatus && (
                <div className="space-y-2 pt-4 border-t">
                  <div className="flex items-center justify-between">
                    <span>CPU Usage:</span>
                    <span>{systemStatus.cpu_percent?.toFixed(1)}%</span>
                  </div>

                  <div className="flex items-center justify-between">
                    <span>Memory Usage:</span>
                    <span>{systemStatus.memory_percent?.toFixed(1)}%</span>
                  </div>

                  {systemStatus.gpu_info && systemStatus.gpu_info.length > 0 && (
                    <div className="flex items-center justify-between">
                      <span>GPU Memory:</span>
                      <span>{systemStatus.gpu_info[0].memory_percent?.toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default ModelInteraction;
